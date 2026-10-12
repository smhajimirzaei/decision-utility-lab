"""Small published CEM setting, paired with consistent raw-action conditioning.

Single-seed feasibility evidence, not a paper-level reproduction or VLA test.
Uses official CEM/objective classes and the pinned pretrained model unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from .branching import capture, restore
from .jepa import load_metaworld


def make_environment(seed):
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    from metaworld.policies import SawyerReachWallV3Policy

    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE["reach-wall-v3-goal-observable"](
        seed=seed
    )
    env.seeded_rand_vec = False
    env._freeze_rand_vec = False
    env.render_mode, env.camera_name = "rgb_array", "corner2"
    env.width = env.height = 224
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")
    if camera_id != 2:
        raise RuntimeError("Unexpected source camera mapping")
    env.model.cam_pos[camera_id] = [0.75, 0.075, 0.7]
    previous = env.mujoco_renderer
    renderer_options = {
        "default_cam_config": previous.default_cam_config,
        "max_geom": previous.max_geom,
    }
    previous.close()
    env.mujoco_renderer = MujocoRenderer(
        env.model,
        env.data,
        width=224,
        height=224,
        camera_name="corner2",
        **renderer_options,
    )
    return env, SawyerReachWallV3Policy()


def pixels(env):
    frame = env.render()[::-1].copy()
    if frame.shape != (224, 224, 3) or frame.max() == frame.min():
        raise RuntimeError("Empty/constant camera render")
    return frame


def observation(env, obs):
    return {
        "visual": torch.from_numpy(pixels(env))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0),
        "proprio": torch.from_numpy(obs[:4].astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0),
    }


def episode(model, preprocessor, output, seed, modes):
    # Imports succeed only after the pinned loader prepends the upstream source.
    from evals.simu_env_planning.planning.planning.objectives import (
        ReprTargetDistMPCObjective,
    )
    from evals.simu_env_planning.planning.planning.planner import CEMPlanner

    env, expert = make_environment(seed)
    episode_dir = output / f"seed{seed}"
    episode_dir.mkdir()
    try:
        obs, _ = env.reset(seed=seed)
        # Match the source warmup (one zero action for one observed frame).
        obs, *_ = env.step(np.zeros(4))
        start_state, initial_obs = capture(env), obs.copy()
        imageio.imwrite(episode_dir / "initial.png", pixels(env))
        expert_success = False
        # The Meta-World evaluator uses the final expert image, not goal_H*5.
        for _ in range(100):
            obs, _, _, _, info = env.step(np.clip(expert.get_action(obs), -1, 1))
            expert_success |= bool(info.get("success", 0))
        goal_obs = observation(env, obs)
        imageio.imwrite(episode_dir / "goal.png", pixels(env))
        with torch.inference_mode():
            goal_encoding = model.encode(goal_obs)
        records = []
        mean = preprocessor.action_mean.to("cuda:0")
        std = preprocessor.action_std.to("cuda:0")
        for mode in modes:
            restore(env, start_state)
            obs = initial_obs.copy()
            generator = torch.Generator(device="cuda").manual_seed(seed)
            audit = {"sampled_raw_elements": 0, "sampled_outside_bounds": 0}

            def audited_unroll(context, act_suffix):
                raw = act_suffix.reshape(*act_suffix.shape[:2], 5, 4) * std + mean
                audit["sampled_raw_elements"] += raw.numel()
                audit["sampled_outside_bounds"] += int((raw.abs() > 1).sum())
                conditioned = act_suffix
                if mode == "consistent":
                    conditioned = ((raw.clamp(-1, 1) - mean) / std).reshape_as(
                        act_suffix
                    )
                return model.unroll(context, conditioned)

            planner = CEMPlanner(
                unroll=audited_unroll,
                action_dim=20,
                iterations=15,
                num_samples=300,
                num_elites=10,
                horizon=6,
                var_scale=1.0,
                num_act_stepped=3,
                local_generator=generator,
                distribute_planner=False,
                decode_each_iteration=False,
            )
            planner.set_objective(
                ReprTargetDistMPCObjective(
                    {}, target_enc=goal_encoding, sum_all_diffs=False, alpha=0.1
                )
            )
            mode_dir = episode_dir / mode
            mode_dir.mkdir()
            steps, reward_sum, ever_success = 1, 0.0, False
            plans, executed_actions, frames = [], [], [pixels(env)]
            started = time.perf_counter()
            while steps < 100:
                with torch.inference_mode():
                    context = model.encode(observation(env, obs))
                    torch.cuda.synchronize()
                    plan_start = time.perf_counter()
                    plan = planner.plan(
                        context, steps_left=max((100 - steps + 1) // 5, 1)
                    )
                    torch.cuda.synchronize()
                    plan_seconds = time.perf_counter() - plan_start
                    raw = (plan.actions.reshape(-1, 4) * std + mean).cpu().numpy()
                outside = int(np.sum(np.abs(raw) > 1))
                if not np.isfinite(raw).all():
                    raise RuntimeError("Nonfinite selected action")
                # This clipping is exactly what env.step does, made explicit.
                executed = np.clip(raw, -1, 1)
                planned_step = steps
                taken = 0
                for action in executed:
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_sum += float(reward)
                    ever_success |= bool(info.get("success", 0))
                    steps += 1
                    taken += 1
                    executed_actions.append(action.copy())
                    if taken % 5 == 0 or steps == 100:
                        frames.append(pixels(env))
                    if steps >= 100 or terminated or truncated:
                        break
                entry = {
                    "environment_step": planned_step,
                    "plan_seconds": plan_seconds,
                    "actions_executed": taken,
                    "selected_raw_outside_bounds": outside,
                    "selected_raw_elements": raw.size,
                    "minimum_cost_per_iteration": plan.losses.cpu().flatten().tolist(),
                    "success_after_execution": bool(info.get("success", 0)),
                }
                plans.append(entry)
                print(
                    f"PLAN_COMPLETE seed={seed} mode={mode} step={steps} seconds={plan_seconds:.2f} success={entry['success_after_execution']}",
                    flush=True,
                )
                (mode_dir / "progress.json").write_text(
                    json.dumps({"plans": plans, "audit": audit}, indent=2) + "\n"
                )
                if terminated or truncated:
                    break
            record = {
                "seed": seed,
                "mode": mode,
                "steps": steps,
                "ever_success": ever_success,
                "final_success": bool(info.get("success", 0)),
                "reward_sum": reward_sum,
                "expert_goal_ever_success": expert_success,
                "seconds": time.perf_counter() - started,
                "plans": plans,
                "audit": audit,
            }
            audit["sampled_raw_outside_fraction"] = audit[
                "sampled_outside_bounds"
            ] / max(audit["sampled_raw_elements"], 1)
            np.save(mode_dir / "executed_actions.npy", np.asarray(executed_actions))
            imageio.mimsave(
                mode_dir / "actual_execution.mp4", frames, fps=6, macro_block_size=1
            )
            imageio.imwrite(mode_dir / "final.png", frames[-1])
            (mode_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(
                f"EPISODE_COMPLETE seed={seed} mode={mode} ever_success={ever_success} seconds={record['seconds']:.2f}",
                flush=True,
            )
        return records
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["published", "consistent"],
        default=["published", "consistent"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "stage": "M1_small_CEM_feasibility",
        "status": "running",
        "task": "reach-wall-v3",
        "scope": "single-seed image-goal control; no VLA and no statistical performance claim",
        "source_setting": {
            "horizon": 6,
            "samples": 300,
            "iterations": 15,
            "elites": 10,
            "executed_groups": 3,
            "frameskip": 5,
            "alpha": 0.1,
            "max_steps": 100,
        },
        "deviations": [
            "single seed instead of 48 episodes",
            "optional decoder disabled",
            "direct environment adapter and simulator restore instead of broad evaluation wrappers",
            "expert goal generated from matched warm-start snapshot",
            "consistent mode clips conditioning to executed raw-action bounds",
        ],
        "episodes": [],
    }
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        model, preprocessor, manifest = load_metaworld(
            args.source, args.manifest, Path("checkpoints/hf"), "cuda:0"
        )
        (args.output / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for seed in args.seeds:
            result["episodes"].extend(
                episode(model, preprocessor, args.output, seed, args.modes)
            )
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "passed"
    except Exception as exc:
        result["status"], result["error"] = "failed", repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("M1_PLANNER_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
