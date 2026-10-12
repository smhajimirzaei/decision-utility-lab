"""Paired 10/50-action opportunity audit, isolated from existing scorers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import metaworld
import numpy as np
import torch

from .branching import capture, restore
from .cadence_contract import forecast_valid_mask, prefix_bank, trace_prefix_success
from .jepa import grouped_actions, load_metaworld
from .pilot import check_branch, TASKS
from .repeated_outcomes import continuation_seeds, summarize_repeats
from .shared_pilot import (
    SharedEnvironment,
    continuation,
    policy_chunk,
    reset_policy,
    wm_observation,
)
from .vla_pilot import load_policy


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def replay_errors(first, second):
    keys = (
        "actions",
        "rewards",
        "outcomes",
        "final_observation",
        "final_integration",
        "final_vla_frame",
        "final_wm_frame",
    )
    errors = {
        key: (
            float(np.max(np.abs(first[key].astype(float) - second[key].astype(float))))
            if first[key].shape == second[key].shape
            else float("inf")
        )
        for key in keys
    }
    if any(value != 0 for value in errors.values()):
        raise RuntimeError(f"Paired continuation failed exact replay: {errors}")
    return errors


def physical_future(shared, snapshot, initial_obs, bank):
    """Capture only pre-stop/terminal physical frames, padding masked tails."""
    images, proprios, stop_steps, early_flags = [], [], [], []
    for prefix in bank:
        restore(shared.env, snapshot)
        shared.obs = initial_obs.copy()
        initial_frame = shared.frame("wm")
        candidate_images, candidate_proprios, flags = [], [], []
        for index, action in enumerate(prefix):
            _, term, trunc, success = shared.step(action)
            flags.append((term, trunc, success))
            if (index + 1) % 5 == 0:
                candidate_images.append(shared.frame("wm"))
                candidate_proprios.append(shared.obs[:4].copy())
            if term or trunc or success:
                break
        stop_steps.append(len(flags))
        early_flags.append(np.asarray(flags, dtype=bool))
        last_image = candidate_images[-1] if candidate_images else initial_frame
        last_proprio = candidate_proprios[-1] if candidate_proprios else initial_obs[:4]
        while len(candidate_images) < 10:
            candidate_images.append(last_image.copy())
            candidate_proprios.append(last_proprio.copy())
        images.append(candidate_images)
        proprios.append(candidate_proprios)
    return (
        np.asarray(images),
        np.asarray(proprios, dtype=np.float32),
        np.asarray(stop_steps),
        early_flags,
    )


def audit(
    policy,
    preprocess,
    postprocess,
    world,
    wm_preprocessor,
    shared,
    task,
    seed,
    step,
    output,
    config,
):
    started = time.perf_counter()
    folder = output / f"{task}_seed{seed}" / f"state{step}"
    (folder / "features").mkdir(parents=True)
    (folder / "outcomes").mkdir()
    initial_obs = shared.obs.copy()
    snapshot = capture(shared.env)
    initial_wm = shared.frame("wm")
    initial_vla = shared.frame("vla")
    imageio.imwrite(folder / "features/current_wm.png", initial_wm)
    imageio.imwrite(folder / "features/current_vla.png", initial_vla)
    simulator_replay = check_branch(
        shared.env, initial_obs.copy(), getattr(metaworld.policies, TASKS[task])()
    )
    shared.obs = initial_obs.copy()
    if not simulator_replay["passed"] or not np.array_equal(
        initial_wm, shared.frame("wm")
    ):
        raise RuntimeError("Simulator/auxiliary camera replay failed")
    proposal_seed = 100000 + seed * 1000 + step
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(proposal_seed)
        reset_policy(policy, preprocess, postprocess)
        raw = np.stack(
            [
                policy_chunk(policy, preprocess, postprocess, shared)
                for _ in range(config["candidate_count"])
            ]
        )
    banks = {length: prefix_bank(raw, length) for length in config["prefix_lengths"]}
    if not np.array_equal(banks[10], banks[50][:, :10]):
        raise RuntimeError("Paired conditions do not share proposals")
    # Predictions precede all future/outcome capture; no value scorer is applied.
    with torch.inference_mode():
        context = world.encode(wm_observation(shared))
        torch.cuda.synchronize()
        forecast_started = time.perf_counter()
        long_prediction = world.unroll(
            context, grouped_actions(banks[50], wm_preprocessor, "cuda:0")
        )
        torch.cuda.synchronize()
        long_seconds = time.perf_counter() - forecast_started
        forecast_started = time.perf_counter()
        short_prediction = world.unroll(
            context, grouped_actions(banks[10], wm_preprocessor, "cuda:0")
        )
        torch.cuda.synchronize()
        short_seconds = time.perf_counter() - forecast_started
        errors = {}
        for key in ("visual", "proprio"):
            expected = short_prediction[key]
            observed = long_prediction[key][: len(expected)]
            if (
                expected.shape != observed.shape
                or len(long_prediction[key]) != 11
                or len(expected) != 3
            ):
                raise RuntimeError("Unexpected autoregressive prefix alignment")
            if not torch.isfinite(long_prediction[key]).all():
                raise RuntimeError("Nonfinite long forecast")
            errors[key] = float((expected - observed).abs().max().item())
            if errors[key] > config["forecast_prefix_equivalence_atol"]:
                raise RuntimeError(
                    f"Independent short/long forecasts disagree: {errors}"
                )
        predicted = long_prediction["visual"][1:].transpose(0, 1)
        predicted_proprio = long_prediction["proprio"][1:].transpose(0, 1)
    np.savez_compressed(
        folder / "features/deployment_inputs.npz",
        raw_proposals=raw,
        executed_full_proposals=banks[50],
        proprio=initial_obs[:4],
        current_latent=context["visual"].cpu().numpy(),
        current_proprio_latent=context["proprio"].cpu().numpy(),
        predicted_latents=predicted.cpu().numpy(),
        predicted_proprio_latents=predicted_proprio.cpu().numpy(),
    )
    real_images, real_props, stop_steps, prefix_flags = physical_future(
        shared, snapshot, initial_obs, banks[50]
    )
    valid = forecast_valid_mask(stop_steps)
    with torch.inference_mode():
        # Encode one candidate at a time to keep the larger observation audit bounded.
        actual_chunks = [
            world.encode(
                {
                    "visual": torch.from_numpy(real_images[i : i + 1]).permute(
                        0, 1, 4, 2, 3
                    ),
                    "proprio": torch.from_numpy(real_props[i : i + 1]),
                }
            )
            for i in range(len(raw))
        ]
        target = torch.cat([a["visual"] for a in actual_chunks], dim=0)
        actual_proprio = torch.cat([a["proprio"] for a in actual_chunks], dim=0)
        if predicted.shape != target.shape:
            raise RuntimeError("Misaligned physical future and model forecast")
        mse = (predicted - target).square().flatten(2).mean(-1).cpu().numpy()
        persistence = (
            (context["visual"][:, -1:] - target)
            .square()
            .flatten(2)
            .mean(-1)
            .cpu()
            .numpy()
        )
        if not np.isfinite(mse).all() or not np.isfinite(persistence).all():
            raise RuntimeError("Nonfinite forecast errors")
    np.savez_compressed(
        folder / "outcomes/actual_future_features.npz",
        latent=target.cpu().numpy(),
        proprio=real_props,
        proprio_latent=actual_proprio.cpu().numpy(),
        valid_mask=valid,
        first_stop_or_limit=stop_steps,
    )
    seeds = continuation_seeds(seed, step, config["continuation_repeats"])
    conditions = []
    for length in config["prefix_lengths"]:
        condition_dir = folder / "outcomes" / f"prefix{length}"
        condition_dir.mkdir()
        repeat_outcomes, replays = [], []
        for repeat_index, repeat_seed in enumerate(seeds):
            branches = []
            for candidate, prefix in enumerate(banks[length]):
                branch = continuation(
                    policy,
                    preprocess,
                    postprocess,
                    shared,
                    snapshot,
                    initial_obs,
                    prefix,
                    400 - step,
                    repeat_seed,
                )
                prefix_count = min(length, len(branch["actions"]))
                if not np.array_equal(
                    branch["actions"][:prefix_count], prefix[:prefix_count]
                ):
                    raise RuntimeError(
                        "Trace actions differ from declared proposal prefix"
                    )
                physical_count = min(length, len(prefix_flags[candidate]))
                if not np.array_equal(
                    branch["outcomes"][:physical_count],
                    prefix_flags[candidate][:physical_count],
                ):
                    raise RuntimeError(
                        "Repeated prefix outcomes differ from physical audit"
                    )
                np.savez_compressed(
                    condition_dir
                    / f"candidate{candidate}_repeat{repeat_index}_trace.npz",
                    **{
                        key: branch[key]
                        for key in (
                            "actions",
                            "rewards",
                            "outcomes",
                            "final_observation",
                            "final_integration",
                        )
                    },
                )
                branches.append(branch)
            replay = continuation(
                policy,
                preprocess,
                postprocess,
                shared,
                snapshot,
                initial_obs,
                banks[length][0],
                400 - step,
                repeat_seed,
            )
            replays.append(replay_errors(branches[0], replay))
            repeat_outcomes.append(
                [
                    {
                        key: b[key]
                        for key in ("success", "final_success", "steps", "reward_sum")
                    }
                    for b in branches
                ]
            )
            print(
                f"CADENCE_REPEAT task={task} seed={seed} step={step} prefix={length} repeat={repeat_index} replay=True",
                flush=True,
            )
        early = [trace_prefix_success(flags, length) for flags in prefix_flags]
        condition = {
            "execution_prefix": length,
            "continuation_chunk": 50,
            "remaining_horizon": 400 - step,
            "continuation_seeds": seeds,
            "branch_repeats": repeat_outcomes,
            "continuation_replay_by_repeat": replays,
            "early_success_within_prefix": early,
            "scorer_training_eligible": [not x for x in early],
            **summarize_repeats(repeat_outcomes),
        }
        write_json(condition_dir / "labels.json", condition)
        conditions.append(condition)
    record = {
        "episode_id": f"{task}_seed{seed}",
        "task": task,
        "seed": seed,
        "state_step": step,
        "instruction": shared.instruction,
        "candidate_count": len(raw),
        "proposal_seed": proposal_seed,
        "proposal_sha256": hashlib.sha256(banks[50].tobytes()).hexdigest(),
        "simulator_replay": simulator_replay,
        "forecast_prefix_max_abs_error": errors,
        "forecast_seconds": {"10": short_seconds, "50": long_seconds},
        "model_mse": [
            [float(v) if ok else None for v, ok in zip(row, mask)]
            for row, mask in zip(mse, valid)
        ],
        "persistence_mse": [
            [float(v) if ok else None for v, ok in zip(row, mask)]
            for row, mask in zip(persistence, valid)
        ],
        "forecast_valid_mask": valid.tolist(),
        "first_stop_or_limit": stop_steps.tolist(),
        "conditions": conditions,
        "seconds_including_branch_audit": time.perf_counter() - started,
        "scope": "Matched one-intervention development outcomes; no value head or closed-loop selector",
    }
    write_json(folder / "outcomes/state.json", record)
    write_json(
        folder / "features/schema.json",
        {
            "instruction": shared.instruction,
            "prefix_lengths": [10, 50],
            "proprio_dimensions": 4,
            "world_transitions": 10,
            "world_frameskip": 5,
            "forbidden_inputs": [
                "reward",
                "success",
                "full observation",
                "goal coordinates",
                "branch labels",
                "actual futures",
            ],
            "padded_actual_frames": "Offline only; saved mask excludes after-stop frames from metrics",
        },
    )
    restore(shared.env, snapshot)
    shared.obs = initial_obs.copy()
    return record


def episode(
    policy,
    preprocess,
    postprocess,
    world,
    wm_preprocessor,
    task,
    seed,
    output,
    steps,
    config,
):
    shared = SharedEnvironment(task, seed)
    folder = output / f"{task}_seed{seed}"
    folder.mkdir()
    reset_policy(policy, preprocess, postprocess)
    torch.manual_seed(seed)
    started = time.perf_counter()
    step = 0
    success = False
    reward_sum = 0.0
    states = []
    actions = []
    frames = [shared.frame("vla")]
    imageio.imwrite(folder / "initial.png", frames[0])
    try:
        while step < 400:
            if step in steps:
                with torch.random.fork_rng(devices=[0]):
                    record = audit(
                        policy,
                        preprocess,
                        postprocess,
                        world,
                        wm_preprocessor,
                        shared,
                        task,
                        seed,
                        step,
                        output,
                        config,
                    )
                reset_policy(policy, preprocess, postprocess)
                states.append(record)
                print(
                    f'CADENCE_STATE task={task} seed={seed} step={step} seconds={record["seconds_including_branch_audit"]:.2f}',
                    flush=True,
                )
            chunk = policy_chunk(policy, preprocess, postprocess, shared)
            for action in np.clip(chunk, -1, 1):
                reward, term, trunc, flag = shared.step(action)
                reward_sum += reward
                actions.append(action.copy())
                success |= flag
                step += 1
                if step % 10 == 0 or success or step == 400:
                    frames.append(shared.frame("vla"))
                if success or term or trunc or step >= 400:
                    break
            if success or term or trunc:
                break
        result = {
            "task": task,
            "seed": seed,
            "success": success,
            "steps": step,
            "reward_sum": reward_sum,
            "seconds_including_branch_audits": time.perf_counter() - started,
            "baseline_execution_chunk": 50,
            "states": states,
        }
        np.save(folder / "baseline_executed_actions.npy", np.asarray(actions))
        imageio.mimsave(
            folder / "baseline_actual_execution.mp4", frames, fps=8, macro_block_size=1
        )
        write_json(folder / "result.json", result)
        print(
            f"CADENCE_EPISODE task={task} seed={seed} states={len(states)} success={success}",
            flush=True,
        )
        return result
    finally:
        shared.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config", type=Path, default=Path("configs/cadence_development.json")
    )
    p.add_argument("--stage", choices=["pilot", "main"], required=True)
    p.add_argument("--episode-seed", type=int)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.stage == "pilot":
        if args.episode_seed is not None:
            p.error("Pilot has a fixed complete seed grid")
        seeds = config["pilot_seeds"]
        steps = config["pilot_decision_steps"]
    else:
        if args.episode_seed not in config["main_seeds"]:
            p.error("Main shard must use one declared seed")
        seeds = [args.episode_seed]
        steps = config["main_decision_steps"]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "collection_protocol.json", config)
    result = {
        "status": "running",
        "stage": args.stage,
        "episodes": [],
        "tasks": config["tasks"],
        "source_seeds": seeds,
        "decision_steps": steps,
        "prefix_lengths": config["prefix_lengths"],
        "candidate_count": config["candidate_count"],
        "continuation_repeats": config["continuation_repeats"],
        "value_scorer": None,
        "scope": "Development-only opportunity/forecast diagnostic; no fit, gate or final test",
    }
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        policy, preprocess, postprocess, policy_report = load_policy(
            Path("external/lerobot"),
            json.loads(Path("configs/smolvla_metaworld_manifest.json").read_text()),
            Path("checkpoints/vla_hf"),
            "cuda",
        )
        world, wm_preprocessor, world_report = load_metaworld(
            Path("external/jepa-wms"),
            Path("configs/jepa_metaworld_manifest.json"),
            Path("checkpoints/hf"),
            "cuda:0",
        )
        for name, report in (("vla", policy_report), ("jepa", world_report)):
            write_json(args.output / f"{name}_manifest.json", report)
        result["package_versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "metaworld",
                "mujoco",
                "gymnasium",
                "torch",
                "transformers",
                "lerobot",
            )
        }
        for task in config["tasks"]:
            for seed in seeds:
                result["episodes"].append(
                    episode(
                        policy,
                        preprocess,
                        postprocess,
                        world,
                        wm_preprocessor,
                        task,
                        seed,
                        args.output,
                        steps,
                        config,
                    )
                )
                write_json(args.output / "progress.json", result)
        result["status"] = "passed"
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", result)
    print("CADENCE_STAGE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
