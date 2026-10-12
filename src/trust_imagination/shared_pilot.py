"""Shared-simulator compatibility and paired candidate-outcome instrumentation.

No task scorer or gate is trained here. Candidate outcomes are offline labels,
never model inputs. Continuations reset the stateless frozen policy and use
matched explicit Torch seeds. The baseline retains 50-action execution.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import time

import imageio.v2 as imageio
import metaworld
import mujoco
import numpy as np
import torch

from .branching import capture, restore, integration_state
from .jepa import grouped_actions, load_metaworld
from .pilot import check_branch, TASKS
from .vla_pilot import load_policy


class SharedEnvironment:
    def __init__(self, task, seed):
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        from lerobot.envs.metaworld import TASK_DESCRIPTIONS

        random.seed(seed)
        np.random.seed(seed)
        benchmark = metaworld.MT1(task, seed=42)
        self.env = benchmark.train_classes[task](
            render_mode="rgb_array", width=480, height=480, camera_name="corner2"
        )
        self.env.set_task(benchmark.train_tasks[0])
        cam = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")
        if cam != 2:
            raise RuntimeError("Unexpected corner2 camera mapping")
        self.env.model.cam_pos[cam] = [0.75, 0.075, 0.7]
        self.env.reset()
        self.env._freeze_rand_vec = False
        self.env.seeded_rand_vec = True
        self.env.seed(seed)
        self.obs, _ = self.env.reset(seed=seed)
        main = self.env.mujoco_renderer
        self.wm_renderer = MujocoRenderer(
            self.env.model,
            self.env.data,
            width=224,
            height=224,
            camera_name="corner2",
            default_cam_config=main.default_cam_config,
            max_geom=main.max_geom,
        )
        self.instruction = TASK_DESCRIPTIONS[task]

    def frame(self, model):
        if model == "vla":
            raw = self.env.render()
            result = np.flip(raw, (0, 1)).copy()
            expected = (480, 480, 3)
        else:
            # The auxiliary renderer must follow restored MjData, too.
            self.wm_renderer.data = self.env.data
            for viewer in self.wm_renderer._viewers.values():
                viewer.data = self.env.data
            raw = self.wm_renderer.render("rgb_array")
            result = np.flip(raw, 0).copy()
            expected = (224, 224, 3)
        if result.shape != expected or result.max() == result.min():
            raise RuntimeError(f"Invalid {model} camera: {result.shape}")
        return result

    def step(self, action):
        self.obs, reward, terminated, truncated, info = self.env.step(
            np.clip(action, -1, 1)
        )
        return (
            float(reward),
            bool(terminated),
            bool(truncated),
            bool(info.get("success", 0)),
        )

    def close(self):
        self.wm_renderer.close()
        self.env.close()


def policy_chunk(policy, preprocess, postprocess, shared):
    batch = preprocess(
        {
            "observation.state": torch.from_numpy(shared.obs[:4].astype(np.float32)),
            "observation.images.camera1": torch.from_numpy(shared.frame("vla"))
            .permute(2, 0, 1)
            .float()
            / 255,
            "task": shared.instruction,
        }
    )
    if batch["observation.state"].shape != (1, 4):
        raise RuntimeError("Unexpected ordinary sensor shape")
    with torch.inference_mode():
        actions = postprocess(policy.predict_action_chunk(batch)).cpu().numpy()[0]
    if actions.shape != (50, 4) or not np.isfinite(actions).all():
        raise RuntimeError("Invalid VLA proposal")
    return actions


def wm_observation(shared):
    return {
        "visual": torch.from_numpy(shared.frame("wm")).permute(2, 0, 1)[None, None],
        "proprio": torch.from_numpy(shared.obs[:4].astype(np.float32))[None, None],
    }


def reset_policy(policy, preprocess, postprocess):
    policy.reset()
    preprocess.reset()
    postprocess.reset()


def continuation(
    policy, preprocess, postprocess, shared, snapshot, obs, prefix, remaining, seed
):
    restore(shared.env, snapshot)
    shared.obs = obs.copy()
    reset_policy(policy, preprocess, postprocess)
    rewards, outcomes, actions = [], [], []
    endpoint_obs = None
    # No cached base-policy queue crosses branches. Each branch gets identical
    # noise streams for the frozen 50-action continuation after its prefix.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        pending = np.asarray(prefix)
        while len(actions) < remaining:
            for action in np.clip(pending, -1, 1):
                reward, term, trunc, success = shared.step(action)
                rewards.append(reward)
                outcomes.append((term, trunc, success))
                actions.append(action.copy())
                if len(actions) == len(prefix):
                    endpoint_obs = shared.obs[:4].copy()
                if success or term or trunc or len(actions) >= remaining:
                    break
            if (
                outcomes[-1][2]
                or outcomes[-1][0]
                or outcomes[-1][1]
                or len(actions) >= remaining
            ):
                break
            pending = policy_chunk(policy, preprocess, postprocess, shared)
    return {
        "success": any(row[2] for row in outcomes),
        "final_success": outcomes[-1][2],
        "steps": len(actions),
        "reward_sum": float(sum(rewards)),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "outcomes": np.asarray(outcomes),
        "final_observation": shared.obs.copy(),
        "final_integration": integration_state(shared.env),
        "final_vla_frame": shared.frame("vla"),
        "final_wm_frame": shared.frame("wm"),
        "endpoint_proprio": endpoint_obs,
    }


def candidate_audit(
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
    candidate_count,
    continuation_repeats=1,
    readout=None,
):
    folder = output / f"{task}_seed{seed}" / f"state{step}"
    (folder / "features").mkdir(parents=True)
    (folder / "outcomes").mkdir()
    initial_obs = shared.obs.copy()
    snapshot = capture(shared.env)
    first_wm_frame = shared.frame("wm")
    initial_vla_frame = shared.frame("vla")
    imageio.imwrite(folder / "features" / "current_vla.png", initial_vla_frame)
    imageio.imwrite(folder / "features" / "current_wm.png", first_wm_frame)
    expert = getattr(metaworld.policies, TASKS[task])()
    replay = check_branch(shared.env, initial_obs.copy(), expert)
    shared.obs = initial_obs.copy()
    if not replay["passed"] or not np.array_equal(first_wm_frame, shared.frame("wm")):
        raise RuntimeError(
            f"Simulator or auxiliary-camera replay failed: {task}/{seed}/{step}"
        )
    proposal_seed = 100000 + seed * 1000 + step
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(proposal_seed)
        reset_policy(policy, preprocess, postprocess)
        raw_bank = np.stack(
            [
                policy_chunk(policy, preprocess, postprocess, shared)
                for _ in range(candidate_count)
            ]
        )
    # Frozen token embeddings are an explicit instruction feature. This is a
    # lexical conditioning baseline, not a claim of unseen-task generalization.
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    language_batch = preprocess(
        {
            "observation.state": torch.from_numpy(initial_obs[:4].astype(np.float32)),
            "observation.images.camera1": torch.from_numpy(initial_vla_frame)
            .permute(2, 0, 1)
            .float()
            / 255,
            "task": shared.instruction,
        }
    )
    with torch.inference_mode():
        tokens = language_batch[OBS_LANGUAGE_TOKENS]
        mask = language_batch[OBS_LANGUAGE_ATTENTION_MASK].to(torch.float32)
        embeddings = policy.model.vlm_with_expert.embed_language_tokens(tokens)
        instruction_embedding = (embeddings.float() * mask[..., None]).sum(
            1
        ) / mask.sum(1, keepdim=True).clamp_min(1)
    # Ten executed actions, exactly two JEPA steps, equally for every candidate.
    bank = np.clip(raw_bank[:, :10], -1, 1).astype(np.float32)
    real_images, real_props, early_success = [], [], []
    for prefix in bank:
        restore(shared.env, snapshot)
        shared.obs = initial_obs.copy()
        images, props, flags = [], [], []
        for i, action in enumerate(prefix):
            _, _, _, success = shared.step(action)
            flags.append(success)
            if (i + 1) % 5 == 0:
                images.append(shared.frame("wm"))
                props.append(shared.obs[:4].copy())
        real_images.append(images)
        real_props.append(props)
        early_success.append(any(flags))
    restore(shared.env, snapshot)
    shared.obs = initial_obs.copy()
    with torch.inference_mode():
        context = world.encode(wm_observation(shared))
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = world.unroll(
            context, grouped_actions(bank, wm_preprocessor, "cuda:0")
        )
        torch.cuda.synchronize()
        unroll_seconds = time.perf_counter() - started
        actual = world.encode(
            {
                "visual": torch.from_numpy(np.asarray(real_images)).permute(
                    0, 1, 4, 2, 3
                ),
                "proprio": torch.from_numpy(np.asarray(real_props, dtype=np.float32)),
            }
        )
        predicted = prediction["visual"][1:].transpose(0, 1)
        target = actual["visual"]
        if predicted.shape != target.shape or not torch.isfinite(predicted).all():
            raise RuntimeError("Nonfinite or misaligned shared-model prediction")
        mse = (predicted - target).square().flatten(2).mean(-1)
        persistence = (context["visual"][:, -1:] - target).square().flatten(2).mean(-1)
    np.savez_compressed(
        folder / "features" / "deployment_inputs.npz",
        proprio=initial_obs[:4],
        raw_proposals=raw_bank,
        executed_prefixes=bank,
        current_latent=context["visual"].cpu().numpy(),
        predicted_latents=predicted.cpu().numpy(),
        current_proprio_latent=context["proprio"].cpu().numpy(),
        predicted_proprio_latents=prediction["proprio"][1:]
        .transpose(0, 1)
        .cpu()
        .numpy(),
        instruction_embedding=instruction_embedding.cpu().numpy(),
    )
    # Actual future features and branch outcomes are strictly label-side files.
    np.savez_compressed(
        folder / "outcomes" / "actual_future_features.npz",
        latent=target.cpu().numpy(),
        proprio=np.asarray(real_props),
        proprio_latent=actual["proprio"].cpu().numpy(),
    )
    # Frozen predicted-future choice is made before downstream outcome labels.
    readout_scores = None
    if readout is not None:
        predicted_scores = readout.score(
            predicted[:, -1, 0].cpu().numpy(),
            prediction["proprio"][1:].transpose(0, 1)[:, -1].cpu().numpy(),
            instruction_embedding.cpu().numpy(),
            400 - step - 10,
        )
        actual_scores = readout.score(
            target[:, -1, 0].cpu().numpy(),
            actual["proprio"][:, -1].cpu().numpy(),
            instruction_embedding.cpu().numpy(),
            400 - step - 10,
        )
        readout_scores = {
            "predicted_future": predicted_scores.tolist(),
            "actual_future_offline": actual_scores.tolist(),
        }
    from .repeated_outcomes import continuation_seeds, summarize_repeats

    repeat_seeds = continuation_seeds(seed, step, continuation_repeats)
    repeat_outcomes, repeat_replay_errors = [], []
    for repeat_index, continuation_seed in enumerate(repeat_seeds):
        branches = [
            continuation(
                policy,
                preprocess,
                postprocess,
                shared,
                snapshot,
                initial_obs,
                prefix,
                400 - step,
                continuation_seed,
            )
            for prefix in bank
        ]
        repeated = continuation(
            policy,
            preprocess,
            postprocess,
            shared,
            snapshot,
            initial_obs,
            bank[0],
            400 - step,
            continuation_seed,
        )
        replay_errors = {
            key: (
                float(
                    np.max(
                        np.abs(
                            branches[0][key].astype(float) - repeated[key].astype(float)
                        )
                    )
                )
                if branches[0][key].shape == repeated[key].shape
                else float("inf")
            )
            for key in (
                "actions",
                "rewards",
                "outcomes",
                "final_observation",
                "final_integration",
                "final_vla_frame",
                "final_wm_frame",
            )
        }
        if any(value != 0 for value in replay_errors.values()):
            raise RuntimeError(
                f"Stochastic continuation repeat {repeat_index} replay failed: {replay_errors}"
            )
        repeat_replay_errors.append(replay_errors)
        for index, branch in enumerate(branches):
            trace_name = (
                f"candidate{index}_trace.npz"
                if repeat_index == 0
                else f"candidate{index}_repeat{repeat_index}_trace.npz"
            )
            np.savez_compressed(
                folder / "outcomes" / trace_name,
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
        repeat_outcomes.append(
            [
                {
                    key: branch[key]
                    for key in ("success", "final_success", "steps", "reward_sum")
                }
                for branch in branches
            ]
        )
        print(
            f"CONTINUATION_REPEAT task={task} seed={seed} step={step} repeat={repeat_index} replay=True",
            flush=True,
        )
    outcomes = repeat_outcomes[0]
    replay_errors = repeat_replay_errors[0]
    continuation_seed = repeat_seeds[0]
    base_success = outcomes[0]["success"]
    labels = [
        (
            "helps"
            if row["success"] and not base_success
            else (
                "hurts"
                if base_success and not row["success"]
                else "tie_both_success" if base_success else "tie_both_failure"
            )
        )
        for row in outcomes
    ]
    distances = np.sqrt(((bank[1:] - bank[0]) ** 2).mean((1, 2))).tolist()
    record = {
        "episode_id": f"{task}_seed{seed}",
        "task": task,
        "seed": seed,
        "state_step": step,
        "instruction": shared.instruction,
        "simulator_replay": replay,
        "continuation_replay_max_abs_errors": replay_errors,
        "candidate_count": candidate_count,
        "execution_prefix": 10,
        "continuation_chunk": 50,
        "remaining_horizon": 400 - step,
        "proposal_seed": proposal_seed,
        "continuation_seed": continuation_seed,
        "proposal_prefix_rms_distance_from_base": distances,
        "model_mse": mse.cpu().tolist(),
        "persistence_mse": persistence.cpu().tolist(),
        "candidate_bank_two_step_unroll_seconds": unroll_seconds,
        "branches": outcomes,
        "alternative_labels_relative_to_candidate0": labels,
        "early_success_within_prefix": early_success,
        "scorer_training_eligible": [not value for value in early_success],
        "scorer_training_endpoint": "final (10-step) actual latent only; no 5-step value labels",
        "continuation_repeats": continuation_repeats,
        "continuation_seeds": repeat_seeds,
        "branch_repeats": repeat_outcomes,
        "continuation_replay_by_repeat": repeat_replay_errors,
        "readout_scores": readout_scores,
        "readout_checkpoint_sha256": (
            readout.spec["checkpoint_sha256"] if readout is not None else None
        ),
        "world_model_selected_index": (
            int(np.argmax(readout_scores["predicted_future"]))
            if readout_scores is not None
            else None
        ),
        "scope": "matched-seed one-intervention outcomes, not closed-loop control",
        **summarize_repeats(repeat_outcomes),
    }
    (folder / "outcomes" / "labels.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    (folder / "features" / "schema.json").write_text(
        json.dumps(
            {
                "instruction": shared.instruction,
                "candidate_count": candidate_count,
                "execution_prefix": 10,
                "proprio_dimensions": 4,
                "instruction_embedding": "masked mean of frozen VLA input token embeddings",
                "forbidden_inputs": [
                    "simulator reward",
                    "success",
                    "full observation",
                    "goal coordinates",
                    "expert future",
                    "branch labels",
                ],
            },
            indent=2,
        )
        + "\n"
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
    candidate_count,
    decision_steps,
    continuation_repeats=1,
    readout=None,
):
    shared = SharedEnvironment(task, seed)
    folder = output / f"{task}_seed{seed}"
    folder.mkdir()
    reset_policy(policy, preprocess, postprocess)
    torch.manual_seed(seed)
    started = time.perf_counter()
    step, success, reward_sum = 0, False, 0.0
    audits, actions, frames = [], [], [shared.frame("vla")]
    imageio.imwrite(folder / "initial.png", frames[0])
    try:
        while step < 400:
            if step in decision_steps:
                # Keep the baseline noise/processor state independent of audits.
                with torch.random.fork_rng(devices=[0]):
                    record = candidate_audit(
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
                        candidate_count,
                        continuation_repeats,
                        readout,
                    )
                reset_policy(policy, preprocess, postprocess)
                audits.append(record)
                print(
                    f"SHARED_STATE task={task} seed={seed} step={step} labels={record['alternative_labels_relative_to_candidate0']} replay=True",
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
        record = {
            "task": task,
            "seed": seed,
            "success": success,
            "steps": step,
            "reward_sum": reward_sum,
            "seconds_including_branch_audits": time.perf_counter() - started,
            "baseline_execution_chunk": 50,
            "states": audits,
        }
        np.save(folder / "baseline_executed_actions.npy", np.asarray(actions))
        imageio.mimsave(
            folder / "baseline_actual_execution.mp4", frames, fps=8, macro_block_size=1
        )
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"SHARED_EPISODE task={task} seed={seed} steps={step} success={success}",
            flush=True,
        )
        return record
    finally:
        shared.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--decision-steps", nargs="+", type=int, default=[0, 50])
    parser.add_argument("--continuation-repeats", type=int, default=1)
    parser.add_argument("--readout-dir", type=Path)
    args = parser.parse_args()
    if args.candidates < 2 or any(
        step < 0 or step >= 400 or step % 50 for step in args.decision_steps
    ):
        parser.error(
            "Use at least two candidates and decision steps at 50-action boundaries below 400"
        )
    if not 1 <= args.continuation_repeats <= 100:
        parser.error("Use one to one hundred matched continuation repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "stage": "shared_runtime_and_paired_branch_audit",
        "status": "running",
        "episodes": [],
        "scope": "compatibility and unscored candidate outcomes; no trained scorer/gate",
        "candidate_count": args.candidates,
        "decision_steps": args.decision_steps,
        "source_seeds": args.seeds,
        "continuation_repeats": args.continuation_repeats,
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
            (args.output / f"{name}_manifest.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        readout = None
        if args.readout_dir is not None:
            from .readout_study import FrozenReadout

            readout = FrozenReadout(args.readout_dir)
            result["frozen_readout"] = readout.spec
            result["scope"] = (
                "Frozen evaluator with matched continuation repeats; no gate or closed-loop performance"
            )
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
        for task in args.tasks:
            for seed in args.seeds:
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
                        args.candidates,
                        args.decision_steps,
                        args.continuation_repeats,
                        readout,
                    )
                )
                (args.output / "progress.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "passed"
    except Exception as exc:
        result["status"], result["error"] = "failed", repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SHARED_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
