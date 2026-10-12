"""M1 finite-output and matched-transition diagnostics, not policy evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from .branching import capture, restore
from .jepa import grouped_actions, load_metaworld
from .pilot import make_environment


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def summary_tensor(value):
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.float().mean()),
        "finite": bool(torch.isfinite(value).all()),
    }


def diagnostic(model, preprocessor, task, seed, output, device):
    env, expert = make_environment(task, seed)
    try:
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner2")
        if camera_id != 2:
            raise RuntimeError(f"Source camera index changed: corner2={camera_id}")
        # Match the official evaluator, including its explicit vertical flip.
        env.model.cam_pos[camera_id] = [0.75, 0.075, 0.7]
        obs, _ = env.reset(seed=seed)

        def frame():
            pixels = env.render()[::-1].copy()
            if pixels.shape != (224, 224, 3) or pixels.max() == pixels.min():
                raise RuntimeError("Rendered observation is empty or constant")
            return pixels

        for _ in range(20):
            obs, *_ = env.step(np.clip(expert.get_action(obs), -1, 1))
        context_frames = [frame()]
        context_proprio = [obs[:4].copy()]
        for _ in range(5):
            obs, *_ = env.step(np.clip(expert.get_action(obs), -1, 1))
        context_frames.append(frame())
        context_proprio.append(obs[:4].copy())
        snapshot = capture(env)
        base_actions = []
        for _ in range(10):
            action = np.clip(expert.get_action(obs), -1, 1).astype(np.float32)
            base_actions.append(action)
            obs, *_ = env.step(action)
        restore(env, snapshot)
        base_actions = np.stack(base_actions)
        opposite = base_actions.copy()
        opposite[:, :3] *= -1
        rng = np.random.default_rng(seed + 1000)
        bank = np.stack(
            [
                base_actions,
                np.zeros_like(base_actions),
                opposite,
                rng.uniform(-1, 1, base_actions.shape).astype(np.float32),
            ]
        )
        names = ["scripted_expert", "zero", "opposite_xyz", "random"]
        actual_frames, actual_proprio, outcomes = [], [], []
        for actions in bank:
            restore(env, snapshot)
            frames, proprios, infos = [], [], []
            for index, action in enumerate(actions):
                obs, reward, terminated, truncated, info = env.step(action)
                infos.append(
                    {
                        "reward": float(reward),
                        "success": float(info.get("success", 0)),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if (index + 1) % 5 == 0:
                    frames.append(frame())
                    proprios.append(obs[:4].copy())
            actual_frames.append(frames)
            actual_proprio.append(proprios)
            outcomes.append(infos)
        # The published task has num_frames=num_proprios=1. ctxt_window=2
        # governs autoregressive history, not two initial observed frames.
        visual = (
            torch.from_numpy(np.stack(context_frames[-1:]))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
        )
        proprio = torch.from_numpy(
            np.asarray(context_proprio[-1:], dtype=np.float32)
        ).unsqueeze(0)
        packed = grouped_actions(bank, preprocessor, device)
        with torch.inference_mode():
            # Warm up once; record encoding and candidate unroll separately.
            model.encode({"visual": visual, "proprio": proprio})
            synchronize(device)
            start = time.perf_counter()
            context = model.encode({"visual": visual, "proprio": proprio})
            synchronize(device)
            encode_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prediction = model.unroll(context, packed)
            synchronize(device)
            unroll_seconds = time.perf_counter() - start
            actual_visual = torch.from_numpy(np.asarray(actual_frames)).permute(
                0, 1, 4, 2, 3
            )
            actual_prop = torch.from_numpy(np.asarray(actual_proprio, dtype=np.float32))
            encoded_actual = model.encode(
                {"visual": actual_visual, "proprio": actual_prop}
            )
            predicted_visual = prediction["visual"][visual.shape[1] :].transpose(0, 1)
            target_visual = encoded_actual["visual"]
            if predicted_visual.shape != target_visual.shape:
                raise RuntimeError(
                    f"Prediction/time alignment mismatch: {predicted_visual.shape}, {target_visual.shape}"
                )
            model_mse = (predicted_visual - target_visual).square().flatten(2).mean(-1)
            persistence_mse = (
                (context["visual"][:, -1:] - target_visual).square().flatten(2).mean(-1)
            )
        record = {
            "task": task,
            "seed": seed,
            "candidate_names": names,
            "model_visual_mse_by_candidate_and_step": model_mse.cpu().tolist(),
            "persistence_visual_mse_by_candidate_and_step": persistence_mse.cpu().tolist(),
            "encode_seconds": encode_seconds,
            "four_candidate_two_step_unroll_seconds": unroll_seconds,
            "raw_visual": summary_tensor(visual),
            "raw_proprio": summary_tensor(proprio),
            "normalized_packed_actions": summary_tensor(packed),
            "predicted_visual": summary_tensor(predicted_visual),
            "outcomes": outcomes,
            "initial_observed_frames": 1,
            "autoregressive_context_window": 2,
            "state_step": 25,
            "future_simulator_steps": [30, 35],
        }
        if not record["predicted_visual"]["finite"]:
            raise RuntimeError("Nonfinite prediction")
        task_dir = output / f"{task}_seed{seed}"
        task_dir.mkdir()
        imageio.imwrite(task_dir / "context.png", context_frames[-1])
        imageio.imwrite(task_dir / "raw_unflipped.png", context_frames[-1][::-1])
        for index, name in enumerate(names):
            imageio.imwrite(task_dir / f"actual_{name}.png", actual_frames[index][-1])
        np.savez_compressed(
            task_dir / "inputs_and_transitions.npz",
            actions=bank,
            context_frames=np.stack(context_frames),
            context_proprio=np.asarray(context_proprio),
            actual_frames=np.asarray(actual_frames),
            actual_proprio=np.asarray(actual_proprio),
        )
        (task_dir / "diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["reach-v3", "push-v3", "door-open-v3"]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "stage": "M1_matched_transition_diagnostic",
        "status": "running",
        "scope": "scripted-expert instrumentation; not a VLA or planner reproduction",
        "packages": {},
        "tasks": [],
    }
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model, preprocessor, report = load_metaworld(
            args.source, args.manifest, Path("checkpoints/hf"), args.device
        )
        (args.output / "model_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"MODEL_LOADED parameters={report['parameter_count']} device={args.device}",
            flush=True,
        )
        for package in (
            "torch",
            "torchvision",
            "metaworld",
            "mujoco",
            "numpy",
            "tensordict",
            "huggingface_hub",
        ):
            result["packages"][package] = importlib.metadata.version(package)
        for task in args.tasks:
            record = diagnostic(model, preprocessor, task, 0, args.output, args.device)
            result["tasks"].append(record)
            print(
                f"DIAGNOSTIC_COMPLETE {task} unroll_seconds={record['four_candidate_two_step_unroll_seconds']:.4f}",
                flush=True,
            )
        result["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None
        )
        result["peak_host_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("M1_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
