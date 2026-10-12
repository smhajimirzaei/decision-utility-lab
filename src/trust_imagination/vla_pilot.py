"""Frozen SmolVLA feasibility in its own pinned LeRobot environment.

The official camera, language descriptions and four-value sensor interface are
used unchanged. Simulator reward and success are logging outputs only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import imageio.v2 as imageio
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(source, manifest, cache, device):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.factory import load_pretrained_policy_processors

    actual_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != manifest["source_revision"]:
        raise RuntimeError("LeRobot source revision mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("LeRobot source has local changes")
    policy_dir = Path(
        snapshot_download(
            manifest["policy_repository"],
            revision=manifest["policy_revision"],
            cache_dir=str(cache),
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    weight = policy_dir / manifest["weights_file"]
    if (
        weight.stat().st_size != manifest["weights_bytes"]
        or sha256(weight) != manifest["weights_sha256"]
    ):
        raise RuntimeError("Policy weights failed provenance verification")
    normalizer = policy_dir / manifest["normalizer_file"]
    if sha256(normalizer) != manifest["normalizer_sha256"]:
        raise RuntimeError("Policy normalization failed provenance verification")
    stats = load_file(str(normalizer))
    observed_stats = {key: list(value.shape) for key, value in stats.items()}
    if list(stats["observation.state.mean"].shape) != [4] or list(
        stats["action.mean"].shape
    ) != [4]:
        raise RuntimeError("Unexpected four-value sensor/action normalizer")
    base_dir = Path(
        snapshot_download(
            manifest["base_repository"],
            revision=manifest["base_revision"],
            cache_dir=str(cache),
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
        )
    )
    config = SmolVLAConfig.from_pretrained(policy_dir)
    # Architecture/tokenizer from a pinned snapshot; all policy weights then load
    # strictly from the verified checkpoint. Avoid downloading unused base weights.
    config.load_vlm_weights = False
    config.vlm_model_name = str(base_dir)
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(policy_dir, config=config, strict=True)
    policy.eval().requires_grad_(False)
    preprocess, postprocess = load_pretrained_policy_processors(
        policy_dir,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": str(base_dir)},
        },
    )
    report = {
        **manifest,
        "actual_weights_sha256": sha256(weight),
        "strict_weight_load": True,
        "normalizer_shapes": observed_stats,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "base_metadata_files": {
            p.name: sha256(p) for p in base_dir.iterdir() if p.is_file()
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in [
                "lerobot",
                "torch",
                "torchvision",
                "transformers",
                "metaworld",
                "mujoco",
                "gymnasium",
            ]
        },
        "python": platform.python_version(),
        "inference_device": device,
        "loaded_config": (
            config.to_dict() if hasattr(config, "to_dict") else str(config)
        ),
    }
    return policy, preprocess, postprocess, report


def episode(policy, preprocess, postprocess, task, seed, output, max_steps):
    from lerobot.envs.metaworld import MetaworldEnv

    episode_dir = output / f"{task}_seed{seed}"
    episode_dir.mkdir()
    env = MetaworldEnv(task=task, obs_type="pixels_agent_pos")
    policy.reset()
    preprocess.reset()
    postprocess.reset()
    torch.manual_seed(seed)
    np.random.seed(seed)
    records, actions, frames = [], [], []
    started = time.perf_counter()
    try:
        obs, _ = env.reset(seed=seed)
        instruction = env.task_description
        imageio.imwrite(episode_dir / "initial.png", obs["pixels"])
        frames.append(obs["pixels"].copy())
        np.save(episode_dir / "initial_proprio.npy", obs["agent_pos"])
        reward_sum, success, step, outside, raw_elements = 0.0, False, 0, 0, 0
        while step < max_steps:
            frame = obs["pixels"]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.max() == frame.min():
                raise RuntimeError("Invalid camera frame")
            state = np.asarray(obs["agent_pos"], dtype=np.float32)
            if state.shape != (4,) or not np.isfinite(state).all():
                raise RuntimeError("Invalid ordinary proprioceptive input")
            batch = preprocess(
                {
                    "observation.state": torch.from_numpy(state),
                    "observation.images.camera1": torch.from_numpy(frame.copy())
                    .permute(2, 0, 1)
                    .float()
                    / 255,
                    "task": instruction,
                }
            )
            if batch["observation.state"].shape != (1, 4):
                raise RuntimeError("Processor changed sensor dimensionality")
            with torch.inference_mode():
                if policy.config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()
                chunk = policy.predict_action_chunk(batch)
                if policy.config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                infer_seconds = time.perf_counter() - infer_start
                raw = postprocess(chunk).cpu().numpy()[0]
            if raw.shape != (50, 4) or not np.isfinite(raw).all():
                raise RuntimeError(f"Invalid predicted action chunk: {raw.shape}")
            if not records:
                np.save(episode_dir / "first_raw_chunk.npy", raw)
            outside += int((np.abs(raw) > 1).sum())
            raw_elements += raw.size
            chunk_start = step
            for action in np.clip(raw[: policy.config.n_action_steps], -1, 1):
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                success |= bool(info.get("is_success", False))
                actions.append(action.copy())
                step += 1
                if step % 5 == 0 or success or step == max_steps:
                    frames.append(obs["pixels"].copy())
                if terminated or truncated or step >= max_steps:
                    break
            records.append(
                {
                    "step": chunk_start,
                    "inference_seconds": infer_seconds,
                    "executed_actions": step - chunk_start,
                    "success": success,
                }
            )
            print(
                f"VLA_CHUNK task={task} seed={seed} step={step} infer_seconds={infer_seconds:.3f} success={success}",
                flush=True,
            )
            if terminated or truncated:
                break
        result = {
            "task": task,
            "seed": seed,
            "instruction": instruction,
            "success": success,
            "steps": step,
            "reward_sum": reward_sum,
            "seconds": time.perf_counter() - started,
            "raw_action_outside_fraction": outside / max(raw_elements, 1),
            "chunks": records,
            "input_camera_shape": list(frames[0].shape),
            "input_state_shape": [4],
            "max_steps": max_steps,
            "execution_chunk": policy.config.n_action_steps,
        }
        np.save(episode_dir / "executed_actions.npy", np.asarray(actions))
        imageio.mimsave(
            episode_dir / "actual_execution.mp4", frames, fps=16, macro_block_size=1
        )
        imageio.imwrite(episode_dir / "final.png", frames[-1])
        (episode_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"VLA_EPISODE_COMPLETE task={task} seed={seed} success={success} seconds={result['seconds']:.2f}",
            flush=True,
        )
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["reach-v3", "push-v3", "door-open-v3"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "stage": "frozen_SmolVLA_feasibility",
        "status": "running",
        "episodes": [],
        "scope": "six base-policy episodes; no gate, no statistical generalization claim",
    }
    started = time.perf_counter()
    try:
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        manifest = json.loads(args.manifest.read_text())
        policy, preprocess, postprocess, report = load_policy(
            args.source, manifest, Path("checkpoints/vla_hf"), args.device
        )
        (args.output / "model_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        for task in args.tasks:
            for seed in args.seeds:
                result["episodes"].append(
                    episode(
                        policy,
                        preprocess,
                        postprocess,
                        task,
                        seed,
                        args.output,
                        args.max_steps,
                    )
                )
                (args.output / "progress.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
        result["inference_device"] = args.device
        result["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        )
        result["status"] = "passed"
    except Exception as exc:
        result["status"], result["error"] = "failed", repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("VLA_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
