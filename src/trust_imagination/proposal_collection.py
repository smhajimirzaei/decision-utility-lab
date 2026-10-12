"""Isolated paired proposal collection; prior experiment modules stay unchanged."""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import time
import numpy as np
from .proposal_contract import pool_contract, runtime_decision, validate_config


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
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
    cfg,
):
    import imageio.v2 as imageio
    import metaworld
    import torch
    from .branching import capture, restore
    from .cadence_pilot import physical_future, replay_errors
    from .cadence_contract import forecast_valid_mask, trace_prefix_success
    from .jepa import grouped_actions
    from .pilot import check_branch, TASKS
    from .repeated_outcomes import continuation_seeds, summarize_repeats
    from .shared_pilot import continuation, policy_chunk, reset_policy, wm_observation

    started = time.perf_counter()
    folder = output / f"{task}_seed{seed}" / f"state{step}"
    (folder / "features").mkdir(parents=True)
    (folder / "outcomes").mkdir()
    initial_obs = shared.obs.copy()
    snapshot = capture(shared.env)
    wm_frame = shared.frame("wm")
    vla_frame = shared.frame("vla")
    imageio.imwrite(folder / "features/current_wm.png", wm_frame)
    imageio.imwrite(folder / "features/current_vla.png", vla_frame)
    simulator_replay = check_branch(
        shared.env, initial_obs.copy(), getattr(metaworld.policies, TASKS[task])()
    )
    shared.obs = initial_obs.copy()
    if not simulator_replay["passed"] or not np.array_equal(
        wm_frame, shared.frame("wm")
    ):
        raise RuntimeError("Simulator or native-camera replay failed")
    proposal_seed = cfg["proposal_seed_offset"] + seed * 1000 + step
    perturbation_seed = cfg["perturbation_seed_offset"] + seed * 1000 + step
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(proposal_seed)
        reset_policy(policy, preprocess, postprocess)
        raw = np.stack(
            [policy_chunk(policy, preprocess, postprocess, shared) for _ in range(8)]
        )
    design = pool_contract(raw, perturbation_seed)
    pool = design["unique"]
    count = len(pool)
    # Construct candidates and predict before any physical future or label.
    with torch.inference_mode():
        context = world.encode(wm_observation(shared))
        visual = []
        proprio = []
        equivalence = {"visual": 0.0, "proprio": 0.0}
        torch.cuda.synchronize()
        prediction_start = time.perf_counter()
        for first in range(0, count, cfg["forecast_batch_limit"]):
            actions = pool[first : first + cfg["forecast_batch_limit"]]
            long = world.unroll(
                context, grouped_actions(actions, wm_preprocessor, "cuda:0")
            )
            short = world.unroll(
                context, grouped_actions(actions[:, :10], wm_preprocessor, "cuda:0")
            )
            for key in equivalence:
                if (
                    len(long[key]) != 11
                    or len(short[key]) != 3
                    or not torch.isfinite(long[key]).all()
                ):
                    raise RuntimeError("Invalid forecast transition contract")
                error = float((short[key] - long[key][:3]).abs().max().item())
                equivalence[key] = max(equivalence[key], error)
                if error > cfg["forecast_prefix_equivalence_atol"]:
                    raise RuntimeError("Independent short/long forecasts disagree")
            visual.append(long["visual"][1:].transpose(0, 1))
            proprio.append(long["proprio"][1:].transpose(0, 1))
        predicted = torch.cat(visual)
        predicted_proprio = torch.cat(proprio)
        torch.cuda.synchronize()
        prediction_seconds = time.perf_counter() - prediction_start
    np.savez_compressed(
        folder / "features/deployment_inputs.npz",
        raw_proposals=raw,
        logical_pool=design["logical"],
        executed_unique_pool=pool,
        pool_to_unique=design["mapping"],
        perturbation_directions=design["directions"],
        perturbation_envelope=design["envelope"],
        proprio=initial_obs[:4],
        current_latent=context["visual"].cpu().numpy(),
        current_proprio_latent=context["proprio"].cpu().numpy(),
        predicted_latents=predicted.cpu().numpy(),
        predicted_proprio_latents=predicted_proprio.cpu().numpy(),
    )
    images, physical_proprios, stops, physical_flags = physical_future(
        shared, snapshot, initial_obs, pool
    )
    valid = forecast_valid_mask(stops)
    with torch.inference_mode():
        actual = [
            world.encode(
                {
                    "visual": torch.from_numpy(images[i : i + 1]).permute(
                        0, 1, 4, 2, 3
                    ),
                    "proprio": torch.from_numpy(physical_proprios[i : i + 1]),
                }
            )
            for i in range(count)
        ]
        target = torch.cat([x["visual"] for x in actual])
        target_proprio = torch.cat([x["proprio"] for x in actual])
        if target.shape != predicted.shape:
            raise RuntimeError("Physical future shape differs")
        mse = (predicted - target).square().flatten(2).mean(-1).cpu().numpy()
        persistence = (
            (context["visual"][:, -1:] - target)
            .square()
            .flatten(2)
            .mean(-1)
            .cpu()
            .numpy()
        )
    np.savez_compressed(
        folder / "outcomes/actual_future_features.npz",
        latent=target.cpu().numpy(),
        proprio=physical_proprios,
        proprio_latent=target_proprio.cpu().numpy(),
        valid_mask=valid,
        first_stop_or_limit=stops,
    )
    seeds = continuation_seeds(seed, step, 3)
    repeated = []
    replays = []
    for repeat, repeat_seed in enumerate(seeds):
        branches = []
        for candidate, prefix in enumerate(pool):
            b = continuation(
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
            n = min(50, len(b["actions"]))
            if not np.array_equal(b["actions"][:n], prefix[:n]) or not np.array_equal(
                b["outcomes"][:n], physical_flags[candidate][:n]
            ):
                raise RuntimeError(
                    "Repeated trace differs from shared physical action prefix"
                )
            np.savez_compressed(
                folder / "outcomes" / f"unique{candidate}_repeat{repeat}_trace.npz",
                **{
                    k: b[k]
                    for k in (
                        "actions",
                        "rewards",
                        "outcomes",
                        "final_observation",
                        "final_integration",
                    )
                },
            )
            branches.append(b)
        base = continuation(
            policy,
            preprocess,
            postprocess,
            shared,
            snapshot,
            initial_obs,
            pool[0],
            400 - step,
            repeat_seed,
        )
        replays.append(replay_errors(branches[0], base))
        repeated.append(
            [
                {k: b[k] for k in ("success", "final_success", "steps", "reward_sum")}
                for b in branches
            ]
        )
        print(
            f"PROPOSAL_REPEAT task={task} seed={seed} step={step} repeat={repeat} unique={count} replay=True",
            flush=True,
        )
    conditions = []
    for name, indices in design["banks"].items():
        outcomes = [[r[i] for i in indices] for r in repeated]
        early = [trace_prefix_success(physical_flags[i], 50) for i in indices]
        actions = pool[indices]
        condition = {
            "name": name,
            "unique_indices": indices,
            "execution_prefix": 50,
            "continuation_chunk": 50,
            "remaining_horizon": 400 - step,
            "continuation_seeds": seeds,
            "branch_repeats": outcomes,
            "early_success_within_prefix": early,
            "scorer_training_eligible": [not x for x in early],
            "distinct_executed_proposals": len({x.tobytes() for x in actions}),
            "action_at_bound_fraction": float(np.mean(np.abs(actions) == 1)),
            "mean_pair_action_rms": float(
                np.mean(
                    [
                        np.sqrt(np.mean((actions[i] - actions[j]) ** 2))
                        for i in range(8)
                        for j in range(i + 1, 8)
                    ]
                )
            ),
            **summarize_repeats(outcomes),
        }
        conditions.append(condition)
    record = {
        "episode_id": f"{task}_seed{seed}",
        "task": task,
        "seed": seed,
        "state_step": step,
        "instruction": shared.instruction,
        "proposal_seed": proposal_seed,
        "perturbation_seed": perturbation_seed,
        "logical_pool_size": 12,
        "unique_count": count,
        "pool_to_unique": design["mapping"].tolist(),
        "bank_unique_indices": design["banks"],
        "unique_pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest(),
        "unique_branch_repeats": repeated,
        "continuation_replay_by_repeat": replays,
        "simulator_replay": simulator_replay,
        "forecast_prefix_max_abs_error": equivalence,
        "forecast_seconds": prediction_seconds,
        "model_mse": [
            [float(v) if ok else None for v, ok in zip(row, mask)]
            for row, mask in zip(mse, valid)
        ],
        "persistence_mse": [
            [float(v) if ok else None for v, ok in zip(row, mask)]
            for row, mask in zip(persistence, valid)
        ],
        "forecast_valid_mask": valid.tolist(),
        "first_stop_or_limit": stops.tolist(),
        "conditions": conditions,
        "seconds_including_branch_audit": time.perf_counter() - started,
    }
    write(folder / "outcomes/state.json", record)
    write(
        folder / "features/schema.json",
        {
            "instruction": shared.instruction,
            "bank_logical_indices": cfg["banks"],
            "unique_pool_mapping": "Byte-identical clipped proposals executed once and referenced by both banks",
            "world_frameskip": 5,
            "world_transitions": 10,
            "forbidden_deployment_inputs": [
                "reward",
                "success",
                "full observation",
                "goal coordinates",
                "actual futures",
            ],
            "padded_actual_frames": "Offline only; never use masked physical endpoints as observed data",
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
    cfg,
):
    import imageio.v2 as imageio
    import torch
    from .shared_pilot import SharedEnvironment, policy_chunk, reset_policy

    shared = SharedEnvironment(task, seed)
    folder = output / f"{task}_seed{seed}"
    folder.mkdir()
    started = time.perf_counter()
    step = 0
    states = []
    actions = []
    success = False
    reward_sum = 0.0
    reset_policy(policy, preprocess, postprocess)
    torch.manual_seed(seed)
    try:
        frames = [shared.frame("vla")]
        imageio.imwrite(folder / "initial.png", frames[0])
        while step < 400:
            if step in steps:
                with torch.random.fork_rng(devices=[0]):
                    state = audit(
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
                        cfg,
                    )
                reset_policy(policy, preprocess, postprocess)
                states.append(state)
                print(
                    f'PROPOSAL_STATE task={task} seed={seed} step={step} unique={state["unique_count"]}',
                    flush=True,
                )
            for action in np.clip(
                policy_chunk(policy, preprocess, postprocess, shared), -1, 1
            ):
                reward, term, trunc, flag = shared.step(action)
                actions.append(action.copy())
                step += 1
                success |= flag
                reward_sum += reward
                if step % 10 == 0 or success or step == 400:
                    frames.append(shared.frame("vla"))
                if success or term or trunc or step == 400:
                    break
            if success or term or trunc:
                break
        result = {
            "task": task,
            "seed": seed,
            "steps": step,
            "success": success,
            "reward_sum": reward_sum,
            "states": states,
            "baseline_execution_chunk": 50,
            "seconds_including_branch_audits": time.perf_counter() - started,
        }
        np.save(folder / "baseline_executed_actions.npy", np.asarray(actions))
        imageio.mimsave(
            folder / "baseline_actual_execution.mp4", frames, fps=8, macro_block_size=1
        )
        write(folder / "result.json", result)
        print(f"PROPOSAL_EPISODE task={task} seed={seed} steps={step}", flush=True)
        return result
    finally:
        shared.close()


def run(output, cfg, budget_seconds):
    import torch
    from .jepa import load_metaworld
    from .vla_pilot import load_policy

    output.mkdir(parents=True, exist_ok=False)
    write(output / "declared_protocol.json", cfg)
    pilot_folder = output / "pilot"
    main_folder = output / "main"
    pilot_folder.mkdir()
    main_folder.mkdir()
    stages = {
        name: {"status": "running", "stage": name, "episodes": [], "value_scorer": None}
        for name in ("pilot", "main")
    }
    grid = [(t, s) for s in cfg["main_seeds"] for t in cfg["tasks"]]
    started = time.monotonic()
    run = {
        "status": "running",
        "runtime_decisions": [],
        "no_outcome_based_progression": True,
        "experiment_budget_seconds": budget_seconds,
        "completed_main_episodes": [],
        "remaining_main_episodes": [],
    }

    def save():
        completed = {(e["task"], e["seed"]) for e in stages["main"]["episodes"]}
        run["completed_main_episodes"] = [
            {"task": t, "seed": s} for t, s in grid if (t, s) in completed
        ]
        run["remaining_main_episodes"] = [
            {"task": t, "seed": s} for t, s in grid if (t, s) not in completed
        ]
        run["elapsed_seconds"] = time.monotonic() - started
        for name, stage in stages.items():
            write(output / name / "summary.json", stage)
        write(output / "run_summary.json", run)

    def timed_out(*_):
        raise TimeoutError("Declared episode/pilot runtime guard exceeded")

    signal.signal(signal.SIGALRM, timed_out)
    try:
        save()
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
            write(output / f"{name}_manifest.json", report)
        run["package_versions"] = {
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
        pilot_remaining = budget_seconds - (time.monotonic() - started)
        if pilot_remaining < cfg["runtime"]["remaining_margin_seconds"]:
            raise TimeoutError("Model loading exhausted the experiment envelope")
        signal.setitimer(
            signal.ITIMER_REAL,
            min(cfg["runtime"]["pilot_stage_timeout_seconds"], pilot_remaining),
        )
        try:
            for task in cfg["tasks"]:
                stages["pilot"]["episodes"].append(
                    episode(
                        policy,
                        preprocess,
                        postprocess,
                        world,
                        wm_preprocessor,
                        task,
                        cfg["pilot_seed"],
                        pilot_folder,
                        cfg["pilot_decision_steps"],
                        cfg,
                    )
                )
                save()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        stages["pilot"]["status"] = "passed"
        save()
        for task, seed in grid:
            support = next(e for e in stages["pilot"]["episodes"] if e["task"] == task)
            remaining = budget_seconds - (time.monotonic() - started)
            decision = runtime_decision(support, remaining, cfg["runtime"])
            decision.update(task=task, seed=seed)
            run["runtime_decisions"].append(decision)
            save()
            print("PROPOSAL_RUNTIME " + json.dumps(decision), flush=True)
            if not decision["allowed"]:
                break
            signal.setitimer(
                signal.ITIMER_REAL, cfg["runtime"]["episode_timeout_seconds"]
            )
            try:
                stages["main"]["episodes"].append(
                    episode(
                        policy,
                        preprocess,
                        postprocess,
                        world,
                        wm_preprocessor,
                        task,
                        seed,
                        main_folder,
                        cfg["main_decision_steps"],
                        cfg,
                    )
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            save()
        stages["main"]["status"] = (
            "passed" if not run["remaining_main_episodes"] else "partial_runtime_budget"
        )
        run["status"] = stages["main"]["status"]
        run["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = repr(exc)
        for stage in stages.values():
            if stage["status"] == "running":
                stage["status"] = "failed"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/proposal_opportunity_20261011.json"),
    )
    p.add_argument("--budget-seconds", type=int, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_config(cfg)
    if not 300 <= args.budget_seconds <= cfg["runtime"]["experiment_cap_seconds"]:
        p.error("Budget must be300..6000s, below provider deadline with export margin")
    run(args.output, cfg, args.budget_seconds)


if __name__ == "__main__":
    main()
