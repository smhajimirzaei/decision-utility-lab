"""Collect the fixed main grid in complete episodes under runtime-only guards."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import signal
import time
import torch
from .cadence_pilot import episode, write_json
from .jepa import load_metaworld
from .vla_pilot import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot-source", type=Path, required=True)
    p.add_argument("--previous-run", type=Path)
    p.add_argument("--budget-minutes", type=int, choices=[75, 100], default=100)
    args = p.parse_args()
    config = json.loads(Path("configs/cadence_main_development.json").read_text())
    pilot = json.loads((args.pilot_source / "summary.json").read_text())
    if pilot["status"] != "passed" or {
        (e["task"], e["seed"]) for e in pilot["episodes"]
    } != {(task, 26) for task in config["tasks"]}:
        raise AssertionError("Successful fixed pilot required")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.pilot_source, args.output / "pilot")
    write_json(args.output / "declared_protocol.json", config)
    folder = args.output / "main"
    folder.mkdir()
    started = time.monotonic()
    runtime = config["main_runtime"]
    main_summary = {
        "status": "running",
        "stage": "main",
        "episodes": [],
        "source_seeds": config["main_seeds"],
        "decision_steps": config["main_decision_steps"],
        "prefix_lengths": config["prefix_lengths"],
        "candidate_count": 8,
        "continuation_repeats": 3,
        "value_scorer": None,
    }
    run = {
        "status": "running",
        "completed_main_episodes": [],
        "remaining_main_episodes": [],
        "completed_main_seeds": [],
        "remaining_main_seeds": config["main_seeds"].copy(),
        "runtime_decisions": [],
        "no_outcome_based_progression": True,
        "pilot_source": str(args.pilot_source),
    }
    run["activation_experiment_budget_minutes"] = args.budget_minutes
    grid = [(task, seed) for seed in config["main_seeds"] for task in config["tasks"]]

    def save():
        completed = {(e["task"], e["seed"]) for e in main_summary["episodes"]}
        run["completed_main_episodes"] = [
            {"task": t, "seed": s} for t, s in grid if (t, s) in completed
        ]
        run["remaining_main_episodes"] = [
            {"task": t, "seed": s} for t, s in grid if (t, s) not in completed
        ]
        run["completed_main_seeds"] = [
            s
            for s in config["main_seeds"]
            if all((t, s) in completed for t in config["tasks"])
        ]
        run["remaining_main_seeds"] = [
            s for s in config["main_seeds"] if s not in run["completed_main_seeds"]
        ]
        run["elapsed_seconds"] = time.monotonic() - started
        write_json(folder / "summary.json", main_summary)
        write_json(args.output / "run_summary.json", run)

    try:
        if args.previous_run:
            previous = json.loads((args.previous_run / "main/summary.json").read_text())
            previous_config = json.loads(
                (args.previous_run / "declared_protocol.json").read_text()
            )
            if previous_config != config:
                raise AssertionError("Resume scientific/runtime declaration differs")
            for e in previous["episodes"]:
                if (e["task"], e["seed"]) not in grid:
                    raise AssertionError("Undeclared previous episode")
                shutil.copytree(
                    args.previous_run / "main" / f'{e["task"]}_seed{e["seed"]}',
                    folder / f'{e["task"]}_seed{e["seed"]}',
                )
                main_summary["episodes"].append(e)
            run["resumed_from"] = str(args.previous_run)
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
            write_json(folder / f"{name}_manifest.json", report)
        pilot_by_task = {e["task"]: e for e in pilot["episodes"]}
        for task, seed in grid:
            if any(
                e["task"] == task and e["seed"] == seed
                for e in main_summary["episodes"]
            ):
                continue
            support = pilot_by_task[task]
            audit_time = max(
                s["seconds_including_branch_audit"] for s in support["states"]
            )
            overhead = support["seconds_including_branch_audits"] - sum(
                s["seconds_including_branch_audit"] for s in support["states"]
            )
            estimate = (
                runtime["safety_factor"]
                * (runtime["max_states_per_episode"] * audit_time + overhead)
                + runtime["fixed_overhead_seconds"]
            )
            remaining = args.budget_minutes * 60 - (time.monotonic() - started)
            allowed = (
                estimate <= runtime["per_episode_timeout_minutes"] * 60
                and estimate <= remaining - runtime["remaining_margin_seconds"]
            )
            decision = {
                "task": task,
                "seed": seed,
                "estimated_seconds": estimate,
                "remaining_seconds": remaining,
                "allowed": allowed,
                "rule": "Task-matched pilot timing, four-state worst case; no candidate-success-based progression",
            }
            run["runtime_decisions"].append(decision)
            save()
            print("CADENCE_MAIN_RUNTIME_DECISION " + json.dumps(decision), flush=True)
            if not allowed:
                break

            def timed_out(*_):
                raise TimeoutError("Complete-episode runtime cap reached")

            signal.signal(signal.SIGALRM, timed_out)
            signal.setitimer(
                signal.ITIMER_REAL, runtime["per_episode_timeout_minutes"] * 60
            )
            try:
                e = episode(
                    policy,
                    preprocess,
                    postprocess,
                    world,
                    wm_preprocessor,
                    task,
                    seed,
                    folder,
                    config["main_decision_steps"],
                    config,
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            main_summary["episodes"].append(e)
            save()
        main_summary["status"] = (
            "passed" if not run["remaining_main_episodes"] else "partial_runtime_budget"
        )
        run["status"] = main_summary["status"]
        main_summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        main_summary["status"] = "failed"
        run["status"] = "failed"
        run["error"] = repr(exc)
        raise
    finally:
        main_summary["elapsed_seconds_this_activation"] = time.monotonic() - started
        save()
    print("CADENCE_MAIN_COMPLETE " + run["status"], flush=True)


if __name__ == "__main__":
    main()
