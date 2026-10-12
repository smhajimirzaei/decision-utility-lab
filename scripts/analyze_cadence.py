"""Trace-first analysis of declared paired-prefix shards; never fits models."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from trust_imagination.cadence_contract import (
    condition_metrics,
    forecast_valid_mask,
    prefix_bank,
    trace_prefix_success,
)
from trust_imagination.repeated_outcomes import continuation_seeds, summarize_repeats


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_stage(
    source, config, stage, expected_seed=None, expected_grid=None, allow_partial=False
):
    summary = read(source / "summary.json")
    if (
        summary["status"]
        not in (("passed", "partial_runtime_budget") if allow_partial else ("passed",))
        or summary["stage"] != stage
    ):
        raise AssertionError("Complete successful stage required")
    grid = {(e["task"], e["seed"]) for e in summary["episodes"]}
    expected = (
        set(expected_grid)
        if expected_grid is not None
        else {(task, expected_seed) for task in config["tasks"]}
    )
    if grid != expected or len(summary["episodes"]) != len(expected):
        raise AssertionError("Declared complete task/seed grid differs")
    verified = 0
    replays = 0
    rows = []
    for episode in summary["episodes"]:
        allowed = (
            config["pilot_decision_steps"]
            if stage == "pilot"
            else config["main_decision_steps"]
        )
        reached = [step for step in allowed if step < episode["steps"]]
        if sorted(s["state_step"] for s in episode["states"]) != sorted(reached):
            raise AssertionError("Reached decision states missing or added")
        for state in episode["states"]:
            folder = source / state["episode_id"] / f"state{state['state_step']}"
            if not state["simulator_replay"]["passed"] or any(
                x > config["forecast_prefix_equivalence_atol"]
                for x in state["forecast_prefix_max_abs_error"].values()
            ):
                raise AssertionError("Simulator/forecast prefix verification failed")
            with np.load(
                folder / "features/deployment_inputs.npz", allow_pickle=False
            ) as inputs:
                raw = inputs["raw_proposals"]
                full = inputs["executed_full_proposals"].copy()
                if not np.array_equal(full, prefix_bank(raw, 50)) or len(full) != 8:
                    raise AssertionError("Clipped full proposal bank differs")
                if (
                    hashlib.sha256(full.tobytes()).hexdigest()
                    != state["proposal_sha256"]
                ):
                    raise AssertionError("Proposal hash differs")
                for key in inputs.files:
                    if not np.isfinite(inputs[key]).all():
                        raise AssertionError("Nonfinite deployment inputs")
                predicted = inputs["predicted_latents"].copy()
                current = inputs["current_latent"].copy()
                if predicted.shape[1] != 10:
                    raise AssertionError("Ten long transitions required")
            with np.load(
                folder / "outcomes/actual_future_features.npz", allow_pickle=False
            ) as future:
                mask = forecast_valid_mask(future["first_stop_or_limit"])
                if (
                    not np.array_equal(mask, future["valid_mask"])
                    or mask.tolist() != state["forecast_valid_mask"]
                ):
                    raise AssertionError("Forecast mask differs")
                actual = future["latent"]
                if actual.shape != predicted.shape or not np.isfinite(actual).all():
                    raise AssertionError("Invalid physical future latent array")
                recomputed = {
                    "model_mse": np.square(predicted - actual)
                    .reshape(8, 10, -1)
                    .mean(-1),
                    "persistence_mse": np.square(current[:, -1:] - actual)
                    .reshape(8, 10, -1)
                    .mean(-1),
                }
                for key, values in recomputed.items():
                    reported = np.asarray(
                        [
                            [np.nan if v is None else v for v in row]
                            for row in state[key]
                        ]
                    )
                    if reported.shape != (8, 10) or not np.allclose(
                        values[mask], reported[mask], rtol=1e-5, atol=1e-6
                    ):
                        raise AssertionError(
                            "Forecast errors differ from saved raw latent arrays"
                        )
            conditions = {c["execution_prefix"]: c for c in state["conditions"]}
            if set(conditions) != {10, 50}:
                raise AssertionError("Both paired prefix conditions required")
            by_bank = {}
            for length, condition in conditions.items():
                if (
                    condition["remaining_horizon"] != 400 - state["state_step"]
                    or condition["continuation_chunk"] != 50
                ):
                    raise AssertionError("Endpoint/continuation contract changed")
                if condition["continuation_seeds"] != continuation_seeds(
                    episode["seed"], state["state_step"], 3
                ):
                    raise AssertionError("Matched repeat seed contract changed")
                if (
                    len(condition["branch_repeats"]) != 3
                    or len(condition["continuation_replay_by_repeat"]) != 3
                ):
                    raise AssertionError("Exactly three outcomes and replays required")
                expected_summary = summarize_repeats(condition["branch_repeats"])
                for key, value in expected_summary.items():
                    if condition[key] != value:
                        raise AssertionError(f"Repeat summary differs: {key}")
                for errors in condition["continuation_replay_by_repeat"]:
                    if any(x != 0 for x in errors.values()):
                        raise AssertionError("Nonexact continuation replay")
                    replays += 1
                for repeat, branches in enumerate(condition["branch_repeats"]):
                    if len(branches) != 8:
                        raise AssertionError("Eight candidates required")
                    for candidate, branch in enumerate(branches):
                        path = (
                            folder
                            / "outcomes"
                            / f"prefix{length}"
                            / f"candidate{candidate}_repeat{repeat}_trace.npz"
                        )
                        with np.load(path, allow_pickle=False) as trace:
                            actions = trace["actions"]
                            flags = trace["outcomes"]
                            rewards = trace["rewards"]
                            steps = branch["steps"]
                            if (
                                actions.shape != (steps, 4)
                                or flags.shape != (steps, 3)
                                or rewards.shape != (steps,)
                            ):
                                raise AssertionError("Invalid trace shapes")
                            if (
                                not 1 <= steps <= 400 - state["state_step"]
                                or not np.isfinite(actions).all()
                                or np.max(np.abs(actions)) > 1
                            ):
                                raise AssertionError("Invalid trace bounds")
                            if not np.array_equal(
                                actions[: min(length, steps)],
                                full[candidate, : min(length, steps)],
                            ):
                                raise AssertionError(
                                    "Executed prefix differs from common proposal"
                                )
                            if (
                                flags[:-1].any()
                                or bool(flags[:, 2].any()) != branch["success"]
                                or bool(flags[-1, 2]) != branch["final_success"]
                            ):
                                raise AssertionError(
                                    "Ever-success/first-stop endpoint differs"
                                )
                            if (
                                not flags[-1].any()
                                and steps != 400 - state["state_step"]
                            ):
                                raise AssertionError(
                                    "Trace stops before horizon without terminal event"
                                )
                            if (
                                trace_prefix_success(flags, length)
                                != condition["early_success_within_prefix"][candidate]
                            ):
                                raise AssertionError("Early-prefix success differs")
                            if not np.isclose(
                                rewards.sum(),
                                branch["reward_sum"],
                                rtol=1e-10,
                                atol=1e-8,
                            ):
                                raise AssertionError("Reward summary differs")
                        verified += 1
                success = [
                    [b["success"] for b in repeat]
                    for repeat in condition["branch_repeats"]
                ]
                for bank in (4, 8):
                    by_bank.setdefault(str(bank), {})[str(length)] = condition_metrics(
                        success, condition["early_success_within_prefix"], bank
                    )
            for key in ("model_mse", "persistence_mse"):
                if len(state[key]) != 8:
                    raise AssertionError("Eight forecast rows required")
                for values, valid in zip(state[key], mask):
                    if len(values) != 10 or any(
                        (value is None) != (not ok) for value, ok in zip(values, valid)
                    ):
                        raise AssertionError("Post-stop errors were not masked")
            diversity = {}
            for length in (10, 50):
                pairs = [
                    np.sqrt(np.mean((full[i, :length] - full[j, :length]) ** 2))
                    for i in range(8)
                    for j in range(i + 1, 8)
                ]
                diversity[str(length)] = float(np.mean(pairs))
            rows.append(
                {
                    "episode_id": state["episode_id"],
                    "task": state["task"],
                    "seed": state["seed"],
                    "step": state["state_step"],
                    "banks": by_bank,
                    "action_pair_rms": diversity,
                    "model_mse": state["model_mse"],
                    "persistence_mse": state["persistence_mse"],
                    "forecast_seconds": state["forecast_seconds"],
                    "forecast_prefix_max_abs_error": state[
                        "forecast_prefix_max_abs_error"
                    ],
                }
            )
    return summary, rows, verified, replays


def aggregate(rows):
    episode_ids = sorted({row["episode_id"] for row in rows})

    def episode_mean(fn, subset=rows):
        values = []
        for episode_id in episode_ids:
            valid = [fn(row) for row in subset if row["episode_id"] == episode_id]
            valid = [x for x in valid if x is not None]
            if valid:
                values.append(float(np.mean(valid)))
        return {
            "mean": float(np.mean(values)) if values else None,
            "episodes_with_support": len(values),
        }

    result = {
        "episodes": len(episode_ids),
        "states": len(rows),
        "banks": {},
        "forecast_by_transition": [],
        "forecast_common_endpoint_support": {},
    }
    metrics = [
        "base",
        "always_one",
        "uniform_random",
        "best_measured_offline",
        "helpful_opportunity",
        "harmful_opportunity",
        "mixed_candidate_means",
        "alternative_help_frequency",
        "alternative_harm_frequency",
        "early_prefix_candidate_fraction",
    ]
    for bank in ("4", "8"):
        result["banks"][bank] = {}
        for length in ("10", "50"):
            result["banks"][bank][length] = {
                key: episode_mean(lambda row: row["banks"][bank][length][key])
                for key in metrics
            }
            result["banks"][bank][length]["helpful_state_count"] = sum(
                r["banks"][bank][length]["helpful_opportunity"] for r in rows
            )
            result["banks"][bank][length]["harmful_state_count"] = sum(
                r["banks"][bank][length]["harmful_opportunity"] for r in rows
            )
            result["banks"][bank][length]["mixed_state_count"] = sum(
                r["banks"][bank][length]["mixed_candidate_means"] for r in rows
            )
            result["banks"][bank][length]["action_pair_rms"] = episode_mean(
                lambda row: row["action_pair_rms"][length]
            )
        result["banks"][bank]["paired_50_minus_10"] = {
            key: episode_mean(
                lambda row: float(row["banks"][bank]["50"][key])
                - float(row["banks"][bank]["10"][key])
            )
            for key in metrics
        }
        common = [
            r
            for r in rows
            if all(
                r["banks"][bank][length]["all_candidates_post_prefix_eligible"]
                for length in ("10", "50")
            )
        ]
        result["banks"][bank]["common_post_prefix_eligible"] = {
            "states": len(common),
            "episodes": len({r["episode_id"] for r in common}),
            "opportunity_gain_50_minus_10": episode_mean(
                lambda row: (
                    row["banks"][bank]["50"]["best_measured_offline"]
                    - row["banks"][bank]["50"]["base"]
                )
                - (
                    row["banks"][bank]["10"]["best_measured_offline"]
                    - row["banks"][bank]["10"]["base"]
                ),
                common,
            ),
        }
        categories = {"both": 0, "ten_only": 0, "fifty_only": 0, "neither": 0}
        for row in rows:
            a = row["banks"][bank]["10"]["helpful_opportunity"]
            b = row["banks"][bank]["50"]["helpful_opportunity"]
            categories[
                (
                    "both"
                    if a and b
                    else "ten_only" if a else "fifty_only" if b else "neither"
                )
            ] += 1
        result["banks"][bank]["helpful_opportunity_overlap"] = categories
    for transition in range(10):
        item = {"executed_actions": (transition + 1) * 5}
        for key in ("model_mse", "persistence_mse"):

            def mean_valid(row, key=key):
                values = [
                    candidate[transition]
                    for candidate in row[key]
                    if candidate[transition] is not None
                ]
                return float(np.mean(values)) if values else None

            item[key] = episode_mean(mean_valid)
        item["valid_candidate_states"] = sum(
            r["model_mse"][c][transition] is not None for r in rows for c in range(8)
        )
        result["forecast_by_transition"].append(item)
    for actions in (10, 50):

        def common_error(row, key):
            values = [
                row[key][candidate][actions // 5 - 1]
                for candidate in range(8)
                if row["model_mse"][candidate][9] is not None
            ]
            return float(np.mean(values)) if values else None

        result["forecast_common_endpoint_support"][str(actions)] = {
            key: episode_mean(lambda r, key=key: common_error(r, key))
            for key in ("model_mse", "persistence_mse")
        }
    result["largest_short_long_prefix_prediction_difference"] = max(
        (max(r["forecast_prefix_max_abs_error"].values()) for r in rows), default=None
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = read(args.source / "declared_protocol.json")
    run = read(args.source / "run_summary.json")
    args.output.mkdir(parents=True, exist_ok=False)
    pilot, pilot_rows, pilot_traces, pilot_replays = validate_stage(
        args.source / "pilot", config, "pilot", 26
    )
    main_rows = []
    main_traces = 0
    main_replays = 0
    episodes = []
    if "completed_main_episodes" in run:
        grid = {(item["task"], item["seed"]) for item in run["completed_main_episodes"]}
        declared = {
            (task, seed) for task in config["tasks"] for seed in config["main_seeds"]
        }
        if not grid <= declared or len(grid) != len(run["completed_main_episodes"]):
            raise AssertionError("Undeclared/duplicate main episode")
        summary, main_rows, main_traces, main_replays = validate_stage(
            args.source / "main", config, "main", expected_grid=grid, allow_partial=True
        )
        episodes = summary["episodes"]
    else:
        for seed in run["completed_main_seeds"]:
            if seed not in config["main_seeds"]:
                raise AssertionError("Undeclared main seed")
            summary, rows, traces, replays = validate_stage(
                args.source / f"seed{seed}", config, "main", seed
            )
            main_rows += rows
            main_traces += traces
            main_replays += replays
            episodes += summary["episodes"]
    if run["status"] == "passed" and sorted(run["completed_main_seeds"]) != sorted(
        config["main_seeds"]
    ):
        raise AssertionError("Incomplete main grid marked passed")
    result = {
        "status": "verified",
        "source_run": args.source.name,
        "declared_main_grid_complete": sorted(run["completed_main_seeds"])
        == sorted(config["main_seeds"]),
        "completed_main_seeds": run["completed_main_seeds"],
        "remaining_main_seeds": run["remaining_main_seeds"],
        "completed_main_episodes": run.get("completed_main_episodes", []),
        "remaining_main_episodes": run.get("remaining_main_episodes", []),
        "pilot": aggregate(pilot_rows),
        "main": aggregate(main_rows),
        "pilot_traces_verified": pilot_traces,
        "main_traces_verified": main_traces,
        "exact_replay_checks": pilot_replays + main_replays,
        "source_policy_successes": sum(e["success"] for e in episodes),
        "source_policy_episodes": len(episodes),
        "runtime_decisions": run["runtime_decisions"],
        "scope": "Development-only paired prefix contrast; no fitted value head or world-model intervention benefit claim",
    }
    for name, value in [
        ("summary.json", result),
        ("main_states.json", main_rows),
        ("pilot_states.json", pilot_rows),
    ]:
        (args.output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
