"""Audit unique physical traces before any paired proposal-bank aggregate."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from trust_imagination.cadence_contract import (
    condition_metrics,
    forecast_valid_mask,
    trace_prefix_success,
)
from trust_imagination.proposal_repair_contract import pool_contract, validate_config
from trust_imagination.repeated_outcomes import continuation_seeds, summarize_repeats


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def episode_mean(rows, fn):
    values = []
    for ep in sorted({r["episode_id"] for r in rows}):
        v = [fn(r) for r in rows if r["episode_id"] == ep]
        v = [x for x in v if x is not None]
        if v:
            values.append(np.mean(v))
    return {
        "mean": float(np.mean(values)) if values else None,
        "episodes_with_support": len(values),
    }


def aggregate(rows):
    result = {
        "states": len(rows),
        "episodes": len({r["episode_id"] for r in rows}),
        "banks": {},
    }
    keys = (
        "base",
        "always_one",
        "uniform_random",
        "best_measured_offline",
        "helpful_opportunity",
        "harmful_opportunity",
        "alternative_help_frequency",
        "alternative_harm_frequency",
        "early_prefix_candidate_fraction",
    )
    for size in ("4", "8"):
        result["banks"][size] = {}
        for bank in ("policy", "mixed", "repair"):
            v = {
                key: episode_mean(rows, lambda r, key=key: r["banks"][bank][size][key])
                for key in keys
            }
            v["helpful_states"] = sum(
                r["banks"][bank][size]["helpful_opportunity"] for r in rows
            )
            v["harmful_states"] = sum(
                r["banks"][bank][size]["harmful_opportunity"] for r in rows
            )
            v["fixed_candidates"] = {
                str(i): episode_mean(rows, lambda r, i=i: r["candidate_means"][bank][i])
                for i in range(int(size))
            }
            best = [
                {
                    i
                    for i, p in enumerate(r["candidate_means"][bank][: int(size)])
                    if p == max(r["candidate_means"][bank][: int(size)])
                }
                for r in rows
            ]
            useful = [
                b - {0}
                for b, r in zip(best, rows)
                if r["banks"][bank][size]["helpful_opportunity"]
            ]
            v["all_state_best_index_intersection"] = (
                sorted(set.intersection(*best)) if best else None
            )
            v["helpful_state_best_index_intersection"] = (
                sorted(set.intersection(*useful)) if useful else None
            )
            result["banks"][size][bank] = v
        for left, right in [
            ("repair", "mixed"),
            ("repair", "policy"),
            ("mixed", "policy"),
        ]:
            result["banks"][size][left + "_minus_" + right] = {
                key: episode_mean(
                    rows,
                    lambda r, key=key: float(r["banks"][left][size][key])
                    - float(r["banks"][right][size][key]),
                )
                for key in keys
            }
    result["by_seed"] = {
        str(s): {
            "episodes": len({r["episode_id"] for r in rows if r["seed"] == s}),
            "helpful_repair8_states": sum(
                r["banks"]["repair"]["8"]["helpful_opportunity"]
                for r in rows
                if r["seed"] == s
            ),
        }
        for s in sorted({r["seed"] for r in rows})
    }
    return result


def validate_stage(folder, cfg, name, expected, allow_partial=False):
    summary = read(folder / "summary.json")
    if summary["stage"] != name or summary["status"] not in (
        "passed",
        "partial_runtime_budget",
    ):
        raise AssertionError(
            "Only completed successful or runtime-partial stages can be analyzed"
        )
    grid = {(e["task"], e["seed"]) for e in summary["episodes"]}
    if grid != set(expected) or len(grid) != len(summary["episodes"]):
        raise AssertionError("Declared episode grid differs")
    if name == "pilot" and summary["status"] != "passed":
        raise AssertionError("Complete technical pilot required")
    rows = []
    traces = 0
    replays = 0
    for episode in summary["episodes"]:
        decisions = cfg[f"{name}_decision_steps"]
        if sorted(s["state_step"] for s in episode["states"]) != [
            s for s in decisions if s < episode["steps"]
        ]:
            raise AssertionError("Reached state set differs")
        for state in episode["states"]:
            path = folder / state["episode_id"] / f"state{state['state_step']}"
            if not state["simulator_replay"]["passed"] or any(
                v > cfg["forecast_prefix_equivalence_atol"]
                for v in state["forecast_prefix_max_abs_error"].values()
            ):
                raise AssertionError("Simulator or prefix equivalence failed")
            with np.load(
                path / "features/deployment_inputs.npz", allow_pickle=False
            ) as d:
                design = pool_contract(d["raw_proposals"], state["perturbation_seed"])
                full = d["executed_unique_pool"]
                predicted = d["predicted_latents"]
                current = d["current_latent"]
                if (
                    not np.array_equal(full, design["unique"])
                    or not np.array_equal(d["pool_to_unique"], design["mapping"])
                    or not np.array_equal(d["logical_pool"], design["logical"])
                ):
                    raise AssertionError("Proposal/deduplication construction differs")
                if any(not np.isfinite(d[k]).all() for k in d.files):
                    raise AssertionError("Nonfinite deployment input")
            n = len(full)
            if (
                n != state["unique_count"]
                or hashlib.sha256(full.tobytes()).hexdigest()
                != state["unique_pool_sha256"]
                or design["banks"] != state["bank_unique_indices"]
            ):
                raise AssertionError("Pool metadata differs")
            if (
                state["proposal_seed"]
                != 100000 + episode["seed"] * 1000 + state["state_step"]
                or state["perturbation_seed"]
                != 200000 + episode["seed"] * 1000 + state["state_step"]
            ):
                raise AssertionError("Proposal seed formula differs")
            with np.load(
                path / "outcomes/actual_future_features.npz", allow_pickle=False
            ) as a:
                mask = forecast_valid_mask(a["first_stop_or_limit"])
                actual = a["latent"]
                if (
                    not np.array_equal(mask, a["valid_mask"])
                    or mask.tolist() != state["forecast_valid_mask"]
                    or actual.shape != predicted.shape
                    or predicted.shape[:2] != (n, 10)
                ):
                    raise AssertionError("Forecast shape/mask differs")
                if any(not np.isfinite(a[k]).all() for k in a.files):
                    raise AssertionError("Nonfinite physical arrays")
                errors = {
                    "model_mse": np.square(predicted - actual)
                    .reshape(n, 10, -1)
                    .mean(-1),
                    "persistence_mse": np.square(current[:, -1:] - actual)
                    .reshape(n, 10, -1)
                    .mean(-1),
                }
                for key, v in errors.items():
                    stored = np.asarray(
                        [
                            [np.nan if x is None else x for x in row]
                            for row in state[key]
                        ]
                    )
                    if (
                        stored.shape != (n, 10)
                        or not np.array_equal(np.isnan(stored), ~mask)
                        or not np.allclose(v[mask], stored[mask], rtol=1e-5, atol=1e-6)
                    ):
                        raise AssertionError("Raw forecast MSE/masking differs")
            if (
                state["first_stop_or_limit"]
                != np.asarray(
                    [min(50, b["steps"]) for b in state["unique_branch_repeats"][0]]
                ).tolist()
            ):
                raise AssertionError(
                    "Physical stop steps differ from executed prefix traces"
                )
            repeated = state["unique_branch_repeats"]
            if len(repeated) != 3 or any(len(r) != n for r in repeated):
                raise AssertionError("Unique outcome grid differs")
            early = [None] * n
            first_prefix_flags = [None] * n
            for repeat, branches in enumerate(repeated):
                for candidate, b in enumerate(branches):
                    with np.load(
                        path
                        / "outcomes"
                        / f"unique{candidate}_repeat{repeat}_trace.npz",
                        allow_pickle=False,
                    ) as t:
                        actions = t["actions"]
                        flags = t["outcomes"]
                        rewards = t["rewards"]
                        steps = b["steps"]
                        if (
                            actions.shape != (steps, 4)
                            or flags.shape != (steps, 3)
                            or rewards.shape != (steps,)
                            or not 1 <= steps <= 400 - state["state_step"]
                        ):
                            raise AssertionError("Unique trace bounds differ")
                        if (
                            not np.isfinite(actions).all()
                            or np.max(np.abs(actions)) > 1
                            or not np.isfinite(rewards).all()
                        ):
                            raise AssertionError("Invalid actions/rewards")
                        if not np.array_equal(
                            actions[: min(50, steps)], full[candidate, : min(50, steps)]
                        ):
                            raise AssertionError("Physical prefix differs")
                        if (
                            flags[:-1].any()
                            or bool(flags[:, 2].any()) != b["success"]
                            or bool(flags[-1, 2]) != b["final_success"]
                        ):
                            raise AssertionError("First-stop/success flags differ")
                        if not flags[-1].any() and steps != 400 - state["state_step"]:
                            raise AssertionError("Unexplained early stop")
                        if not np.isclose(
                            rewards.sum(), b["reward_sum"], rtol=1e-10, atol=1e-8
                        ):
                            raise AssertionError("Reward sum differs")
                        observed = trace_prefix_success(flags, 50)
                        if first_prefix_flags[candidate] is None:
                            first_prefix_flags[candidate] = flags[
                                : min(50, steps)
                            ].copy()
                        elif not np.array_equal(
                            first_prefix_flags[candidate], flags[: min(50, steps)]
                        ):
                            raise AssertionError(
                                "Physical prefix flags change across repeats"
                            )
                        if (
                            early[candidate] is not None
                            and early[candidate] != observed
                        ):
                            raise AssertionError(
                                "Early-prefix success changes across repeats"
                            )
                        early[candidate] = observed
                    traces += 1
            if len(state["continuation_replay_by_repeat"]) != 3 or any(
                v != 0
                for e in state["continuation_replay_by_repeat"]
                for v in e.values()
            ):
                raise AssertionError("Exact base continuation replay failed")
            replays += 3
            banks = {}
            supports = {}
            indices = {}
            if len(state["conditions"]) != 3 or {
                c["name"] for c in state["conditions"]
            } != {"policy", "mixed", "repair"}:
                raise AssertionError("Exactly three distinct bank conditions required")
            for c in state["conditions"]:
                cname = c["name"]
                ix = design["banks"][cname]
                expected_outcomes = [[r[i] for i in ix] for r in repeated]
                if (
                    c["unique_indices"] != ix
                    or c["branch_repeats"] != expected_outcomes
                    or c["continuation_seeds"]
                    != continuation_seeds(episode["seed"], state["state_step"], 3)
                ):
                    raise AssertionError(
                        "Shared branch reuse/continuation seed differs"
                    )
                if (
                    c["execution_prefix"] != 50
                    or c["continuation_chunk"] != 50
                    or c["remaining_horizon"] != 400 - state["state_step"]
                ):
                    raise AssertionError("Bank control contract differs")
                if c["early_success_within_prefix"] != [early[i] for i in ix]:
                    raise AssertionError("Early label differs")
                if c["scorer_training_eligible"] != [not early[i] for i in ix]:
                    raise AssertionError("Scorer eligibility differs")
                if c["distinct_executed_proposals"] != len(
                    {full[i].tobytes() for i in ix}
                ) or not np.isclose(
                    c["action_at_bound_fraction"], np.mean(np.abs(full[ix]) == 1)
                ):
                    raise AssertionError("Uniqueness/saturation differs")
                for k, v in summarize_repeats(expected_outcomes).items():
                    if c[k] != v:
                        raise AssertionError("Bank repeat metadata differs")
                success = [[int(b["success"]) for b in r] for r in expected_outcomes]
                banks[cname] = {
                    str(size): condition_metrics(
                        success, c["early_success_within_prefix"], size
                    )
                    for size in (4, 8)
                }
                supports[cname] = {
                    str(size): bool(
                        mask[ix[:size], 9].all()
                        and not any(early[i] for i in ix[:size])
                    )
                    for size in (4, 8)
                }
                indices[cname] = ix
            if (
                set(banks) != {"policy", "mixed", "repair"}
                or not banks["policy"]["4"]
                == banks["mixed"]["4"]
                == banks["repair"]["4"]
            ):
                raise AssertionError("Shared four-candidate control differs")
            rows.append(
                {
                    "episode_id": state["episode_id"],
                    "task": state["task"],
                    "seed": state["seed"],
                    "step": state["state_step"],
                    "unique_count": n,
                    "indices": indices,
                    "banks": banks,
                    "common_support": supports,
                    "model_mse": state["model_mse"],
                    "persistence_mse": state["persistence_mse"],
                    "candidate_means": {
                        name: np.mean(
                            [[int(r[i]["success"]) for i in ix] for r in repeated],
                            axis=0,
                        ).tolist()
                        for name, ix in indices.items()
                    },
                }
            )
    return rows, traces, replays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = read(a.source / "declared_protocol.json")
    validate_config(cfg)
    run = read(a.source / "run_summary.json")
    if run["status"] not in ("passed", "partial_runtime_budget"):
        raise AssertionError("Successful/partial completed run required")
    pilot, pilot_traces, pilot_replays = validate_stage(
        a.source / "pilot", cfg, "pilot", [(t, cfg["pilot_seed"]) for t in cfg["tasks"]]
    )
    grid = {(t, s) for s in cfg["main_seeds"] for t in cfg["tasks"]}
    completed = {(r["task"], r["seed"]) for r in run["completed_main_episodes"]}
    if not completed <= grid or len(completed) != len(run["completed_main_episodes"]):
        raise AssertionError("Undeclared/duplicate main grid")
    remaining = {(r["task"], r["seed"]) for r in run["remaining_main_episodes"]}
    if remaining != grid - completed or (run["status"] == "passed") != bool(
        completed == grid
    ):
        raise AssertionError("Completion/partial metadata differs")
    main_rows, main_traces, main_replays = validate_stage(
        a.source / "main", cfg, "main", completed, allow_partial=True
    )
    a.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "verified",
        "source_run": a.source.name,
        "main_grid_complete": completed == grid,
        "pilot": aggregate(pilot),
        "main": aggregate(main_rows),
        "pilot_unique_traces_verified": pilot_traces,
        "main_unique_traces_verified": main_traces,
        "exact_base_replays": pilot_replays + main_replays,
        "remaining_main_episodes": run["remaining_main_episodes"],
        "scope": "Development-only fixed candidate-design comparison; no fitted selector or closed-loop benefit",
    }
    for filename, value in (
        ("summary.json", report),
        ("pilot_states.json", pilot),
        ("main_states.json", main_rows),
    ):
        (a.output / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("pilot", "main")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
