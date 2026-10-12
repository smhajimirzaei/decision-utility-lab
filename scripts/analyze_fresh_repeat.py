"""Analyze all matched continuation repeats using the frozen readout/controls."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from trust_imagination.readout_study import FrozenReadout, linear_predict
from trust_imagination.task_evaluator import encode_features


def episode_mean(records, key):
    groups = {}
    for r in records:
        groups.setdefault(r["episode_id"], []).append(r[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def analyze(source, frozen):
    summary = json.loads((source / "summary.json").read_text())
    if summary["status"] != "passed":
        raise ValueError("Complete successful shard required")
    readout = FrozenReadout(frozen)
    priors = json.loads((frozen / "training_priors.json").read_text())
    heads = {}
    for key in ["direct_action", "current_only"]:
        file = frozen / f"{key}_checkpoint.npz"
        with np.load(file, allow_pickle=False) as data:
            heads[key] = {k: data[k].copy() for k in data.files}
    states = [s for ep in summary["episodes"] for s in ep["states"]]
    decisions = {
        str(bank): {
            name: []
            for name in [
                "base",
                "always_candidate_one",
                "uniform_random_including_base",
                "predicted_future",
                "actual_future_offline",
                "direct_action",
                "current_only",
                "simulator_upper_bound",
            ]
        }
        for bank in [4, 8]
    }
    all_decisions = {
        bank: {name: [] for name in methods} for bank, methods in decisions.items()
    }
    exclusions = []
    score_checks = []
    repeat_variability = []
    candidate_effect_counts = {
        "helps": 0,
        "hurts": 0,
        "tie_both_success": 0,
        "tie_both_failure": 0,
    }
    error_rows = []
    omitted = {str(b): 0 for b in [4, 8]}
    repeat_count = 0
    for s in states:
        if s["readout_checkpoint_sha256"] != readout.spec["checkpoint_sha256"]:
            raise AssertionError("Readout changed between selection and analysis")
        if not s["simulator_replay"]["passed"] or any(
            v != 0
            for repeat in s["continuation_replay_by_repeat"]
            for v in repeat.values()
        ):
            raise AssertionError("Repeat replay failed")
        repeats = s["branch_repeats"]
        repeat_count += len(repeats)
        if len(repeats) != 3:
            raise ValueError("Expected the declared three matched repeats")
        labels = np.asarray([[int(b["success"]) for b in repeat] for repeat in repeats])
        for r in labels:
            for y in r[1:]:
                label = (
                    "helps"
                    if y > r[0]
                    else (
                        "hurts"
                        if y < r[0]
                        else "tie_both_success" if y else "tie_both_failure"
                    )
                )
                candidate_effect_counts[label] += 1
        state_id = f"{s['episode_id']}/state{s['state_step']}"
        with np.load(
            source / state_id / "features/deployment_inputs.npz", allow_pickle=False
        ) as d:
            current = encode_features(
                d["current_latent"][0, 0, 0],
                d["current_proprio_latent"][0, 0],
                d["instruction_embedding"],
                s["remaining_horizon"],
                heads["direct_action"],
            )
            current = np.broadcast_to(current, (s["candidate_count"], len(current)))
            scores = {
                "predicted_future": np.asarray(s["readout_scores"]["predicted_future"]),
                "actual_future_offline": np.asarray(
                    s["readout_scores"]["actual_future_offline"]
                ),
                "direct_action": linear_predict(
                    heads["direct_action"],
                    np.concatenate(
                        (
                            current,
                            d["executed_prefixes"].reshape(s["candidate_count"], 40),
                        ),
                        -1,
                    ),
                ),
                "current_only": linear_predict(heads["current_only"], current),
            }
            reproduced = readout.score(
                d["predicted_latents"][:, -1, 0],
                d["predicted_proprio_latents"][:, -1],
                d["instruction_embedding"],
                s["remaining_horizon"] - s["execution_prefix"],
            )
            error = float(np.max(np.abs(reproduced - scores["predicted_future"])))
            # Float32 pooling/projection reductions can differ slightly between
            # Linux and Windows. Preserve immutable VM scores and exact choices.
            if error > 5e-6:
                raise AssertionError("Fresh predicted score reproduction failed")
            if any(
                int(np.argmax(reproduced[:bank]))
                != int(np.argmax(scores["predicted_future"][:bank]))
                for bank in [4, 8]
            ):
                raise AssertionError("Nested-bank deployment choice changed")
            if int(np.argmax(reproduced)) != s["world_model_selected_index"]:
                raise AssertionError("Saved deployment choice changed")
            score_checks.append(
                {
                    "state_id": state_id,
                    "max_abs_error": error,
                    "scores": len(reproduced),
                }
            )
            with np.load(
                source / state_id / "outcomes/actual_future_features.npz",
                allow_pickle=False,
            ) as actual:
                offline = readout.score(
                    actual["latent"][:, -1, 0],
                    actual["proprio_latent"][:, -1],
                    d["instruction_embedding"],
                    s["remaining_horizon"] - s["execution_prefix"],
                )
                actual_error = float(
                    np.max(np.abs(offline - scores["actual_future_offline"]))
                )
                if actual_error > 5e-6:
                    raise AssertionError(
                        "Actual-future diagnostic score reproduction failed"
                    )
                if any(
                    int(np.argmax(offline[:bank]))
                    != int(np.argmax(scores["actual_future_offline"][:bank]))
                    for bank in [4, 8]
                ):
                    raise AssertionError("Nested-bank offline choice changed")
                score_checks[-1]["offline_actual_score_error"] = actual_error
        repeat_variability.append(
            {
                "state_id": state_id,
                "candidates": s["candidate_count"],
                "variable_candidates": int(np.sum(labels.min(0) != labels.max(0))),
                "base_variable": bool(labels[:, 0].min() != labels[:, 0].max()),
                "success_frequency_by_candidate": labels.mean(0).tolist(),
            }
        )
        probability_controls = {
            "global_prior": np.full(s["candidate_count"], priors["global_prior"]),
            "task_prior": np.full(
                s["candidate_count"], priors["task_priors"][s["task"]]
            ),
        }
        for name, p in (scores | probability_controls).items():
            # Average binary Brier over repeats, not squared error to mean labels.
            eligible = np.asarray(s["scorer_training_eligible"], bool)
            if eligible.any():
                error_rows.append(
                    {
                        "episode_id": s["episode_id"],
                        "method": name,
                        "brier": float(
                            ((p[None, :] - labels) ** 2)[:, eligible].mean()
                        ),
                    }
                )
        for bank in [4, 8]:
            if s["candidate_count"] < bank:
                raise ValueError("Declared nested bank missing")
            eligible_bank = all(s["scorer_training_eligible"][:bank])
            if not eligible_bank:
                omitted[str(bank)] += 1
                exclusions.append(
                    {
                        "state_id": state_id,
                        "bank": bank,
                        "early_prefix_success_candidates": [
                            i
                            for i, v in enumerate(s["scorer_training_eligible"][:bank])
                            if not v
                        ],
                    }
                )
            y = labels[:, :bank]
            base = y[:, 0]
            mixed = bool(np.any(y.min(1) != y.max(1)))
            choices = {
                "base": 0,
                "always_candidate_one": 1,
                "uniform_random_including_base": None,
                **{name: int(np.argmax(p[:bank])) for name, p in scores.items()},
                "simulator_upper_bound": "oracle",
            }
            for name, c in choices.items():
                if c is None:
                    selected = float(y.mean())
                    helps = float(((y - base[:, None]) > 0).mean())
                    hurts = float(((y - base[:, None]) < 0).mean())
                elif c == "oracle":
                    # Pick one fixed candidate using its three-repeat frequency.
                    c = int(np.argmax(y.mean(0)))
                    selected = float(y[:, c].mean())
                    helps = float((y[:, c] > base).mean())
                    hurts = float((y[:, c] < base).mean())
                else:
                    selected = float(y[:, c].mean())
                    helps = float((y[:, c] > base).mean())
                    hurts = float((y[:, c] < base).mean())
                row = {
                    "episode_id": s["episode_id"],
                    "state_id": state_id,
                    "selected_index": c,
                    "success": selected,
                    "base_success": float(base.mean()),
                    "gain": selected - float(base.mean()),
                    "help_frequency": helps,
                    "harm_frequency": hurts,
                    "mixed_bank": mixed,
                }
                all_decisions[str(bank)][name].append(row)
                if eligible_bank:
                    decisions[str(bank)][name].append(row)
    metrics = {
        name: episode_mean([r for r in error_rows if r["method"] == name], "brier")
        for name in heads.keys()
        | {"predicted_future", "actual_future_offline", "global_prior", "task_prior"}
    }

    def summarize(decision_rows):
        return {
            bank: {
                name: {
                    "states": len(rows),
                    "episode_mean_success": episode_mean(rows, "success"),
                    "episode_mean_gain_vs_base": episode_mean(rows, "gain"),
                    "episode_mean_help_frequency": episode_mean(rows, "help_frequency"),
                    "episode_mean_harm_frequency": episode_mean(rows, "harm_frequency"),
                    "mixed_states": sum(r["mixed_bank"] for r in rows),
                    "decisions": rows,
                }
                for name, rows in methods.items()
            }
            for bank, methods in decision_rows.items()
        }

    controls = summarize(decisions)
    all_controls = summarize(all_decisions)
    tasks = sorted({ep["task"] for ep in summary["episodes"]})
    per_task_brier = {
        task: {
            name: episode_mean(
                [
                    r
                    for r in error_rows
                    if r["method"] == name
                    and r["episode_id"].startswith(task + "_seed")
                ],
                "brier",
            )
            for name in metrics
        }
        for task in tasks
    }
    return {
        "status": "passed",
        "source_run": source.name,
        "source_episodes": len(summary["episodes"]),
        "source_policy_episode_successes": sum(
            ep["success"] for ep in summary["episodes"]
        ),
        "states": len(states),
        "candidate_continuation_labels": sum(
            len(s["branch_repeats"]) * s["candidate_count"] for s in states
        ),
        "eligible_candidate_continuation_labels": sum(
            len(s["branch_repeats"]) * sum(s["scorer_training_eligible"])
            for s in states
        ),
        "brier_population": "Prefix-eligible candidates only, identical masks for every method; repeat means then state means then complete episode means",
        "per_task_binary_brier_over_repeats": per_task_brier,
        "exact_repeat_replay_checks": repeat_count,
        "omitted_selection_states": omitted,
        "episode_mean_binary_brier_over_repeats": metrics,
        "alternative_paired_repeat_labels": candidate_effect_counts,
        "continuation_variability": repeat_variability,
        "controls": controls,
        "all_state_controls": all_controls,
        "early_prefix_exclusions": exclusions,
        "fresh_deployment_verification": score_checks,
        "training_only_priors": priors,
        "score_verification_atol": 5e-6,
        "score_verification_history": "Initial 1e-6 absolute check was too tight for cross-system float32 reductions: observed max 1.3411e-6 across predicted/actual scores, with every nested-bank argmax unchanged. Immutable VM scores are used; no model or decision was changed.",
        "selection_populations": "All-state controls include every collected decision and are deployment-relevant; controls retains legacy all-prefix-eligible subset for comparability. Eligibility is offline instrumentation, never deployment input.",
        "runtime_seconds": summary["elapsed_seconds"],
        "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        "readout_checkpoint_sha256": readout.spec["checkpoint_sha256"],
        "comparator_sha256": {
            k: hashlib.sha256((frozen / f"{k}_checkpoint.npz").read_bytes()).hexdigest()
            for k in heads
        },
        "scope": "Fresh development shard, three finite matched seeds; episode-disjoint from scorer fitting; not final test or closed-loop performance",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.source, a.frozen)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "controls",
                    "all_state_controls",
                    "fresh_deployment_verification",
                }
            },
            indent=2,
        )
    )
