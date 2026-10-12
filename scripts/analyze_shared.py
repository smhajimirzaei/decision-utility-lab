"""Summarize paired labels without confusing alternatives with a learned selector."""

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def analyze(root):
    summary = json.loads((root / "summary.json").read_text())
    states = [state for episode in summary["episodes"] for state in episode["states"]]
    comparisons = Counter(
        label
        for state in states
        for label in state["alternative_labels_relative_to_candidate0"][1:]
    )
    mse = np.asarray([state["model_mse"] for state in states])
    persistence = np.asarray([state["persistence_mse"] for state in states])
    report = {
        "run_id": root.name,
        "status": summary["status"],
        "episodes": len(summary["episodes"]),
        "baseline_successes": sum(ep["success"] for ep in summary["episodes"]),
        "decision_states": len(states),
        "alternative_comparisons": dict(comparisons),
        "states_with_any_helpful_alternative": sum(
            "helps" in state["alternative_labels_relative_to_candidate0"][1:]
            for state in states
        ),
        "states_with_any_harmful_alternative": sum(
            "hurts" in state["alternative_labels_relative_to_candidate0"][1:]
            for state in states
        ),
        "replay_passed": all(
            state["simulator_replay"]["passed"]
            and all(
                value == 0
                for value in state["continuation_replay_max_abs_errors"].values()
            )
            for state in states
        ),
        "mean_prefix_rms_distance": float(
            np.mean(
                [state["proposal_prefix_rms_distance_from_base"] for state in states]
            )
        ),
        "model_lower_than_persistence": int((mse < persistence).sum()),
        "correlated_feature_comparisons": int(mse.size),
        "mean_model_mse": float(mse.mean()),
        "mean_persistence_mse": float(persistence.mean()),
        "median_bank_unroll_seconds": float(
            np.median(
                [
                    state.get(
                        "candidate_bank_two_step_unroll_seconds",
                        state.get("four_candidate_two_step_unroll_seconds"),
                    )
                    for state in states
                ]
            )
        ),
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        "eligible_candidate_labels": sum(
            sum(state["scorer_training_eligible"]) for state in states
        ),
        "scope": "Unscored alternative outcomes, correlated within states/episodes; no learned reranking or gate",
    }
    for size in (4, 8):
        selected = [state for state in states if state["candidate_count"] >= size]
        if selected:
            report[f"bank{size}_simulator_selector_success_states"] = sum(
                any(b["success"] for b in state["branches"][:size])
                for state in selected
            )
            report[f"bank{size}_base_success_states"] = sum(
                state["branches"][0]["success"] for state in selected
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
