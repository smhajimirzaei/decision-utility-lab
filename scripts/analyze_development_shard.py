"""Evaluate already frozen ranking models on a declared development shard."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from trust_imagination.ranking_study import (
    deployment_checks,
    evaluate,
    load_checkpoint,
    load_fresh,
)


def analyze(source, config_path, frozen, ranking, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    summary = json.loads((source / "summary.json").read_text())
    expected = {(task, seed) for task in config["tasks"] for seed in config["seeds"]}
    actual = [(ep["task"], ep["seed"]) for ep in summary["episodes"]]
    if (
        summary["status"] != "passed"
        or len(actual) != len(expected)
        or set(actual) != expected
    ):
        raise ValueError("Complete declared task x seed grid required")
    for ep in summary["episodes"]:
        for state in ep["states"]:
            if (
                state["candidate_count"] != config["candidate_count"]
                or state["execution_prefix"] != config["execution_prefix"]
                or state["state_step"] not in config["decision_steps"]
                or not state["simulator_replay"]["passed"]
            ):
                raise ValueError("State differs from declared collection contract")
    heads = {
        name: load_checkpoint(frozen / f"{file}_checkpoint.npz")
        for name, file in [
            ("actual_future", "selected"),
            ("direct_action", "direct_action"),
            ("current_only", "current_only"),
        ]
    }
    maps = {
        key: heads["actual_future"][key]
        for key in ["visual_projection", "text_projection"]
    }
    selected = {}
    rank_heads = {}
    contracts = {}
    for family in ["actual_future", "direct_action"]:
        descriptor = ranking / f"selected_{family}_ranker.json"
        spec = json.loads(descriptor.read_text())
        checkpoint = ranking / spec["checkpoint"]
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha != spec["checkpoint_sha256"]:
            raise ValueError("Previously frozen ranker changed")
        rank_heads[family] = load_checkpoint(checkpoint)
        if any(not np.array_equal(rank_heads[family][key], maps[key]) for key in maps):
            raise ValueError("Frozen maps differ")
        selected[family] = spec
        contracts[family] = {
            "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
            "checkpoint_sha256": sha,
        }
        # Native verification loads frozen descriptors from the output folder.
        (output / descriptor.name).write_bytes(descriptor.read_bytes())
        (output / checkpoint.name).write_bytes(checkpoint.read_bytes())
    rows, features, accessed = load_fresh(
        source, {"fresh_diagnostic_seeds": config["seeds"]}, maps
    )
    metrics, scores = evaluate(rows, features, heads, selected, rank_heads)
    native = deployment_checks(source, rows, scores, output)
    result = {
        "status": "passed",
        "source_run": source.name,
        "split": "development",
        "no_fits_or_model_selection": True,
        "source_grid": sorted(expected),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_summary_sha256": hashlib.sha256(
            (source / "summary.json").read_bytes()
        ).hexdigest(),
        "frozen_contracts": contracts,
        "diagnostics": metrics,
        "feature_states_opened": accessed,
        "scope": "Fixed representative development shard; post-freeze diagnosis, not an independent final test or closed-loop result",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "deployment_verification.json").write_text(
        json.dumps(native, indent=2) + "\n"
    )
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(output / "utilities.npz", **scores)
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_grid": result["source_grid"],
                "all_state_eight_candidate_results": {
                    k: {q: v for q, v in m.items() if q != "decisions"}
                    for k, m in metrics["all_states"]["8"].items()
                },
                "pair_metrics": metrics["pair_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["source", "config", "frozen", "ranking", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    analyze(a.source, a.config, a.frozen, a.ranking, a.output)
