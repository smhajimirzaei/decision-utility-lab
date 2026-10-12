"""Declared ranking fits on cached development data; old test arrays excluded."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .pairwise_ranking import (
    FrozenRanker,
    fit_ranker,
    grouped_states,
    pair_metrics,
    pairs,
    selection_metrics,
    utility,
)
from .readout_study import linear_predict
from .task_evaluator import encode_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def development(study, config):
    rows = json.loads((study / "rows.json").read_text())
    seeds = set(config["train_seeds"]) | set(config["validation_seeds"])
    if set(config["train_seeds"]) & set(config["validation_seeds"]) or seeds & set(
        config["excluded_source_seeds"]
    ):
        raise ValueError("Declared development splits overlap")
    for row in rows:
        expected = "train" if row["seed"] in config["train_seeds"] else "validation"
        if row["seed"] not in seeds or row["split"] != expected:
            raise ValueError("Cached row crosses the declared episode split")
        row["outcomes"] = [int(row["success"])]
    cache = load_checkpoint(study / "development_feature_cache.npz")
    for key in ["actual", "predicted", "current", "actions"]:
        if len(cache[key]) != len(rows) or not np.isfinite(cache[key]).all():
            raise ValueError("Cached feature alignment/finite check failed")
    grouped_states(rows)
    return rows, cache


def choose(trials):
    best = min(t["validation"]["episode_weighted_mean_effect_mse"] for t in trials)
    near = [
        t
        for t in trials
        if t["validation"]["episode_weighted_mean_effect_mse"] <= best + 1e-9
    ]
    complexity = {"null": 0, "absolute": 1, "linear": 2, "linear_plus_fixed_tanh": 3}
    return min(
        near,
        key=lambda t: (
            complexity.get(t["kind"], complexity.get(t["representation"], 3)),
            -t.get("l2", 0),
            t["name"],
        ),
    )


def frozen_descriptor(output, family, selected, model, maps, train_rows, val_rows):
    file = output / f"selected_{family}_ranker.npz"
    np.savez_compressed(file, **model, **maps)
    spec = {
        "input_family": family,
        "name": selected["name"],
        "kind": selected["kind"],
        "representation": selected["representation"],
        "l2": selected.get("l2"),
        "checkpoint": file.name,
        "checkpoint_sha256": digest(file),
        "training_episode_ids": sorted({r["episode_id"] for r in train_rows}),
        "validation_episode_ids": sorted({r["episode_id"] for r in val_rows}),
        "validation_pair_effect_mse": selected["validation"][
            "episode_weighted_mean_effect_mse"
        ],
        "output": "Ranking utility; not continuation success probability",
        "fresh_outcomes_used_for_fit_or_selection": False,
        "deployment_input": "Encoded camera, ordinary proprioception, instruction embedding, horizon; direct model additionally takes proposed ten-action prefixes",
    }
    path = output / f"selected_{family}_ranker.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def load_fresh(source, config, maps):
    summary = json.loads((source / "summary.json").read_text())
    if summary["status"] != "passed":
        raise ValueError("Complete fresh source required")
    rows = []
    bank = {key: [] for key in ["actual", "predicted", "current", "actions"]}
    opened_states = []
    for episode in summary["episodes"]:
        if episode["seed"] not in config["fresh_diagnostic_seeds"]:
            raise ValueError("Unexpected fresh diagnostic seed; refuse feature access")
        for state in episode["states"]:
            sid = f"{state['episode_id']}/state{state['state_step']}"
            if len(state["branch_repeats"]) != 3 or any(
                v != 0
                for repeat in state["continuation_replay_by_repeat"]
                for v in repeat.values()
            ):
                raise ValueError("Declared matched-repeat audit failed")
            with np.load(
                source / sid / "features/deployment_inputs.npz", allow_pickle=False
            ) as d:
                with np.load(
                    source / sid / "outcomes/actual_future_features.npz",
                    allow_pickle=False,
                ) as a:
                    remaining = state["remaining_horizon"] - state["execution_prefix"]
                    actual = encode_features(
                        a["latent"][:, -1, 0],
                        a["proprio_latent"][:, -1],
                        d["instruction_embedding"],
                        remaining,
                        maps,
                    )
                    predicted = encode_features(
                        d["predicted_latents"][:, -1, 0],
                        d["predicted_proprio_latents"][:, -1],
                        d["instruction_embedding"],
                        remaining,
                        maps,
                    )
                    current = encode_features(
                        d["current_latent"][0, 0, 0],
                        d["current_proprio_latent"][0, 0],
                        d["instruction_embedding"],
                        state["remaining_horizon"],
                        maps,
                    )
                    bank["actual"].extend(actual)
                    bank["predicted"].extend(predicted)
                    bank["current"].extend(np.broadcast_to(current, actual.shape))
                    bank["actions"].extend(
                        d["executed_prefixes"].reshape(len(actual), 40)
                    )
            for candidate in range(state["candidate_count"]):
                rows.append(
                    {
                        "episode_id": state["episode_id"],
                        "state_id": sid,
                        "task": state["task"],
                        "seed": state["seed"],
                        "split": "fresh_diagnostic",
                        "candidate_index": candidate,
                        "eligible": state["scorer_training_eligible"][candidate],
                        "outcomes": [
                            int(repeat[candidate]["success"])
                            for repeat in state["branch_repeats"]
                        ],
                    }
                )
            opened_states.append(sid)
    return (
        rows,
        {k: np.asarray(v, dtype=np.float32) for k, v in bank.items()},
        opened_states,
    )


def evaluate(rows, features, frozen_heads, selected, selected_heads):
    direct = np.concatenate((features["current"], features["actions"]), -1)
    scores = {
        "frozen_actual_future": (
            linear_predict(frozen_heads["actual_future"], features["actual"]),
            "absolute",
        ),
        "frozen_predicted_future": (
            linear_predict(frozen_heads["actual_future"], features["predicted"]),
            "absolute",
        ),
        "frozen_direct_action": (
            linear_predict(frozen_heads["direct_action"], direct),
            "absolute",
        ),
        "frozen_current_only": (
            linear_predict(frozen_heads["current_only"], features["current"]),
            "absolute",
        ),
        "selected_actual_ranker_offline": (
            utility(
                selected["actual_future"]["kind"],
                selected_heads["actual_future"],
                features["actual"],
            ),
            selected["actual_future"]["kind"],
        ),
        "selected_predicted_ranker": (
            utility(
                selected["actual_future"]["kind"],
                selected_heads["actual_future"],
                features["predicted"],
            ),
            selected["actual_future"]["kind"],
        ),
        "selected_direct_ranker": (
            utility(
                selected["direct_action"]["kind"],
                selected_heads["direct_action"],
                direct,
            ),
            selected["direct_action"]["kind"],
        ),
    }
    result = {
        "pair_metrics": {
            name: pair_metrics(rows, score, kind)
            for name, (score, kind) in scores.items()
        },
        "episode_ids": sorted({r["episode_id"] for r in rows}),
    }
    methods = {
        "base": "base",
        "always_candidate_one": "one",
        "uniform_random_including_base": "random",
        "best_measured_candidate_offline": "oracle",
        **{k: v[0] for k, v in scores.items()},
    }
    for population, eligible in [("all_states", False), ("prefix_eligible", True)]:
        result[population] = {
            str(b): {
                name: selection_metrics(rows, score, b, eligible)
                for name, score in methods.items()
            }
            for b in [4, 8]
        }
    return result, {name: score for name, (score, kind) in scores.items()}


def deployment_checks(source, rows, scores, output):
    future = FrozenRanker(output / "selected_actual_future_ranker.json")
    direct = FrozenRanker(output / "selected_direct_action_ranker.json")
    checks = []
    for sid, indices in grouped_states(rows).items():
        step = int(sid.split("/state")[-1])
        with np.load(
            source / sid / "features/deployment_inputs.npz", allow_pickle=False
        ) as d:
            p = future.score_future(
                d["predicted_latents"][:, -1, 0],
                d["predicted_proprio_latents"][:, -1],
                d["instruction_embedding"],
                400 - step - 10,
            )
            q = direct.score_direct(
                d["current_latent"][0, 0, 0],
                d["current_proprio_latent"][0, 0],
                d["instruction_embedding"],
                400 - step,
                d["executed_prefixes"],
            )
        for name, actual in [
            ("selected_predicted_ranker", p),
            ("selected_direct_ranker", q),
        ]:
            expected = scores[name][indices]
            error = float(np.max(np.abs(actual - expected)))
            choices_match = all(
                int(np.argmax(actual[:b])) == int(np.argmax(expected[:b]))
                for b in [4, 8]
            )
            if error > 1e-8 or not choices_match:
                raise AssertionError(
                    f"Deployment adapter mismatch: {sid}/{name}/{error}"
                )
            checks.append(
                {
                    "state_id": sid,
                    "method": name,
                    "utilities": len(actual),
                    "max_abs_error": error,
                    "both_bank_choices_match": choices_match,
                }
            )
    return {
        "status": "passed",
        "checks": checks,
        "output_semantics": "utility, not probability",
    }


def run(study, frozen, fresh, config_path, output):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "declared_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, cache = development(study, config)
    maps = {key: cache[key] for key in ["visual_projection", "text_projection"]}
    frozen_heads = {
        name: load_checkpoint(frozen / f"{file}_checkpoint.npz")
        for name, file in [
            ("actual_future", "selected"),
            ("direct_action", "direct_action"),
            ("current_only", "current_only"),
        ]
    }
    for head in frozen_heads.values():
        if any(not np.array_equal(head[key], maps[key]) for key in maps):
            raise ValueError("Frozen feature maps disagree")
    mask_train = np.asarray([r["split"] == "train" and r["eligible"] for r in rows])
    mask_val = np.asarray([r["split"] == "validation" for r in rows])
    train_rows = [r for r, keep in zip(rows, mask_train) if keep]
    val_rows = [r for r, keep in zip(rows, mask_val) if keep]
    train_pairs = pairs(train_rows)
    direct_features = np.concatenate((cache["current"], cache["actions"]), -1)
    trials = []
    chosen = {}
    chosen_heads = {}
    models = {}
    trial_dir = output / "trials"
    trial_dir.mkdir()
    for family, features in [
        ("actual_future", cache["actual"]),
        ("direct_action", direct_features),
    ]:
        candidates = []
        for kind in ["null", "absolute"]:
            head = {} if kind == "null" else frozen_heads[family]
            scores = utility(kind, head, features)
            trial = {
                "name": family + "_" + kind,
                "input_family": family,
                "kind": kind,
                "representation": kind,
                "train": pair_metrics(train_rows, scores[mask_train], kind),
                "validation": pair_metrics(val_rows, scores[mask_val], kind),
            }
            candidates.append(trial)
            models[trial["name"]] = head
        for representation in config["representations"]:
            for objective in config["objectives"]:
                for l2 in config["l2_grid"]:
                    head = fit_ranker(
                        features[mask_train],
                        train_pairs,
                        objective,
                        representation,
                        l2,
                        config["optimizer"]["logistic_updates"],
                        config["optimizer"]["learning_rate"],
                    )
                    scores = utility(objective, head, features)
                    name = f"{family}_{representation}_{objective}_{l2:g}"
                    file = trial_dir / (name + ".npz")
                    np.savez_compressed(file, **head)
                    trial = {
                        "name": name,
                        "input_family": family,
                        "kind": objective,
                        "representation": representation,
                        "l2": l2,
                        "checkpoint_sha256": digest(file),
                        "checkpoint": str(file.relative_to(output)),
                        "train": pair_metrics(
                            train_rows, scores[mask_train], objective
                        ),
                        "validation": pair_metrics(
                            val_rows, scores[mask_val], objective
                        ),
                    }
                    candidates.append(trial)
                    models[name] = head
                    print(
                        f"RANKING_FIT {name} val_effect_mse={trial['validation']['episode_weighted_mean_effect_mse']:.8f}",
                        flush=True,
                    )
        winner = choose(candidates)
        chosen[family] = winner
        chosen_heads[family] = models[winner["name"]]
        # Frozen heads already contain maps; exclude those duplicate keys here.
        head = {k: v for k, v in chosen_heads[family].items() if k not in maps}
        frozen_descriptor(output, family, winner, head, maps, train_rows, val_rows)
        trials.extend(candidates)
    (output / "trial_ledger.json").write_text(json.dumps(trials, indent=2) + "\n")
    (output / "selected_before_fresh.json").write_text(
        json.dumps(chosen, indent=2) + "\n"
    )
    print("RANKERS_FROZEN_BEFORE_FRESH_FEATURE_ACCESS", flush=True)
    diagnostic = {}
    for split in ["train", "validation"]:
        mask = np.asarray([r["split"] == split for r in rows])
        subset = [r for r, keep in zip(rows, mask) if keep]
        diagnostic[split], _ = evaluate(
            subset,
            {k: cache[k][mask] for k in ["actual", "predicted", "current", "actions"]},
            frozen_heads,
            chosen,
            chosen_heads,
        )
    fresh_rows, fresh_features, accessed = load_fresh(fresh, config, maps)
    diagnostic["fresh_diagnostic"], fresh_scores = evaluate(
        fresh_rows, fresh_features, frozen_heads, chosen, chosen_heads
    )
    (output / "fresh_rows.json").write_text(json.dumps(fresh_rows, indent=2) + "\n")
    np.savez_compressed(output / "fresh_utilities.npz", **fresh_scores)
    checks = deployment_checks(fresh, fresh_rows, fresh_scores, output)
    (output / "deployment_verification.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    result = {
        "status": "passed",
        "scope": "Development-only scoring/ranking diagnosis, no independent test or gate",
        "config": config,
        "selected": chosen,
        "diagnostics": diagnostic,
        "new_fits": sum("l2" in trial for trial in trials),
        "selection_candidates_including_controls": len(trials),
        "fresh_model_parameters_changed": False,
        "fresh_features_accessed_after_selection": accessed,
        "training_coverage": pair_metrics(
            train_rows, np.zeros(len(train_rows)), "null"
        ),
        "validation_coverage": pair_metrics(val_rows, np.zeros(len(val_rows)), "null"),
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": {
            "config_sha256": digest(config_path),
            "implementation_sha256": digest(Path(__file__)),
            "pairwise_implementation_sha256": digest(
                Path(__file__).with_name("pairwise_ranking.py")
            ),
            "development_rows_sha256": digest(study / "rows.json"),
            "development_cache_sha256": digest(study / "development_feature_cache.npz"),
            "fresh_summary_sha256": digest(fresh / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ["status", "scope", "new_fits", "elapsed_seconds"]
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["study", "frozen", "fresh", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.frozen, args.fresh, args.config, args.output)
