"""Local, post-inspection cadence features and whole-seed evaluator diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pool(visual, projection, representation):
    x = np.asarray(visual, dtype=np.float32)
    if x.shape[-3:] != (16, 16, 384):
        raise ValueError("Expected native16x16x384 latent grid")
    if representation == "global":
        return x.mean(axis=(-3, -2)) @ projection
    if representation == "spatial":
        coarse = x.reshape(*x.shape[:-3], 4, 4, 4, 4, 384).mean(axis=(-4, -2))
        return (coarse @ projection).reshape(*x.shape[:-3], 256)
    raise ValueError("Unknown representation")


def complete_support(mask, early):
    mask = np.asarray(mask, dtype=bool)
    early = np.asarray(early, dtype=bool)
    if mask.shape != (8, 10) or early.shape != (8,):
        raise ValueError("Eight candidates, ten transitions required")
    return bool(mask[:, 9].all() and not early.any())


def inventory(source, output, config_path, analysis):
    cfg = read(config_path)
    audit = read(analysis / "summary.json")
    if (
        source.name != cfg["source_run"]
        or audit["status"] != "verified"
        or audit["source_run"] != source.name
    ):
        raise ValueError("Verified declared source required")
    output.mkdir(parents=True, exist_ok=False)
    tasks = read(source / "declared_protocol.json")["tasks"]
    rng = np.random.default_rng(cfg["channel_projection_seed"])
    projection = rng.normal(0, 1 / np.sqrt(384), (384, 16)).astype(np.float32)
    features = {}
    rows = []
    states = []
    provenance = []
    started = time.perf_counter()
    shapes = {}
    for dataset in ("main", "pilot"):
        summary_path = source / dataset / "summary.json"
        summary = read(summary_path)
        expected = set(cfg["main_seeds"]) if dataset == "main" else {cfg["pilot_seed"]}
        if {e["seed"] for e in summary["episodes"]} != expected:
            raise ValueError("Unexpected source seed groups")
        for episode in summary["episodes"]:
            for state in episode["states"]:
                state_id = f"{dataset}/{state['episode_id']}/state{state['state_step']}"
                folder = source / state_id
                deployment = folder / "features/deployment_inputs.npz"
                future = folder / "outcomes/actual_future_features.npz"
                provenance.extend(
                    {
                        "path": str(p.relative_to(source)),
                        "sha256": sha(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in (deployment, future)
                )
                with (
                    np.load(deployment, allow_pickle=False) as d,
                    np.load(future, allow_pickle=False) as a,
                ):
                    values = {k: d[k] for k in d.files}
                    actual = {k: a[k] for k in a.files}
                layout = {
                    "deployment": {k: list(v.shape) for k, v in values.items()},
                    "actual": {k: list(v.shape) for k, v in actual.items()},
                }
                if shapes and shapes != layout:
                    raise ValueError("Array schema changed within source")
                shapes = layout
                if any(
                    not np.isfinite(v).all()
                    for v in [*values.values(), *actual.values()]
                ):
                    raise ValueError("Nonfinite source features")
                actions = values["executed_full_proposals"]
                if (
                    not np.array_equal(actions, np.clip(values["raw_proposals"], -1, 1))
                    or sha_bytes(actions) != state["proposal_sha256"]
                ):
                    raise ValueError("Shared proposal identity differs")
                mask = actual["valid_mask"]
                expected_mask = (
                    np.arange(5, 51, 5)[None, :]
                    <= actual["first_stop_or_limit"][:, None]
                )
                if not np.array_equal(mask, expected_mask):
                    raise ValueError("Physical first-stop mask mismatch")
                conditions = {c["execution_prefix"]: c for c in state["conditions"]}
                common = complete_support(
                    mask, conditions[50]["early_success_within_prefix"]
                )
                instruction = state["instruction"]
                record = {
                    "state_id": state_id,
                    "episode_id": state["episode_id"],
                    "dataset": dataset,
                    "seed": state["seed"],
                    "task": state["task"],
                    "step": state["state_step"],
                    "instruction": instruction,
                    "common_support": common,
                    "valid_candidates_by_endpoint": {
                        str(p): int(mask[:, p // 5 - 1].sum()) for p in (10, 50)
                    },
                    "early_success_counts": {
                        str(p): sum(conditions[p]["early_success_within_prefix"])
                        for p in (10, 50)
                    },
                    "distinct_full_proposals": len({x.tobytes() for x in actions}),
                    "mean_pair_action_rms": {
                        str(p): float(
                            np.mean(
                                [
                                    np.sqrt(
                                        np.mean((actions[i, :p] - actions[j, :p]) ** 2)
                                    )
                                    for i in range(8)
                                    for j in range(i + 1, 8)
                                ]
                            )
                        )
                        for p in (10, 50)
                    },
                }
                states.append(record)
                for prefix in cfg["prefixes"]:
                    condition = conditions[prefix]
                    if condition["remaining_horizon"] != 400 - state["state_step"]:
                        raise ValueError("Outcome endpoint mismatch")
                    outcomes = np.asarray(
                        [
                            [int(b["success"]) for b in repeat]
                            for repeat in condition["branch_repeats"]
                        ]
                    ).T
                    if outcomes.shape != (8, 3):
                        raise ValueError("Three matched repeats required")
                    for candidate in range(8):
                        rows.append(
                            {
                                **{
                                    k: record[k]
                                    for k in (
                                        "state_id",
                                        "episode_id",
                                        "dataset",
                                        "seed",
                                        "task",
                                        "step",
                                        "common_support",
                                    )
                                },
                                "prefix": prefix,
                                "candidate_index": candidate,
                                "outcomes": outcomes[candidate].tolist(),
                            }
                        )
                    task = np.eye(len(tasks), dtype=np.float32)[
                        tasks.index(state["task"])
                    ]
                    extra = np.r_[
                        values["proprio"],
                        task,
                        (400 - state["state_step"] - prefix) / 400,
                    ].astype(np.float32)
                    for rep in cfg["representations"]:
                        current = np.r_[
                            pool(values["current_latent"][0, 0, 0], projection, rep),
                            values["current_proprio_latent"][0, 0].mean(0),
                        ]
                        repeated = np.broadcast_to(
                            np.r_[current, extra], (8, len(current) + len(extra))
                        )
                        endpoint = prefix // 5 - 1
                        av = np.c_[
                            pool(actual["latent"][:, endpoint, 0], projection, rep),
                            actual["proprio_latent"][:, endpoint].mean(1),
                        ]
                        pv = np.c_[
                            pool(
                                values["predicted_latents"][:, endpoint, 0],
                                projection,
                                rep,
                            ),
                            values["predicted_proprio_latents"][:, endpoint].mean(1),
                        ]
                        banks = {
                            "current": repeated,
                            "direct": np.c_[
                                repeated, actions[:, :prefix].reshape(8, -1)
                            ],
                            "actual": np.c_[repeated, av],
                            "predicted": np.c_[repeated, pv],
                        }
                        for name, x in banks.items():
                            # Invalid actual endpoint rows are deliberately absent as NaNs.
                            # Common-support rows are guaranteed finite before any fit.
                            if name == "actual":
                                x = x.copy()
                                x[~mask[:, endpoint]] = np.nan
                            features.setdefault(f"{rep}_{name}", []).extend(
                                x.tolist()
                                if name != "direct"
                                else np.pad(x, ((0, 0), (0, 200 - prefix * 4))).tolist()
                            )
                print(
                    f"INVENTORY {state_id} complete_physical_bank={common}", flush=True
                )
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}
    # Direct prefixes have a common storage width only; zero tails are removed
    # by the prefix-specific readout and never treated as additional actions.
    np.savez_compressed(output / "feature_cache.npz", projection=projection, **arrays)
    write(output / "rows.json", rows)
    write(output / "states.json", states)
    write(output / "source_files.json", provenance)
    summaries = {}
    for dataset in ("main", "pilot"):
        selected = [s for s in states if s["dataset"] == dataset]
        summaries[dataset] = {
            "episodes": len({s["episode_id"] for s in selected}),
            "states": len(selected),
            "common_support_states": sum(s["common_support"] for s in selected),
            "excluded_states": [
                s["state_id"] for s in selected if not s["common_support"]
            ],
            "common_support_episodes": sorted(
                {s["episode_id"] for s in selected if s["common_support"]}
            ),
            "valid_candidate_endpoints": {
                str(p): sum(s["valid_candidates_by_endpoint"][str(p)] for s in selected)
                for p in (10, 50)
            },
        }
    write(
        output / "inventory.json",
        {
            "status": "verified",
            "source_run": source.name,
            "config": cfg,
            "native_shapes": shapes,
            "spatial_grid_retained": True,
            "instruction_embedding_present": False,
            "current_raw_proprio_dimensions": 4,
            "feature_dimensions": {k: v.shape[1] for k, v in arrays.items()},
            "datasets": summaries,
            "rows": len(rows),
            "config_sha256": sha(config_path),
            "cache_sha256": sha(output / "feature_cache.npz"),
            "source_analysis_sha256": sha(analysis / "summary.json"),
            "implementation_sha256": sha(__file__),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(json.dumps(summaries, indent=2))


def sha_bytes(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def weights(rows):
    counts = Counter(r["episode_id"] for r in rows)
    return np.asarray([1 / (len(counts) * counts[r["episode_id"]]) for r in rows])


def fit(x, y, w, cfg):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not len(x):
        raise ValueError("Finite complete training data required")
    mean = w @ x
    scale = np.maximum(np.sqrt(w @ ((x - mean) ** 2)), cfg["normalization_floor"])
    z = np.clip(
        (x - mean) / scale, -cfg["normalization_clip"], cfg["normalization_clip"]
    )
    center = w @ z
    intercept = float(w @ y)
    zc = z - center
    weighted = zc * np.sqrt(w[:, None])
    target = (y - intercept) * np.sqrt(w)
    coef = weighted.T @ np.linalg.solve(
        weighted @ weighted.T + cfg["l2"] * np.eye(len(x)), target
    )
    return {
        "mean": mean,
        "scale": scale,
        "center": center,
        "coef": coef,
        "intercept": np.asarray(intercept),
    }


def predict(head, x, cfg):
    if not np.isfinite(x).all():
        raise ValueError("Invalid/padded future cannot be scored")
    z = np.clip(
        (x - head["mean"]) / head["scale"],
        -cfg["normalization_clip"],
        cfg["normalization_clip"],
    )
    return np.clip((z - head["center"]) @ head["coef"] + head["intercept"], 0, 1)


def metrics(rows, scores, bank):
    groups = {}
    for i, r in enumerate(rows):
        if r["candidate_index"] < bank:
            groups.setdefault(r["state_id"], []).append(i)
    records = []
    pair_records = []
    for state, indices in groups.items():
        indices.sort(key=lambda i: rows[i]["candidate_index"])
        if [rows[i]["candidate_index"] for i in indices] != list(range(bank)):
            raise ValueError("Incomplete candidate bank")
        y = np.asarray([rows[i]["outcomes"] for i in indices], dtype=float)
        frequency = y.mean(1)
        if isinstance(scores, str):
            chosen = {
                "base": 0,
                "one": 1,
                "random": None,
                "offline_best": int(np.argmax(frequency)),
            }[scores]
        else:
            s = np.asarray(scores)[indices]
            chosen = int(np.flatnonzero(s >= s.max() - 1e-10)[0])
        value = frequency.mean() if chosen is None else frequency[chosen]
        delta = y - y[0] if chosen is None else y[chosen] - y[0]
        record = {
            "state_id": state,
            "episode_id": rows[indices[0]]["episode_id"],
            "chosen": chosen,
            "success": float(value),
            "base": float(frequency[0]),
            "gain": float(value - frequency[0]),
            "help": float((delta > 0).mean()),
            "harm": float((delta < 0).mean()),
            "regret": float(frequency.max() - value),
            "helpful_opportunity": bool(frequency.max() > frequency[0]),
            "informative": bool(frequency.max() > frequency.min()),
        }
        records.append(record)
        if not isinstance(scores, str):
            pair_values = []
            for i in range(bank):
                for j in range(i + 1, bank):
                    effect = frequency[i] - frequency[j]
                    if abs(effect) > 1e-12:
                        score = s[i] - s[j]
                        pair_values.append(
                            0.5
                            if abs(score) <= 1e-10
                            else float(np.sign(score) == np.sign(effect))
                        )
            if pair_values:
                pair_records.append(
                    {
                        "episode_id": record["episode_id"],
                        "ordering": float(np.mean(pair_values)),
                    }
                )

    def average(items, key):
        episodes = {r["episode_id"] for r in items}
        return (
            float(
                np.mean(
                    [
                        np.mean([r[key] for r in items if r["episode_id"] == ep])
                        for ep in episodes
                    ]
                )
            )
            if episodes
            else None
        )

    result = {
        key: average(records, key)
        for key in ("success", "base", "gain", "help", "harm", "regret")
    }
    result.update(
        states=len(records),
        episodes=len({r["episode_id"] for r in records}),
        informative_states=sum(r["informative"] for r in records),
        helpful_states=sum(r["helpful_opportunity"] for r in records),
        informative_pair_ordering=average(pair_records, "ordering"),
        decisions=records,
    )
    if not isinstance(scores, str):
        indices = [i for i, r in enumerate(rows) if r["candidate_index"] < bank]
        selected = [rows[i] for i in indices]
        y = np.asarray([r["outcomes"] for r in selected], dtype=float)
        result["per_repeat_brier"] = float(
            weights(selected)
            @ np.mean((np.asarray(scores)[indices, None] - y) ** 2, axis=1)
        )
    return result


def study(audit, output, config_path):
    cfg = read(config_path)
    inventory_record = read(audit / "inventory.json")
    if inventory_record["config_sha256"] != sha(config_path) or inventory_record[
        "cache_sha256"
    ] != sha(audit / "feature_cache.npz"):
        raise ValueError("Declared feature contract/cache changed")
    output.mkdir(parents=True, exist_ok=False)
    all_rows = read(audit / "rows.json")
    supported = [i for i, r in enumerate(all_rows) if r["common_support"]]
    rows = [all_rows[i] for i in supported]
    with np.load(audit / "feature_cache.npz", allow_pickle=False) as a:
        features = {
            k: a[k][supported].astype(float) for k in a.files if k != "projection"
        }
    y = np.asarray([np.mean(r["outcomes"]) for r in rows])
    results = {}
    predictions = {}
    heads = 0
    started = time.perf_counter()
    for prefix in cfg["prefixes"]:
        ix = np.asarray([i for i, r in enumerate(rows) if r["prefix"] == prefix])
        pr = [rows[i] for i in ix]
        py = y[ix]
        main = np.asarray([r["dataset"] == "main" for r in pr])
        pilot = ~main
        full_controls = [
            r for r in all_rows if r["prefix"] == prefix and r["dataset"] == "main"
        ]
        contract = {
            "main_full_state_fixed_controls": {
                str(b): {
                    c: metrics(full_controls, c, b)
                    for c in ("base", "one", "random", "offline_best")
                }
                for b in cfg["banks"]
            },
            "representations": {},
        }
        for rep in cfg["representations"]:
            banks = {name: features[f"{rep}_{name}"][ix] for name in cfg["evaluators"]}
            current_width = banks["current"].shape[1]
            banks["direct"] = banks["direct"][:, : current_width + prefix * 4]
            scores = {
                name: np.full(len(pr), np.nan)
                for name in (*cfg["evaluators"], "actual_to_predicted")
            }
            folds = []
            for held in cfg["main_seeds"]:
                train = main & np.asarray([r["seed"] != held for r in pr])
                test = main & ~train
                train_rows = [r for r, m in zip(pr, train) if m]
                if (
                    not train.any()
                    or not test.any()
                    or set(r["seed"] for r in train_rows) & {held}
                ):
                    raise ValueError("Empty/leaking whole-seed fold")
                fold = {
                    "held_seed": held,
                    "train_episodes": sorted({r["episode_id"] for r in train_rows}),
                    "evaluation_episodes": sorted(
                        {r["episode_id"] for r, m in zip(pr, test) if m}
                    ),
                    "training_helpful_states": metrics(train_rows, "base", 8)[
                        "helpful_states"
                    ],
                    "metrics": {},
                }
                for name, x in banks.items():
                    head = fit(x[train], py[train], weights(train_rows), cfg)
                    heads += 1
                    scores[name][test] = predict(head, x[test], cfg)
                    np.savez_compressed(
                        output / f"prefix{prefix}_{rep}_{name}_held{held}.npz", **head
                    )
                    if name == "actual":
                        scores["actual_to_predicted"][test] = predict(
                            head, banks["predicted"][test], cfg
                        )
                test_rows = [r for r, m in zip(pr, test) if m]
                fold["metrics"] = {
                    name: metrics(test_rows, value[test], 8)
                    for name, value in scores.items()
                }
                folds.append(fold)
            # Pilot is separate, never used to train or select a main fold.
            main_rows = [r for r, m in zip(pr, main) if m]
            for name, x in banks.items():
                head = fit(x[main], py[main], weights(main_rows), cfg)
                heads += 1
                scores[name][pilot] = predict(head, x[pilot], cfg)
                np.savez_compressed(
                    output / f"prefix{prefix}_{rep}_{name}_pilot_only.npz", **head
                )
                if name == "actual":
                    scores["actual_to_predicted"][pilot] = predict(
                        head, banks["predicted"][pilot], cfg
                    )
            rep_result = {
                "dimensions": {k: v.shape[1] for k, v in banks.items()},
                "main_folds": folds,
            }
            for dataset, mask in (("main", main), ("pilot", pilot)):
                selected = [r for r, m in zip(pr, mask) if m]
                rep_result[dataset] = {
                    str(b): {
                        **{
                            name: metrics(selected, value[mask], b)
                            for name, value in scores.items()
                        },
                        **{
                            name: metrics(selected, name, b)
                            for name in ("base", "one", "random", "offline_best")
                        },
                    }
                    for b in cfg["banks"]
                }
            contract["representations"][rep] = rep_result
            for name, value in scores.items():
                predictions[f"prefix{prefix}_{rep}_{name}"] = value
        results[str(prefix)] = contract
        print(f"VALUE_DIAGNOSTIC prefix={prefix} complete", flush=True)
    np.savez_compressed(output / "predictions.npz", **predictions)
    write(
        output / "prediction_rows.json",
        {str(p): [r for r in rows if r["prefix"] == p] for p in cfg["prefixes"]},
    )
    write(
        output / "summary.json",
        {
            "status": "passed",
            "config": cfg,
            "config_sha256": sha(config_path),
            "inventory_sha256": sha(audit / "inventory.json"),
            "implementation_sha256": sha(__file__),
            "fits": heads,
            "results": results,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "Matched censored-support development diagnosis only; pilot separate; no model selected, no gate, no closed-loop benefit or independent-test claim",
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["inventory", "study"])
    p.add_argument("--source", type=Path)
    p.add_argument("--analysis", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cadence_value_diagnostic_20261011.json"),
    )
    a = p.parse_args()
    if a.stage == "inventory":
        inventory(a.source, a.output, a.config, a.analysis)
    else:
        study(a.audit, a.output, a.config)


if __name__ == "__main__":
    main()
