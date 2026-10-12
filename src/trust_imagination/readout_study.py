"""Development-only evaluator comparison. Never opens old test feature files."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from .task_evaluator import (
    feature_maps,
    encode_features,
    episode_weights,
    episode_brier,
    predict,
    sigmoid,
)


def variant(features, name):
    if name == "spatial":
        return features
    if name == "global":
        return np.concatenate(
            (
                features[..., :256].reshape(*features.shape[:-1], 16, 16).mean(-2),
                features[..., 256:],
            ),
            -1,
        )
    if name == "spatial_no_proprio":
        return np.concatenate((features[..., :256], features[..., 272:]), -1)
    raise ValueError(f"Unknown feature variant {name}")


def fit_linear(x, y, weights, l2, updates=800, lr=0.01):
    if not np.isfinite(x).all() or not len(x):
        raise ValueError("Invalid training features")
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.05)
    z = np.clip((x - mean) / scale, -10, 10)
    coef = np.zeros(x.shape[1], np.float32)
    intercept = np.zeros(1, np.float32)
    m, v = np.zeros_like(coef), np.zeros_like(coef)
    mb, vb = np.zeros(1, np.float32), np.zeros(1, np.float32)
    for step in range(1, updates + 1):
        residual = (sigmoid(z @ coef + intercept[0]) - y) * weights
        grad, bias = z.T @ residual + l2 * coef, np.asarray(
            [residual.sum()], np.float32
        )
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        mb = 0.9 * mb + 0.1 * bias
        vb = 0.999 * vb + 0.001 * bias**2
        coef -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        intercept -= (
            lr * (mb / (1 - 0.9**step)) / (np.sqrt(vb / (1 - 0.999**step)) + 1e-8)
        )
    return {"coef": coef, "intercept": intercept, "mean": mean, "scale": scale}


def linear_predict(head, x):
    return sigmoid(
        np.clip((x - head["mean"]) / head["scale"], -10, 10) @ head["coef"]
        + head["intercept"][0]
    )


class FrozenReadout:
    def __init__(self, directory):
        root = Path(directory)
        self.spec = json.loads((root / "selected_readout.json").read_text())
        checkpoint = root / self.spec["checkpoint"]
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != self.spec["checkpoint_sha256"]
        ):
            raise ValueError("Frozen readout digest mismatch")
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.head = {k: archive[k].copy() for k in archive.files}
        if any(not np.isfinite(value).all() for value in self.head.values()):
            raise ValueError("Nonfinite readout parameters")

    def score(
        self, visual, proprio_latent, instruction_embedding, remaining_after_prefix
    ):
        if np.asarray(visual).ndim != 4 or np.asarray(proprio_latent).shape != (
            len(visual),
            256,
            16,
        ):
            raise ValueError("Expected native candidate visual/proprioception features")
        x = variant(
            encode_features(
                visual,
                proprio_latent,
                instruction_embedding,
                remaining_after_prefix,
                self.head,
            ),
            self.spec["variant"],
        )
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite readout inputs")
        return (
            linear_predict(self.head, x)
            if self.spec["kind"] == "linear"
            else predict(self.head, x)
        )


def load_development(root, config):
    train, val = set(config["train_seeds"]), set(config["validation_seeds"])
    if train & val or (train | val) & set(config["excluded_source_seeds"]):
        raise ValueError("Development split overlaps excluded episodes")
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "passed":
        raise ValueError("Completed source required")
    rows, banks, maps = (
        [],
        {k: [] for k in ("actual", "predicted", "current", "actions")},
        None,
    )
    for episode in summary["episodes"]:
        # Filter before reading per-episode states, arrays or outcome labels.
        seed = episode["seed"]
        if seed not in train | val:
            continue
        split = "train" if seed in train else "validation"
        for state in episode["states"]:
            state_id = f"{state['episode_id']}/state{state['state_step']}"
            folder = root / state_id
            with (
                np.load(
                    folder / "features/deployment_inputs.npz", allow_pickle=False
                ) as d,
                np.load(
                    folder / "outcomes/actual_future_features.npz", allow_pickle=False
                ) as a,
            ):
                text = d["instruction_embedding"]
                if maps is None:
                    maps = feature_maps(text.size)
                remaining = state["remaining_horizon"] - state["execution_prefix"]
                actual = encode_features(
                    a["latent"][:, -1, 0],
                    a["proprio_latent"][:, -1],
                    text,
                    remaining,
                    maps,
                )
                predicted = encode_features(
                    d["predicted_latents"][:, -1, 0],
                    d["predicted_proprio_latents"][:, -1],
                    text,
                    remaining,
                    maps,
                )
                current = encode_features(
                    d["current_latent"][0, 0, 0],
                    d["current_proprio_latent"][0, 0],
                    text,
                    state["remaining_horizon"],
                    maps,
                )
                banks["actual"].extend(actual)
                banks["predicted"].extend(predicted)
                banks["current"].extend(np.broadcast_to(current, actual.shape))
                banks["actions"].extend(d["executed_prefixes"].reshape(len(actual), 40))
            for c, b in enumerate(state["branches"]):
                rows.append(
                    {
                        "episode_id": state["episode_id"],
                        "state_id": state_id,
                        "task": state["task"],
                        "seed": seed,
                        "split": split,
                        "candidate_index": c,
                        "success": int(b["success"]),
                        "eligible": bool(state["scorer_training_eligible"][c]),
                    }
                )
        print(f"DEVELOPMENT_LOADED {episode['task']} seed={seed}", flush=True)
    if not rows:
        raise ValueError("Empty development source")
    return rows, {k: np.asarray(v, dtype=np.float32) for k, v in banks.items()}, maps


def selections(rows, scores, bank, split):
    grouped = {}
    for i, r in enumerate(rows):
        if r["split"] == split and r["candidate_index"] < bank:
            grouped.setdefault(r["state_id"], []).append(i)
    decisions = []
    for state, indices in grouped.items():
        indices.sort(key=lambda i: rows[i]["candidate_index"])
        if len(indices) != bank or not all(rows[i]["eligible"] for i in indices):
            continue
        y = np.asarray([rows[i]["success"] for i in indices])
        base = int(y[0])
        if isinstance(scores, str) and scores == "random":
            value = float(y.mean())
            helps = float(((y - base) > 0).mean())
            hurts = float(((y - base) < 0).mean())
            chosen = None
        else:
            chosen = (
                0
                if isinstance(scores, str) and scores == "base"
                else (
                    1
                    if isinstance(scores, str) and scores == "one"
                    else int(np.argmax(scores[indices]))
                )
            )
            value = float(y[chosen])
            helps = float(value > base)
            hurts = float(value < base)
        decisions.append(
            {
                "state_id": state,
                "episode_id": rows[indices[0]]["episode_id"],
                "selected_index": chosen,
                "selected_success": value,
                "base_success": base,
                "helps": helps,
                "hurts": hurts,
                "mixed_bank": bool(y.min() != y.max()),
                "oracle_success": int(y.max()),
            }
        )
    if not decisions:
        raise ValueError("No eligible selection states")
    weights = episode_weights(decisions)
    mixed = [d for d in decisions if d["mixed_bank"]]
    return {
        "states": len(decisions),
        "omitted_states": len(grouped) - len(decisions),
        "selected_success_states": sum(d["selected_success"] for d in decisions),
        "helps": sum(d["helps"] for d in decisions),
        "hurts": sum(d["hurts"] for d in decisions),
        "episode_mean_success": float(
            np.dot(weights, [d["selected_success"] for d in decisions])
        ),
        "episode_mean_gain": float(
            np.dot(
                weights, [d["selected_success"] - d["base_success"] for d in decisions]
            )
        ),
        "mixed_states": len(mixed),
        "mixed_selected_success_states": sum(d["selected_success"] for d in mixed),
        "decisions": decisions,
    }


def run(source, old_evaluator, config_path, output):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows, banks, maps = load_development(source, config)
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(output / "development_feature_cache.npz", **banks, **maps)
    y = np.asarray([r["success"] for r in rows], np.float32)
    masks = {
        name: np.asarray([r["split"] == name and r["eligible"] for r in rows])
        for name in ("train", "validation")
    }
    train, val = masks["train"], masks["validation"]
    training = [r for r, m in zip(rows, train) if m]
    validation = [r for r, m in zip(rows, val) if m]
    if len(set(y[train])) < 2:
        raise ValueError("Both training outcomes needed")
    weights = episode_weights(training)
    trials = []
    models = {}
    best = {}
    designs = [
        ("actual_" + v, variant(banks["actual"], v), v)
        for v in config["latent_variants"]
    ]
    designs += [
        (
            "direct_spatial",
            np.concatenate((banks["current"], banks["actions"]), -1),
            "spatial",
        ),
        ("current_spatial", banks["current"], "spatial"),
    ]
    for name, x, v in designs:
        for l2 in config["linear_l2_grid"]:
            head = fit_linear(
                x[train],
                y[train],
                weights,
                l2,
                config["updates"],
                config["learning_rate"],
            )
            score = linear_predict(head, x)
            trial = {
                "name": name,
                "variant": v,
                "kind": "linear",
                "l2": l2,
                "dimensions": x.shape[1],
                "train_brier": episode_brier(training, y[train], score[train]),
                "validation_brier": episode_brier(validation, y[val], score[val]),
            }
            trials.append(trial)
            models[(name, l2)] = head
        best[name] = min(
            [t for t in trials if t["name"] == name],
            key=lambda t: t["validation_brier"],
        )
        print(
            f"READOUT_FIT {name} val_brier={best[name]['validation_brier']:.6f}",
            flush=True,
        )
    with np.load(
        old_evaluator / "frozen_latent_evaluator.npz", allow_pickle=False
    ) as old:
        old_head = {k: old[k].copy() for k in old.files}
    old_scores = predict(old_head, banks["actual"])
    old_report = json.loads((old_evaluator / "summary.json").read_text())
    old_val = episode_brier(validation, y[val], old_scores[val])
    if (
        abs(
            old_val
            - old_report["metrics"]["validation"]["actual_future_evaluator"][
                "episode_mean_brier"
            ]
        )
        > 1e-6
    ):
        raise AssertionError("Original feature/score reproduction failed")
    original = {
        "name": "original_mlp",
        "variant": "spatial",
        "kind": "mlp",
        "dimensions": 305,
        "l2": old_report["latent_head_selection"]["selected_l2"],
        "train_brier": episode_brier(training, y[train], old_scores[train]),
        "validation_brier": old_val,
    }
    candidates = [best["actual_" + v] for v in config["latent_variants"]] + [original]
    winner = min(candidates, key=lambda t: t["validation_brier"])
    head = (
        old_head if winner["kind"] == "mlp" else models[(winner["name"], winner["l2"])]
    )
    np.savez_compressed(
        output / "selected_checkpoint.npz",
        **head,
        **({} if winner["kind"] == "mlp" else maps),
    )
    checkpoint = output / "selected_checkpoint.npz"
    spec = {
        **winner,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "feature_seed": 531,
        "target": "Remaining ever-success after ten-action prefix followed by frozen VLA continuation",
        "training_episodes": sorted({r["episode_id"] for r in training}),
        "validation_episodes": sorted({r["episode_id"] for r in validation}),
        "excluded_old_test_seeds": config["excluded_source_seeds"],
        "fresh_evaluation_required": True,
    }
    (output / "selected_readout.json").write_text(json.dumps(spec, indent=2) + "\n")
    score_fn = linear_predict if winner["kind"] == "linear" else predict
    scores = {
        "actual_future": score_fn(head, variant(banks["actual"], winner["variant"])),
        "predicted_future": score_fn(
            head, variant(banks["predicted"], winner["variant"])
        ),
    }
    for mode, name in [
        ("direct_action", "direct_spatial"),
        ("current_only", "current_spatial"),
    ]:
        b = best[name]
        x = (
            np.concatenate((banks["current"], banks["actions"]), -1)
            if mode == "direct_action"
            else banks["current"]
        )
        h = models[(name, b["l2"])]
        scores[mode] = linear_predict(h, x)
        np.savez_compressed(output / f"{mode}_checkpoint.npz", **h, **maps)
    metrics = {
        split: {
            name: episode_brier([r for r, m in zip(rows, mask) if m], y[mask], p[mask])
            for name, p in scores.items()
        }
        for split, mask in masks.items()
    }
    controls = {
        "base": "base",
        "always_candidate_one": "one",
        "uniform_random_including_base": "random",
        **scores,
    }
    selection = {
        split: {
            str(bank): {
                name: selections(rows, p, bank, split) for name, p in controls.items()
            }
            for bank in (4, 8)
        }
        for split in masks
    }
    curves = []
    for group in config["learning_curve_train_seed_groups"]:
        mask = train & np.asarray([r["seed"] in group for r in rows])
        subset = [r for r, m in zip(rows, mask) if m]
        if winner["kind"] == "linear":
            x = variant(banks["actual"], winner["variant"])
            h = fit_linear(
                x[mask],
                y[mask],
                episode_weights(subset),
                winner["l2"],
                config["updates"],
                config["learning_rate"],
            )
            p = linear_predict(h, x)
        else:
            from .task_evaluator import fit_head

            h = fit_head(
                banks["actual"][mask], y[mask], episode_weights(subset), winner["l2"]
            )
            p = predict(h, banks["actual"])
        curves.append(
            {
                "episode_count": len({r["episode_id"] for r in subset}),
                "seeds": group,
                "train_brier": episode_brier(subset, y[mask], p[mask]),
                "validation_brier": episode_brier(validation, y[val], p[val]),
            }
        )
    x_actual = variant(banks["actual"], winner["variant"])
    x_pred = variant(banks["predicted"], winner["variant"])
    drift = {}
    for task in sorted({r["task"] for r in validation}):
        mask = val & np.asarray([r["task"] == task for r in rows])
        trows = [r for r, m in zip(rows, mask) if m]
        drift[task] = {
            "actual_brier": episode_brier(
                trows, y[mask], scores["actual_future"][mask]
            ),
            "predicted_brier": episode_brier(
                trows, y[mask], scores["predicted_future"][mask]
            ),
            "mean_absolute_score_shift": float(
                np.mean(
                    np.abs(
                        scores["predicted_future"][mask] - scores["actual_future"][mask]
                    )
                )
            ),
            "unclipped_training_standardized_feature_rms_shift": float(
                np.sqrt(np.mean(((x_pred[mask] - x_actual[mask]) / head["scale"]) ** 2))
            ),
        }
    np.savez_compressed(output / "development_predictions.npz", **scores)
    result = {
        "status": "passed",
        "scope": "Development/validation only; no old test feature files opened or new test result claimed",
        "config": config,
        "trials": trials,
        "original_mlp": original,
        "selected": spec,
        "metrics": metrics,
        "learning_curves": curves,
        "validation_feature_shift": drift,
        "selectors": selection,
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": {
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": winner,
                "metrics": metrics,
                "learning_curves": curves,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--old-evaluator", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.old_evaluator, a.config, a.output)
