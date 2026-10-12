"""Small frozen evaluator diagnostic using episode-disjoint data and NumPy.

Only actual final-prefix features train the latent evaluator. Its frozen weights
then score predicted features. Direct action scoring is a separate comparator.
This diagnostic never trains a comparative benefit gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def feature_maps(text_width):
    rng = np.random.default_rng(531)
    return {
        "visual_projection": rng.normal(0, 1 / np.sqrt(384), (384, 16)).astype(
            np.float32
        ),
        "text_projection": rng.normal(
            0, 1 / np.sqrt(text_width), (text_width, 32)
        ).astype(np.float32),
    }


def visual_features(latent, projection):
    array = np.asarray(latent, dtype=np.float32)
    if array.shape[-3:] != (16, 16, 384):
        raise ValueError(f"Unexpected latent grid: {array.shape}")
    # Preserve coarse spatial layout; reduce before projecting channels.
    pooled = array.reshape(*array.shape[:-3], 4, 4, 4, 4, 384).mean(axis=(-4, -2))
    return (pooled @ projection).reshape(*array.shape[:-3], 256)


def encode_features(visual, proprio_latent, text, remaining, maps):
    visual = visual_features(visual, maps["visual_projection"])
    proprio = np.asarray(proprio_latent, dtype=np.float32).mean(axis=-2)
    text_feature = (
        np.asarray(text, dtype=np.float32).reshape(-1) @ maps["text_projection"]
    )
    text_feature = np.broadcast_to(text_feature, (*visual.shape[:-1], 32))
    horizon = np.full((*visual.shape[:-1], 1), remaining / 400, dtype=np.float32)
    return np.concatenate((visual, proprio, text_feature, horizon), axis=-1)


def load_rows(root, protocol):
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != "passed":
        raise ValueError("A complete passed source collection is required")
    declared = [
        set(protocol[key])
        for key in ("train_seeds", "validation_seeds", "diagnostic_test_seeds")
    ]
    if any(declared[i] & declared[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Evaluator episode splits overlap")
    rows, actual_bank, predicted_bank, direct_bank = [], [], [], []
    maps = None
    for episode in summary["episodes"]:
        seed = episode["seed"]
        groups = [
            name
            for name, values in zip(("train", "validation", "test"), declared)
            if seed in values
        ]
        if len(groups) != 1:
            raise ValueError(f"Unassigned or ambiguous source seed {seed}")
        for state in episode["states"]:
            folder = root / state["episode_id"] / f"state{state['state_step']}"
            with np.load(
                folder / "features" / "deployment_inputs.npz", allow_pickle=False
            ) as deploy:
                with np.load(
                    folder / "outcomes" / "actual_future_features.npz",
                    allow_pickle=False,
                ) as label_features:
                    text = deploy["instruction_embedding"]
                    if maps is None:
                        maps = feature_maps(text.size)
                    # Only final (10-step) features receive continuation labels.
                    actual = encode_features(
                        label_features["latent"][:, -1, 0],
                        label_features["proprio_latent"][:, -1],
                        text,
                        state["remaining_horizon"] - state["execution_prefix"],
                        maps,
                    )
                    predicted = encode_features(
                        deploy["predicted_latents"][:, -1, 0],
                        deploy["predicted_proprio_latents"][:, -1],
                        text,
                        state["remaining_horizon"] - state["execution_prefix"],
                        maps,
                    )
                    current = encode_features(
                        deploy["current_latent"][0, 0, 0],
                        deploy["current_proprio_latent"][0, 0],
                        text,
                        state["remaining_horizon"],
                        maps,
                    )
                    direct = np.concatenate(
                        (
                            np.broadcast_to(current, actual.shape),
                            deploy["executed_prefixes"].reshape(len(actual), -1),
                        ),
                        axis=-1,
                    )
            for index, outcome in enumerate(state["branches"]):
                rows.append(
                    {
                        "episode_id": state["episode_id"],
                        "state_id": f"{state['episode_id']}/state{state['state_step']}",
                        "task": state["task"],
                        "seed": seed,
                        "split": groups[0],
                        "candidate_index": index,
                        "state_step": state["state_step"],
                        "success": int(outcome["success"]),
                        "eligible": bool(state["scorer_training_eligible"][index]),
                    }
                )
            actual_bank.extend(actual)
            predicted_bank.extend(predicted)
            direct_bank.extend(direct)
    if not rows:
        raise ValueError("No evaluator data")
    return (
        rows,
        np.asarray(actual_bank),
        np.asarray(predicted_bank),
        np.asarray(direct_bank),
        maps,
    )


def episode_weights(rows):
    counts = Counter(row["episode_id"] for row in rows)
    weights = np.asarray(
        [1 / counts[row["episode_id"]] for row in rows], dtype=np.float32
    )
    return weights / weights.sum()


def episode_brier(rows, target, predictions):
    return float(
        np.sum(episode_weights(rows) * (np.asarray(predictions) - target) ** 2)
    )


def fit_head(features, target, weights, regularization, updates=400):
    rng = np.random.default_rng(924)
    mean, scale = features.mean(0), features.std(0)
    scale = np.maximum(scale, 0.05)
    normalized = np.clip((features - mean) / scale, -10, 10)
    params = {
        "w1": rng.normal(
            0, 1 / np.sqrt(features.shape[1]), (features.shape[1], 64)
        ).astype(np.float32),
        "b1": np.zeros(64, np.float32),
        "w2": np.zeros(64, np.float32),
        "b2": np.zeros(1, np.float32),
    }
    moments = {key: np.zeros_like(value) for key, value in params.items()}
    variances = {key: np.zeros_like(value) for key, value in params.items()}
    for iteration in range(1, updates + 1):
        hidden = np.tanh(normalized @ params["w1"] + params["b1"])
        probability = sigmoid(hidden @ params["w2"] + params["b2"][0])
        error = (probability - target) * weights
        hidden_error = error[:, None] * params["w2"][None] * (1 - hidden**2)
        gradients = {
            "w2": hidden.T @ error + regularization * params["w2"],
            "b2": np.asarray([error.sum()], np.float32),
            "w1": normalized.T @ hidden_error + regularization * params["w1"],
            "b1": hidden_error.sum(0),
        }
        for key, gradient in gradients.items():
            moments[key] = 0.9 * moments[key] + 0.1 * gradient
            variances[key] = 0.999 * variances[key] + 0.001 * gradient**2
            params[key] -= (
                0.003
                * (moments[key] / (1 - 0.9**iteration))
                / (np.sqrt(variances[key] / (1 - 0.999**iteration)) + 1e-8)
            )
    return {**params, "mean": mean, "scale": scale}


def predict(head, features):
    normalized = np.clip((features - head["mean"]) / head["scale"], -10, 10)
    return sigmoid(
        np.tanh(normalized @ head["w1"] + head["b1"]) @ head["w2"] + head["b2"][0]
    )


class FrozenTaskEvaluator:
    """Deployment adapter consumes only candidate future latents and instruction."""

    def __init__(self, checkpoint):
        with np.load(checkpoint, allow_pickle=False) as archive:
            expected = {
                "w1",
                "b1",
                "w2",
                "b2",
                "mean",
                "scale",
                "visual_projection",
                "text_projection",
            }
            if set(archive.files) != expected:
                raise ValueError("Unexpected evaluator checkpoint fields")
            self.weights = {key: archive[key].copy() for key in expected}
        if any(not np.isfinite(value).all() for value in self.weights.values()):
            raise ValueError("Nonfinite evaluator weights")
        if self.weights["w1"].shape != (305, 64) or self.weights[
            "visual_projection"
        ].shape != (384, 16):
            raise ValueError("Unexpected evaluator feature architecture")
        if (self.weights["scale"] <= 0).any():
            raise ValueError("Invalid frozen feature scale")

    def score(
        self,
        visual_latents,
        proprio_latents,
        instruction_embedding,
        remaining_after_prefix,
    ):
        visual, proprio = np.asarray(visual_latents), np.asarray(proprio_latents)
        if visual.ndim != 4 or proprio.shape != (len(visual), 256, 16):
            raise ValueError(
                "Expected a candidate batch of native visual/proprioception latents"
            )
        features = encode_features(
            visual, proprio, instruction_embedding, remaining_after_prefix, self.weights
        )
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite deployment features")
        return predict(self.weights, features)


def select_regularization(features, target, rows, train, validation):
    candidates, models = [], []
    training_rows = [rows[i] for i in np.flatnonzero(train)]
    validation_rows = [rows[i] for i in np.flatnonzero(validation)]
    if len(set(target[train])) < 2 or not train.any() or not validation.any():
        raise ValueError("Need both training outcomes and a nonempty validation split")
    for strength in (0.0001, 0.001, 0.01):
        head = fit_head(
            features[train], target[train], episode_weights(training_rows), strength
        )
        score = episode_brier(
            validation_rows, target[validation], predict(head, features[validation])
        )
        candidates.append({"l2": strength, "validation_episode_mean_brier": score})
        models.append(head)
    best = int(
        np.argmin([item["validation_episode_mean_brier"] for item in candidates])
    )
    return models[best], {"grid": candidates, "selected_l2": candidates[best]["l2"]}


def selector_report(rows, predictions, bank_size):
    states = {}
    for index, row in enumerate(rows):
        if row["split"] == "test" and row["candidate_index"] < bank_size:
            states.setdefault(row["state_id"], []).append(index)
    records = []
    for state, indices in states.items():
        indices.sort(key=lambda i: rows[i]["candidate_index"])
        if len(indices) != bank_size or not all(rows[i]["eligible"] for i in indices):
            continue
        selected = indices[int(np.argmax(predictions[indices]))]
        base = rows[indices[0]]["success"]
        chosen = rows[selected]["success"]
        records.append(
            {
                "state_id": state,
                "episode_id": rows[selected]["episode_id"],
                "selected_index": rows[selected]["candidate_index"],
                "base_success": base,
                "selected_success": chosen,
                "oracle_success": max(rows[i]["success"] for i in indices),
                "selected_score": float(predictions[selected]),
                "base_score": float(predictions[indices[0]]),
                "label": (
                    "helps"
                    if chosen > base
                    else (
                        "hurts"
                        if chosen < base
                        else "tie_both_success" if base else "tie_both_failure"
                    )
                ),
            }
        )
    return {
        "states": len(records),
        "source_test_states": len(states),
        "omitted_early_prefix_or_incomplete_states": len(states) - len(records),
        "labels": dict(Counter(item["label"] for item in records)),
        "base_success_states": sum(item["base_success"] for item in records),
        "selected_success_states": sum(item["selected_success"] for item in records),
        "selected_episode_mean_brier": (
            episode_brier(
                records,
                np.asarray([item["selected_success"] for item in records]),
                np.asarray([item["selected_score"] for item in records]),
            )
            if records
            else None
        ),
        "base_episode_mean_brier": (
            episode_brier(
                records,
                np.asarray([item["base_success"] for item in records]),
                np.asarray([item["base_score"] for item in records]),
            )
            if records
            else None
        ),
        "oracle_success_states": sum(item["oracle_success"] for item in records),
        "decisions": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    rows, actual, predicted, direct, maps = load_rows(args.source, protocol)
    target = np.asarray([row["success"] for row in rows], dtype=np.float32)
    masks = {
        name: np.asarray([row["split"] == name and row["eligible"] for row in rows])
        for name in ("train", "validation", "test")
    }
    head, selection = select_regularization(
        actual, target, rows, masks["train"], masks["validation"]
    )
    direct_head, direct_selection = select_regularization(
        direct, target, rows, masks["train"], masks["validation"]
    )
    training_rows = [row for row, keep in zip(rows, masks["train"]) if keep]
    global_prior = float(
        np.sum(episode_weights(training_rows) * target[masks["train"]])
    )
    task_priors = {}
    for task in sorted({row["task"] for row in training_rows}):
        task_mask = masks["train"] & np.asarray([row["task"] == task for row in rows])
        task_priors[task] = float(
            np.sum(
                episode_weights([rows[i] for i in np.flatnonzero(task_mask)])
                * target[task_mask]
            )
        )
    predictions = {
        "actual_future_evaluator": predict(head, actual),
        "predicted_future_evaluator": predict(head, predicted),
        "direct_action_evaluator": predict(direct_head, direct),
        "global_prior": np.full(len(rows), global_prior),
        "task_prior": np.asarray([task_priors[row["task"]] for row in rows]),
    }
    report = {
        "source_run": args.source.name,
        "status": "passed",
        "protocol": protocol,
        "scope": "Six-episode held-out evaluator diagnostic, not gate training or closed-loop performance",
        "label_count": len(rows),
        "eligible_label_count": int(sum(masks.values()).sum()),
        "split_episodes": {
            split: sorted({row["episode_id"] for row in rows if row["split"] == split})
            for split in masks
        },
        "split_eligible_counts": {
            split: int(mask.sum()) for split, mask in masks.items()
        },
        "split_eligible_successes": {
            split: int(target[mask].sum()) for split, mask in masks.items()
        },
        "latent_head_selection": selection,
        "direct_head_selection": direct_selection,
        "baseline_task_priors": task_priors,
        "metrics": {},
        "selectors": {},
    }
    for split, mask in masks.items():
        selected_rows = [rows[i] for i in np.flatnonzero(mask)]
        report["metrics"][split] = {
            name: {
                "episode_mean_brier": episode_brier(
                    selected_rows, target[mask], probability[mask]
                ),
                "mean_probability": float(probability[mask].mean()),
            }
            for name, probability in predictions.items()
        }
    for name in (
        "actual_future_evaluator",
        "predicted_future_evaluator",
        "direct_action_evaluator",
    ):
        report["selectors"][name] = {
            str(size): selector_report(rows, predictions[name], size) for size in (4, 8)
        }
    # Freeze both heads and all preprocessing. No retraining on validation/test.
    np.savez_compressed(args.output / "frozen_latent_evaluator.npz", **head, **maps)
    np.savez_compressed(
        args.output / "frozen_direct_action_evaluator.npz", **direct_head, **maps
    )
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    report["runtime"] = {
        "python": sys.version,
        "numpy": np.__version__,
        "device": "local CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["provenance"] = {
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_summary_sha256": hashlib.sha256(
            (args.source / "summary.json").read_bytes()
        ).hexdigest(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "latent_evaluator_sha256": hashlib.sha256(
            (args.output / "frozen_latent_evaluator.npz").read_bytes()
        ).hexdigest(),
        "direct_evaluator_sha256": hashlib.sha256(
            (args.output / "frozen_direct_action_evaluator.npz").read_bytes()
        ).hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "split_eligible_counts": report["split_eligible_counts"],
                "test_metrics": report["metrics"]["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
