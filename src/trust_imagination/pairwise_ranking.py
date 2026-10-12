"""Within-state ranking with whole-episode weights and train-only transforms."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .readout_study import linear_predict
from .task_evaluator import encode_features


def grouped_states(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["state_id"], []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: rows[i]["candidate_index"])
        if len({rows[i]["candidate_index"] for i in indices}) != len(indices):
            raise ValueError("Duplicate candidate within state")
        if len({rows[i]["episode_id"] for i in indices}) != 1:
            raise ValueError("State crosses episodes")
    return groups


def pairs(rows):
    """Retain ties; weighting is episode -> state -> eligible unordered pair."""
    records = []
    for state, indices in grouped_states(rows).items():
        eligible = [i for i in indices if rows[i]["eligible"]]
        for offset, i in enumerate(eligible):
            for j in eligible[offset + 1 :]:
                yi, yj = np.asarray(rows[i]["outcomes"], dtype=np.float64), np.asarray(
                    rows[j]["outcomes"], dtype=np.float64
                )
                if yi.shape != yj.shape or yi.ndim != 1 or not len(yi):
                    raise ValueError("Matched outcomes required within every pair")
                if not np.isin(yi, [0, 1]).all() or not np.isin(yj, [0, 1]).all():
                    raise ValueError("Binary success outcomes required")
                records.append(
                    (i, j, float((yi - yj).mean()), state, rows[i]["episode_id"])
                )
    if not records:
        raise ValueError("No eligible within-state pairs")
    state_counts = Counter(r[3] for r in records)
    episode_states = Counter(
        {
            s: rows[ix[0]]["episode_id"]
            for s, ix in grouped_states(rows).items()
            if s in state_counts
        }.values()
    )
    weight = np.asarray(
        [
            1 / (len(episode_states) * episode_states[r[4]] * state_counts[r[3]])
            for r in records
        ],
        dtype=np.float64,
    )
    return {
        "i": np.asarray([r[0] for r in records]),
        "j": np.asarray([r[1] for r in records]),
        "target": np.asarray([r[2] for r in records]),
        "weights": weight,
        "states": [r[3] for r in records],
        "episodes": [r[4] for r in records],
    }


def train_transform(features, representation, width=128, seed=782):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite training feature matrix required")
    head = {"mean": x.mean(0), "scale": np.maximum(x.std(0), 0.05)}
    if representation == "linear_plus_fixed_tanh":
        rng = np.random.default_rng(seed)
        head["tanh_projection"] = rng.normal(
            0, 1 / np.sqrt(x.shape[1]), (x.shape[1], width)
        )
        head["tanh_bias"] = rng.uniform(-1, 1, width)
    elif representation != "linear":
        raise ValueError("Unknown representation")
    return head


def transform(head, features):
    z = np.clip(
        (np.asarray(features, dtype=np.float64) - head["mean"]) / head["scale"], -10, 10
    )
    if "tanh_projection" in head:
        z = np.concatenate(
            (z, np.tanh(z @ head["tanh_projection"] + head["tanh_bias"])), axis=-1
        )
    return z


def fit_ranker(
    features, pair_data, objective, representation, l2, updates=800, lr=0.01
):
    if l2 <= 0:
        raise ValueError("Positive regularization required")
    head = train_transform(features, representation)
    z = transform(head, features)
    d = z[pair_data["i"]] - z[pair_data["j"]]
    weight, target = pair_data["weights"], pair_data["target"]
    if objective == "relative_ridge":
        gram = d.T @ (weight[:, None] * d) + l2 * np.eye(d.shape[1])
        coef = np.linalg.solve(gram, d.T @ (weight * target))
    elif objective == "pairwise_logistic":
        coef = np.zeros(d.shape[1])
        m = coef.copy()
        v = coef.copy()
        soft = 0.5 + 0.5 * target
        for step in range(1, updates + 1):
            probability = 1 / (1 + np.exp(-np.clip(d @ coef, -30, 30)))
            gradient = d.T @ (weight * (probability - soft)) + l2 * coef
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient**2
            coef -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    else:
        raise ValueError("Unknown ranking objective")
    head["coef"] = coef
    return head


def utility(kind, head, features):
    if kind == "null":
        return np.zeros(len(features))
    if kind == "absolute":
        return linear_predict(head, np.asarray(features, dtype=np.float32))
    return transform(head, features) @ head["coef"]


def pair_delta(kind, scores, pair_data):
    delta = scores[pair_data["i"]] - scores[pair_data["j"]]
    if kind == "pairwise_logistic":
        return 2 / (1 + np.exp(-np.clip(delta, -30, 30))) - 1
    return np.clip(delta, -1, 1)


def pair_metrics(rows, scores, kind):
    data = pairs(rows)
    predicted = pair_delta(kind, scores, data)
    weight, target = data["weights"], data["target"]
    informative = np.abs(target) > 1e-12
    ordering = np.where(
        np.abs(predicted) <= 1e-12,
        0.5,
        (np.sign(predicted) == np.sign(target)).astype(float),
    )
    repeat_errors = [
        np.mean(
            (
                p
                - (
                    np.asarray(rows[i]["outcomes"], dtype=np.float64)
                    - np.asarray(rows[j]["outcomes"], dtype=np.float64)
                )
            )
            ** 2
        )
        for p, i, j in zip(predicted, data["i"], data["j"])
    ]
    return {
        "pairs": len(target),
        "episodes": len(set(data["episodes"])),
        "states": len(set(data["states"])),
        "informative_pairs": int(informative.sum()),
        "informative_states": len(
            {s for s, keep in zip(data["states"], informative) if keep}
        ),
        "informative_episodes": sorted(
            {e for e, keep in zip(data["episodes"], informative) if keep}
        ),
        "episode_weighted_mean_effect_mse": float(
            np.dot(weight, (predicted - target) ** 2)
        ),
        "episode_weighted_repeat_effect_mse": float(np.dot(weight, repeat_errors)),
        "informative_pair_order_accuracy": (
            float(
                np.dot(weight[informative], ordering[informative])
                / weight[informative].sum()
            )
            if informative.any()
            else None
        ),
    }


def episode_average(records, key):
    groups = {}
    for row in records:
        groups.setdefault(row["episode_id"], []).append(row[key])
    return (
        float(np.mean([np.mean(values) for values in groups.values()]))
        if groups
        else None
    )


def selection_metrics(rows, scores, bank, eligible_only=False):
    decisions = []
    omitted = 0
    for state, indices in grouped_states(rows).items():
        indices = [i for i in indices if rows[i]["candidate_index"] < bank]
        if len(indices) != bank or [
            rows[i]["candidate_index"] for i in indices
        ] != list(range(bank)):
            raise ValueError("Nested bank must preserve candidate indices")
        if eligible_only and not all(rows[i]["eligible"] for i in indices):
            omitted += 1
            continue
        y = np.asarray([rows[i]["outcomes"] for i in indices], dtype=np.float64).T
        base = y[:, 0]
        frequency = y.mean(0)
        if isinstance(scores, str):
            choice = {
                "base": 0,
                "one": 1,
                "random": None,
                "oracle": int(np.argmax(frequency)),
            }[scores]
        else:
            choice = int(np.argmax(np.asarray(scores)[indices]))
        selected = float(y.mean()) if choice is None else float(frequency[choice])
        help_rate = (
            float((y > base[:, None]).mean())
            if choice is None
            else float((y[:, choice] > base).mean())
        )
        harm_rate = (
            float((y < base[:, None]).mean())
            if choice is None
            else float((y[:, choice] < base).mean())
        )
        decisions.append(
            {
                "state_id": state,
                "episode_id": rows[indices[0]]["episode_id"],
                "selected_index": choice,
                "success": selected,
                "base_success": float(base.mean()),
                "gain": selected - float(base.mean()),
                "help": help_rate,
                "harm": harm_rate,
                "regret": float(frequency.max() - selected),
                "mixed_repeat_bank": bool(np.any(y.min(1) != y.max(1))),
                "differing_mean_outcomes": bool(frequency.max() != frequency.min()),
            }
        )
    return {
        "states": len(decisions),
        "omitted_states": omitted,
        "episode_mean_success": episode_average(decisions, "success"),
        "episode_mean_gain": episode_average(decisions, "gain"),
        "episode_mean_help": episode_average(decisions, "help"),
        "episode_mean_harm": episode_average(decisions, "harm"),
        "episode_mean_regret": episode_average(decisions, "regret"),
        "decisions": decisions,
    }


class FrozenRanker:
    """Allowed native inputs only. Output is utility, never success probability."""

    def __init__(self, descriptor):
        descriptor = Path(descriptor)
        self.spec = json.loads(descriptor.read_text())
        checkpoint = descriptor.parent / self.spec["checkpoint"]
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != self.spec["checkpoint_sha256"]
        ):
            raise ValueError("Frozen ranker digest mismatch")
        with np.load(checkpoint, allow_pickle=False) as data:
            self.head = {key: data[key].copy() for key in data.files}
        if any(not np.isfinite(v).all() for v in self.head.values()):
            raise ValueError("Nonfinite ranker checkpoint")

    def score_future(self, visual, proprio, text, remaining_after_prefix):
        if self.spec["input_family"] != "actual_future":
            raise ValueError("Future adapter requires future ranker")
        if np.asarray(visual).ndim != 4 or np.asarray(proprio).shape != (
            len(visual),
            256,
            16,
        ):
            raise ValueError("Native candidate visual/proprioceptive features required")
        x = encode_features(visual, proprio, text, remaining_after_prefix, self.head)
        return utility(self.spec["kind"], self.head, x)

    def score_direct(self, current_visual, current_proprio, text, remaining, prefixes):
        if self.spec["input_family"] != "direct_action":
            raise ValueError("Direct adapter requires direct ranker")
        prefixes = np.asarray(prefixes, dtype=np.float32)
        if prefixes.ndim != 3 or prefixes.shape[1:] != (10, 4):
            raise ValueError("Ten-action four-dimensional prefixes required")
        current = encode_features(
            current_visual, current_proprio, text, remaining, self.head
        )
        x = np.concatenate(
            (
                np.broadcast_to(current, (len(prefixes), 305)),
                prefixes.reshape(len(prefixes), 40),
            ),
            -1,
        )
        return utility(self.spec["kind"], self.head, x)
