"""Fixed feature ablations and train-only normalized nonlinear ridge controls."""

from __future__ import annotations

import numpy as np
from .cadence_value_diagnostic import fit, predict

VARIANT_BLOCKS = {
    "current_only": ("current",),
    "direct": ("current", "actions"),
    "predicted_proprio": ("current", "predicted_proprio"),
    "predicted_visual": ("current", "predicted_visual"),
    "predicted_both": ("current", "predicted_visual", "predicted_proprio"),
    "direct_predicted_proprio": ("current", "actions", "predicted_proprio"),
    "direct_predicted_both": (
        "current",
        "actions",
        "predicted_visual",
        "predicted_proprio",
    ),
    "nonvisual_direct": ("nonvisual_current", "actions"),
    "nonvisual_direct_predicted": ("nonvisual_current", "actions", "predicted_proprio"),
}


def state_episode_id(episode, state):
    """Validate identity across summary schemas before associating inputs/labels."""
    expected = f"{episode['task']}_seed{episode['seed']}"
    if (
        state["episode_id"] != expected
        or state["task"] != episode["task"]
        or state["seed"] != episode["seed"]
    ):
        raise ValueError("State and episode task/seed identity differ")
    return expected


def inputs(blocks, variant):
    """Select an explicit allowlist; labels/support/physical futures cannot enter."""
    names = VARIANT_BLOCKS[variant]
    x = np.concatenate([blocks[name] for name in names], axis=1).astype(float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty deployment features required")
    return x


def fit_model(x, y, w, cfg, map_seed=None):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError("Training shapes differ")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite training data")
    if not len(x) or np.any(w <= 0) or not np.isclose(w.sum(), 1):
        raise ValueError("Positive normalized training weights required")
    if map_seed is None:
        return fit(x, y, w, cfg)
    mean = w @ x
    scale = np.maximum(np.sqrt(w @ ((x - mean) ** 2)), cfg["normalization_floor"])
    z = np.clip(
        (x - mean) / scale, -cfg["normalization_clip"], cfg["normalization_clip"]
    )
    rng = np.random.default_rng(map_seed)
    matrix = rng.normal(
        0, 1 / np.sqrt(x.shape[1]), (x.shape[1], cfg["random_map_width"])
    )
    phase = rng.uniform(0, 2 * np.pi, cfg["random_map_width"])
    phi = np.sqrt(2) * np.cos(z @ matrix + phase)
    center = w @ phi
    intercept = float(w @ y)
    weighted = (phi - center) * np.sqrt(w[:, None])
    coef = weighted.T @ np.linalg.solve(
        weighted @ weighted.T + cfg["l2"] * np.eye(len(x)), (y - intercept) * np.sqrt(w)
    )
    return {
        "mean": mean,
        "scale": scale,
        "matrix": matrix,
        "phase": phase,
        "center": center,
        "coef": coef,
        "intercept": np.asarray(intercept),
    }


def predict_model(head, x, cfg):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or x.ndim != 2 or x.shape[1] != len(head["mean"]):
        raise ValueError("Invalid deployment features")
    if "matrix" not in head:
        return predict(head, x, cfg)
    z = np.clip(
        (x - head["mean"]) / head["scale"],
        -cfg["normalization_clip"],
        cfg["normalization_clip"],
    )
    phi = np.sqrt(2) * np.cos(z @ head["matrix"] + head["phase"])
    return np.clip((phi - head["center"]) @ head["coef"] + head["intercept"], 0, 1)


def model_specs(cfg):
    for rep in cfg["representations"]:
        for variant in cfg["linear_variants"]:
            yield {
                "name": f"{rep}__linear__{variant}",
                "representation": rep,
                "variant": variant,
                "map_seed": None,
                "family": "linear",
            }
        for variant in cfg["nonlinear_variants"]:
            for seed in cfg["random_map_seeds"]:
                yield {
                    "name": f"{rep}__cosine{seed}__{variant}",
                    "representation": rep,
                    "variant": variant,
                    "map_seed": seed,
                    "family": "nonlinear",
                }


def strata(stop_steps, early_success):
    stop_steps = np.asarray(stop_steps)
    early_success = np.asarray(early_success, bool)
    if stop_steps.shape != early_success.shape or not len(stop_steps):
        raise ValueError("Physical union metadata differs")
    if early_success.any():
        return "union_early_success"
    if np.all(stop_steps >= 50):
        return "union_complete_no_early"
    return "union_incomplete_no_early"
