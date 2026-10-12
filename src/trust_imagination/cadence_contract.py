"""Pure checks and summaries for paired prefix contracts; no model inputs here."""

from __future__ import annotations

import numpy as np


def prefix_bank(raw, length):
    raw = np.asarray(raw)
    if (
        raw.ndim != 3
        or raw.shape[1:] != (50, 4)
        or raw.shape[0] < 2
        or not np.isfinite(raw).all()
    ):
        raise ValueError("Require finite B,50,4 proposals including a reference")
    if length not in (10, 50):
        raise ValueError("Declared prefix lengths are ten and fifty")
    return np.clip(raw[:, :length], -1, 1).astype(np.float32)


def trace_prefix_success(flags, length):
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 2 or flags.shape[1] != 3 or not len(flags):
        raise ValueError("Require nonempty termination/truncation/success flags")
    return bool(flags[:length, 2].any())


def forecast_valid_mask(stop_steps, transitions=10, frameskip=5):
    stop_steps = np.asarray(stop_steps, dtype=int)
    if (
        stop_steps.ndim != 1
        or np.any(stop_steps < 1)
        or np.any(stop_steps > transitions * frameskip)
    ):
        raise ValueError("Invalid first-stop or prefix-limit steps")
    return np.arange(1, transitions + 1)[None, :] * frameskip <= stop_steps[:, None]


def condition_metrics(repeat_success, early_success, bank_size):
    values = np.asarray(repeat_success, dtype=float)
    early = np.asarray(early_success, dtype=bool)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < bank_size
        or bank_size not in (4, 8)
    ):
        raise ValueError("Require declared repeat-by-candidate matrix and bank")
    if not np.isin(values, [0, 1]).all() or early.shape != (values.shape[1],):
        raise ValueError("Invalid binary success/early-prefix metadata")
    values = values[:, :bank_size]
    means = values.mean(0)
    differences = values - values[:, :1]
    best = int(np.argmax(means))
    return {
        "base": float(means[0]),
        "always_one": float(means[1]),
        "uniform_random": float(means.mean()),
        "best_measured_offline": float(means[best]),
        "best_index_offline": best,
        "helpful_opportunity": bool(np.any(means > means[0])),
        "harmful_opportunity": bool(np.any(means < means[0])),
        "mixed_candidate_means": bool(np.ptp(means) > 0),
        "alternative_help_frequency": float((differences[:, 1:] > 0).mean()),
        "alternative_harm_frequency": float((differences[:, 1:] < 0).mean()),
        "early_prefix_candidate_fraction": float(early[:bank_size].mean()),
        "all_candidates_post_prefix_eligible": bool(not early[:bank_size].any()),
    }
