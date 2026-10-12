"""Input-only proposal measurements; outcome timing is a separate report utility."""

from collections import defaultdict
import numpy as np


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def separation(value):
    x = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite logical candidates required")
    pairs = [rms(x[i] - x[j]) for i in range(len(x)) for j in range(i)]
    return {
        "pair_rms": float(np.mean(pairs)),
        "reference_rms": float(np.mean([rms(v - x[0]) for v in x[1:]])),
    }


def participation(value):
    x = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    e = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    return float(e.sum() ** 2 / (e @ e)) if e.sum() > 0 else 0.0


def action_measurements(actions):
    x = np.asarray(actions, dtype=np.float64)
    if (
        x.ndim != 3
        or x.shape[1:] != (50, 4)
        or not np.isfinite(x).all()
        or np.abs(x).max() > 1
    ):
        raise ValueError("Bounded finite fifty-step native actions required")
    return {
        "translation": separation(x[:, :, :3]),
        "gripper": separation(x[:, :, 3]),
        "exact_bound_by_channel": (np.abs(x) >= 1).mean((0, 1)).tolist(),
        "near_bound_by_channel": (np.abs(x) >= 0.95).mean((0, 1)).tolist(),
        "translation_temporal_delta_rms": rms(np.diff(x[:, :, :3], axis=1)),
        "gripper_temporal_delta_rms": rms(np.diff(x[:, :, 3], axis=1)),
        "mean_signed_command_sum": x.sum(1).mean(0).tolist(),
        "translation_participation": participation(x[:, :, :3]),
        "byte_unique_logical_actions": len({v.astype(np.float32).tobytes() for v in x}),
    }


def latent_measurements(predicted, current=None):
    x = np.asarray(predicted, dtype=np.float64)
    result = separation(x)
    denom = np.linalg.norm(x[1:], axis=1) * np.linalg.norm(x[0])
    valid = denom > 0
    result["reference_cosine_distance"] = (
        float(np.mean(1 - (x[1:][valid] @ x[0]) / denom[valid]))
        if valid.any()
        else None
    )
    result["cosine_pairs_supported"] = int(valid.sum())
    if current is not None:
        change = rms(x[0] - np.asarray(current, dtype=np.float64))
        result["reference_forecast_change_rms"] = change
        result["pair_rms_over_reference_change"] = (
            result["pair_rms"] / change if change > 0 else None
        )
    return result


def report_trace_timing(flags, source_step, horizon=400):
    """Report only: no actions/features are accepted or produced here."""
    x = np.asarray(flags)
    remaining = horizon - source_step
    if (
        x.dtype != np.bool_
        or x.ndim != 2
        or x.shape[1] != 3
        or not 0 < len(x) <= remaining
    ):
        raise ValueError("Bounded boolean terminated/truncated/success trace required")
    if x[:-1].any() or (len(x) < remaining and not x[-1].any()):
        raise ValueError("Trace must end at first stop or remaining horizon")
    hits = np.flatnonzero(x[:, 2])
    first = int(hits[0] + 1) if len(hits) else None
    reason = (
        "success"
        if first
        else "terminated" if x[-1, 0] else "truncated" if x[-1, 1] else "horizon"
    )
    return {
        "success": first is not None,
        "first_success_relative": first,
        "first_success_absolute": source_step + first if first is not None else None,
        "observed_steps": len(x),
        "stop_reason": reason,
        "failure_censored": first is None,
        "success_within_prefix50": first is not None and first <= 50,
    }


def outcome_class(means):
    p = np.asarray(means, dtype=np.float64)
    helpful, harmful = bool((p[1:] > p[0]).any()), bool((p[1:] < p[0]).any())
    if helpful and harmful:
        return "helpful_and_harmful"
    if helpful:
        return "helpful_only"
    if harmful:
        return "harmful_only"
    return (
        "tied_failure" if p[0] == 0 else "tied_success" if p[0] == 1 else "tied_partial"
    )


def episode_mean(records, getter):
    groups = defaultdict(list)
    for r in records:
        v = getter(r)
        if v is not None:
            groups[r["episode_id"]].append(float(v))
    return {
        "mean": (
            float(np.mean([np.mean(v) for v in groups.values()])) if groups else None
        ),
        "episodes_with_support": len(groups),
        "states_with_support": sum(map(len, groups.values())),
    }
