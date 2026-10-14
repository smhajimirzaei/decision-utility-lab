"""Deployment-only comparative features and fixed capacity benefit readouts."""

import numpy as np
from .cadence_value_diagnostic import pool
from .baseline_audit import fit_model

REFERENCE = 3
SPECS = [
    (kind, metadata)
    for kind in ("direct", "endpoint", "trajectory")
    for metadata in (False, True)
]
CONFIG = {
    "normalization_floor": 0.05,
    "normalization_clip": 10.0,
    "random_map_width": 128,
    "l2": 0.1,
}


def comparative_features(
    deployment, projection, tasks, task, step, indices, size, radius, kind, metadata
):
    """Explicit allowlist: no reads of outcome/support/physical-future keys."""
    ix = np.asarray(indices[:size], int)
    actions = deployment["executed_unique_pool"][ix].reshape(size, 200)
    reference = np.broadcast_to(actions[REFERENCE], actions.shape)
    current = np.r_[
        pool(deployment["current_latent"][0, 0, 0], projection, "global"),
        deployment["current_proprio_latent"][0, 0].mean(0),
        deployment["proprio"],
        np.eye(len(tasks))[tasks.index(task)],
        (400 - step - 50) / 400,
    ]
    blocks = [
        np.broadcast_to(current, (size, len(current))),
        actions,
        reference,
        actions - reference,
    ]
    if kind in ("endpoint", "trajectory"):
        visual = pool(deployment["predicted_latents"][ix, :, 0], projection, "global")
        prop = deployment["predicted_proprio_latents"][ix].mean(2)
        trajectory = np.concatenate([visual, prop], axis=-1)
        future = (
            trajectory[:, -1] if kind == "endpoint" else trajectory.reshape(size, -1)
        )
        ref_future = np.broadcast_to(future[REFERENCE], future.shape)
        blocks += [future, ref_future, future - ref_future]
    elif kind != "direct":
        raise ValueError("Unknown feature family")
    if metadata:
        rms = np.sqrt(np.mean((actions - reference) ** 2, axis=1))
        effective_radius = radius if size > 4 else 0.0
        blocks.append(
            np.c_[
                np.full(size, effective_radius),
                np.full(size, np.log(size)),
                rms,
                np.full(size, rms.mean()),
                np.mean(np.abs(actions) == 1, axis=1),
            ]
        )
    result = np.concatenate(blocks, axis=1).astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite deployment feature")
    return result


def labels(outcomes):
    y = np.asarray(outcomes, float)
    delta = y - y[REFERENCE]
    return np.c_[np.mean(delta > 0, axis=1), np.mean(delta < 0, axis=1)]


def fit_benefit(x, target, weights):
    return [fit_model(x, target[:, i], weights, CONFIG, 701) for i in range(2)]


def predict_benefit(heads, x):
    from .baseline_audit import predict_model

    probabilities = np.column_stack([predict_model(h, x, CONFIG) for h in heads])
    probabilities /= np.maximum(1.0, probabilities.sum(1))[:, None]
    return probabilities


def selection(probabilities, threshold):
    p = np.asarray(probabilities, float).copy()
    p[REFERENCE] = 0
    utility = p[:, 0] - p[:, 1]
    choice = int(np.flatnonzero(utility >= utility.max() - 1e-10)[0])
    return REFERENCE if choice == REFERENCE or utility[choice] <= threshold else choice
