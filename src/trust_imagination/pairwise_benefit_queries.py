"""Rebuild deployment-available comparison features for an arbitrary reference."""

import numpy as np


def reference_query(features, kind, metadata, reference):
    x = np.asarray(features, float)
    size = len(x)
    if not 0 <= reference < size:
        raise ValueError("Reference outside bank")
    current = x[:, :40]
    actions = x[:, 40:240].astype(np.float32)
    ref = np.broadcast_to(actions[reference], actions.shape)
    parts = [current, actions, ref, actions - ref]
    if kind in ("endpoint", "trajectory"):
        width = 32 if kind == "endpoint" else 320
        future = x[:, 640 : 640 + width].astype(np.float32)
        ref_future = np.broadcast_to(future[reference], future.shape)
        parts += [future, ref_future, future - ref_future]
    elif kind != "direct":
        raise ValueError("Unknown readout")
    if metadata:
        rms = np.sqrt(np.mean((actions - ref) ** 2, axis=1))
        parts.append(
            np.c_[
                x[:, -5:-3],
                rms,
                np.full(size, rms.mean()),
                np.mean(np.abs(actions) == 1, axis=1),
            ]
        )
    value = np.concatenate(parts, axis=1).astype(float)
    if not np.isfinite(value).all():
        raise ValueError("Invalid comparison input")
    return value


def paired_labels(outcomes, reference):
    y = np.asarray(outcomes, float)
    delta = y - y[reference]
    return np.c_[np.mean(delta > 0, axis=1), np.mean(delta < 0, axis=1)]
