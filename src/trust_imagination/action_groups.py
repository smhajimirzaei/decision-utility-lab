"""The JEPA Meta-World action layout, independent of torch or the simulator."""

from __future__ import annotations

import numpy as np


def pack_actions(actions, mean, std, frameskip=5):
    """Raw B,(H*F),4 -> normalized H,B,(F*4), chronological within each group.

    Match upstream plan_evaluator's inverse `t (f d) -> (t f) d`.
    The caller must execute the same raw actions; clipping here would silently
    make model conditioning and simulator execution disagree.
    """
    raw = np.asarray(actions, dtype=np.float32)
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != 4 or raw.shape[1] == 0:
        raise ValueError("Expected nonempty B,T,4 raw action array")
    if frameskip < 1 or raw.shape[1] % frameskip:
        raise ValueError("Action length must be divisible by positive frameskip")
    if mean.shape != (4,) or std.shape != (4,) or np.any(std <= 0):
        raise ValueError("Expected four finite means and positive standard deviations")
    if not all(np.isfinite(x).all() for x in (raw, mean, std)):
        raise ValueError("Action inputs must be finite")
    if np.any(np.abs(raw) > 1.000001):
        raise ValueError("Raw actions must already respect simulator bounds [-1,1]")
    batch, steps, _ = raw.shape
    normalized = (raw - mean) / std
    return (
        normalized.reshape(batch, steps // frameskip, frameskip * 4)
        .transpose(1, 0, 2)
        .copy()
    )
