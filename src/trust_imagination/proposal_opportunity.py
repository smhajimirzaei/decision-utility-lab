"""Declared policy/mixed bank construction with no simulator or outcome inputs."""

import numpy as np


def comparison_banks(raw_proposals, seed, amplitude=0.1):
    raw = np.asarray(raw_proposals, dtype=np.float32)
    if raw.shape != (8, 50, 4) or not np.isfinite(raw).all():
        raise ValueError("Eight finite native50-action policy proposals required")
    if amplitude != 0.1:
        raise ValueError("This development contract fixes amplitude0.1")
    policy = np.clip(raw, -1, 1)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(2, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    envelope = np.sin(np.pi * (np.arange(50) + 0.5) / 50) ** 2
    alternatives = []
    for direction in directions:
        for sign in (1, -1):
            proposal = policy[0].copy()
            proposal[:, :3] += np.asarray(
                sign * amplitude * envelope[:, None] * direction, dtype=np.float32
            )
            alternatives.append(np.clip(proposal, -1, 1))
    unique_pool = np.concatenate((policy, np.stack(alternatives)), axis=0)
    return {
        "pool": unique_pool,
        "policy_indices": list(range(8)),
        "mixed_indices": [0, 1, 2, 3, 8, 9, 10, 11],
        "directions": directions,
        "envelope": envelope,
        "note": "Only shared candidates0..3 are reused. Perturbations preserve base gripper commands; clipping can create equal actions and must be reported.",
    }
