"""Versioned, input-only radius experiment; original collector remains frozen."""

import numpy as np
from .proposal_opportunity import comparison_banks

REPAIR_RADIUS = 0.25


def radius_repair_pool(raw_proposals, seed):
    original = comparison_banks(raw_proposals, seed)
    reference = original["pool"][0]
    repairs = []
    for direction in original["directions"]:
        for sign in (1, -1):
            proposal = reference.copy()
            proposal[:, :3] += np.asarray(
                sign * REPAIR_RADIUS * original["envelope"][:, None] * direction,
                dtype=np.float32,
            )
            repairs.append(np.clip(proposal, -1, 1))
    logical = np.concatenate((original["pool"], np.stack(repairs)))
    unique = []
    mapping = []
    seen = {}
    for proposal in logical:
        key = proposal.tobytes()
        if key not in seen:
            seen[key] = len(unique)
            unique.append(proposal.copy())
        mapping.append(seen[key])
    mapping = np.asarray(mapping, dtype=np.int64)
    bank_logical = {
        "policy": list(range(8)),
        "mixed": [0, 1, 2, 3, 8, 9, 10, 11],
        "repair": [0, 1, 2, 3, 12, 13, 14, 15],
    }
    return {
        "logical": logical,
        "unique": np.asarray(unique),
        "mapping": mapping,
        "bank_logical_indices": bank_logical,
        "banks": {name: mapping[ix].tolist() for name, ix in bank_logical.items()},
        "directions": original["directions"],
        "envelope": original["envelope"],
        "radii": {"mixed": 0.1, "repair": REPAIR_RADIUS},
    }
