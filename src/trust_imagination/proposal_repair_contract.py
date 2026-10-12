"""Three-bank radius contract; original two-bank declaration stays unchanged."""

from .proposal_contract import validate_config as validate_original, runtime_decision
from .proposal_radius_repair import radius_repair_pool as pool_contract


def validate_config(cfg):
    banks = {
        "policy": list(range(8)),
        "mixed": [0, 1, 2, 3, 8, 9, 10, 11],
        "repair": [0, 1, 2, 3, 12, 13, 14, 15],
    }
    if (
        cfg.get("logical_pool_size") != 16
        or cfg.get("repair_amplitude") != 0.25
        or cfg.get("banks") != banks
    ):
        raise ValueError("Declared radius-only comparison differs")
    old = {
        **cfg,
        "logical_pool_size": 12,
        "banks": {k: v for k, v in banks.items() if k != "repair"},
    }
    validate_original(old)
