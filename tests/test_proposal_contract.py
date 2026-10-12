import json
from pathlib import Path
import numpy as np
import pytest
from trust_imagination.proposal_contract import (
    pool_contract,
    runtime_decision,
    validate_config,
)


def test_aliases_execute_once_but_banks_retain_probability_mass_and_shared_base():
    raw = np.zeros((8, 50, 4), dtype=np.float32)
    d = pool_contract(raw, 9)
    assert d["unique"].shape == (5, 50, 4)
    assert d["banks"]["policy"] == [0] * 8
    assert d["banks"]["mixed"][:4] == [0] * 4
    for logical, unique in zip(d["logical"], d["mapping"]):
        np.testing.assert_array_equal(logical, d["unique"][unique])


def test_runtime_guard_cannot_use_outcomes_and_declines_insufficient_remaining_time():
    runtime = {
        "safety_factor": 1.25,
        "max_states_per_episode": 4,
        "fixed_overhead_seconds": 60,
        "episode_timeout_seconds": 1800,
        "remaining_margin_seconds": 120,
    }
    episode = {
        "states": [{"seconds_including_branch_audit": 200}],
        "seconds_including_branch_audits": 240,
        "success": False,
    }
    a = runtime_decision(episode, 3000, runtime)
    episode["success"] = True
    assert a == runtime_decision(episode, 3000, runtime) and a["allowed"]
    assert not runtime_decision(episode, a["estimated_seconds"] + 119, runtime)[
        "allowed"
    ]
    slow = {
        "states": [{"seconds_including_branch_audit": 500}],
        "seconds_including_branch_audits": 540,
    }
    assert not runtime_decision(slow, 6000, runtime)["allowed"]


def test_collection_contract_rejects_cadence_changes_and_reserved_seed_leakage():
    cfg = json.loads(Path("configs/proposal_opportunity_20261011.json").read_text())
    validate_config(cfg)
    cfg["main_seeds"] = [27, 36]
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg["main_seeds"] = [27, 28]
    cfg["execution_prefix"] = 10
    with pytest.raises(ValueError):
        validate_config(cfg)
