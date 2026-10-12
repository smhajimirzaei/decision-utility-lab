"""Pure comparison identities and timing rules; no simulator/outcome access."""

import numpy as np
from .proposal_opportunity import comparison_banks


def validate_config(cfg):
    expected = {
        "execution_prefix": 50,
        "continuation_chunk": 50,
        "baseline_execution_chunk": 50,
        "episode_horizon": 400,
        "policy_samples": 8,
        "logical_pool_size": 12,
        "continuation_repeats": 3,
        "world_transitions": 10,
        "world_frameskip": 5,
        "perturbation_amplitude": 0.1,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Collection semantics differ from implemented contract")
    if cfg["banks"] != {"policy": list(range(8)), "mixed": [0, 1, 2, 3, 8, 9, 10, 11]}:
        raise ValueError("Bank identity differs from declared comparison")
    used = {cfg["pilot_seed"], *cfg["main_seeds"]}
    if len(used) != 1 + len(cfg["main_seeds"]) or used & set(
        cfg["reserved_validation_test_seeds"]
    ):
        raise ValueError("Pilot/main or reserved seed overlap")


def pool_contract(raw, seed):
    design = comparison_banks(raw, seed)
    logical = design["pool"]
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
    banks = {
        name: mapping[design[f"{name}_indices"]].tolist()
        for name in ("policy", "mixed")
    }
    if banks["policy"][:4] != banks["mixed"][:4] or banks["policy"][0] != 0:
        raise ValueError("Shared base/first-four identity changed")
    return {
        "logical": logical,
        "unique": np.asarray(unique),
        "mapping": mapping,
        "banks": banks,
        "directions": design["directions"],
        "envelope": design["envelope"],
    }


def runtime_decision(pilot_episode, remaining_seconds, runtime):
    states = pilot_episode["states"]
    if len(states) != 1:
        raise ValueError("One-state timing pilot required")
    state_time = states[0]["seconds_including_branch_audit"]
    overhead = max(0, pilot_episode["seconds_including_branch_audits"] - state_time)
    estimate = (
        runtime["safety_factor"]
        * (runtime["max_states_per_episode"] * state_time + overhead)
        + runtime["fixed_overhead_seconds"]
    )
    allowed = (
        estimate <= runtime["episode_timeout_seconds"]
        and estimate <= remaining_seconds - runtime["remaining_margin_seconds"]
    )
    return {
        "estimated_seconds": estimate,
        "remaining_seconds": remaining_seconds,
        "allowed": bool(allowed),
        "rule": "Task-matched technical pilot timing only; no success/opportunity-based progression",
    }
