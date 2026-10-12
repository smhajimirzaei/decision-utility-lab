"""Synthetic terminal traces exercise reuse/masking without a GPU/simulator."""

import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from trust_imagination.cadence_contract import forecast_valid_mask
from trust_imagination.proposal_contract import pool_contract
from trust_imagination.repeated_outcomes import continuation_seeds, summarize_repeats

spec = importlib.util.spec_from_file_location(
    "proposal_trace_audit", Path("scripts/analyze_proposal_opportunity.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def source(tmp_path):
    cfg = json.loads(Path("configs/proposal_opportunity_20261011.json").read_text())
    folder = tmp_path / "pilot"
    path = folder / "push-v3_seed29/state0"
    (path / "features").mkdir(parents=True)
    (path / "outcomes").mkdir()
    raw = np.zeros((8, 50, 4), np.float32)
    d = pool_contract(raw, 229000)
    pool = d["unique"]
    n = len(pool)
    stop = np.asarray([50, 10, 20, 50, 50])
    mask = forecast_valid_mask(stop)
    latent = np.ones((n, 10, 1, 2, 2, 3), np.float32)
    predicted = np.zeros_like(latent)
    np.savez_compressed(
        path / "features/deployment_inputs.npz",
        raw_proposals=raw,
        logical_pool=d["logical"],
        executed_unique_pool=pool,
        pool_to_unique=d["mapping"],
        predicted_latents=predicted,
        current_latent=np.zeros((1, 1, 1, 2, 2, 3), np.float32),
    )
    np.savez_compressed(
        path / "outcomes/actual_future_features.npz",
        latent=latent,
        valid_mask=mask,
        first_stop_or_limit=stop,
    )
    repeats = []
    for repeat in range(3):
        outcomes = []
        for c in range(n):
            steps = int(stop[c]) if c in (1, 2) else 400
            actions = np.zeros((steps, 4), np.float32)
            actions[: min(50, steps)] = pool[c, : min(50, steps)]
            flags = np.zeros((steps, 3), bool)
            if c == 1:
                flags[-1, 2] = True
            if c == 2:
                flags[-1, 0] = True
            np.savez_compressed(
                path / "outcomes" / f"unique{c}_repeat{repeat}_trace.npz",
                actions=actions,
                outcomes=flags,
                rewards=np.zeros(steps),
            )
            outcomes.append(
                {
                    "success": c == 1,
                    "final_success": c == 1,
                    "steps": steps,
                    "reward_sum": 0.0,
                }
            )
        repeats.append(outcomes)
    conditions = []
    for name, ix in d["banks"].items():
        b = [[r[i] for i in ix] for r in repeats]
        early = [i == 1 for i in ix]
        conditions.append(
            {
                "name": name,
                "unique_indices": ix,
                "branch_repeats": b,
                "execution_prefix": 50,
                "continuation_chunk": 50,
                "remaining_horizon": 400,
                "continuation_seeds": continuation_seeds(29, 0, 3),
                "early_success_within_prefix": early,
                "scorer_training_eligible": [not v for v in early],
                "distinct_executed_proposals": len(set(ix)),
                "action_at_bound_fraction": 0.0,
                **summarize_repeats(b),
            }
        )
    error = [[1.0 if valid else None for valid in r] for r in mask]
    record = {
        "episode_id": "push-v3_seed29",
        "seed": 29,
        "task": "push-v3",
        "state_step": 0,
        "simulator_replay": {"passed": True},
        "proposal_seed": 129000,
        "perturbation_seed": 229000,
        "unique_count": n,
        "unique_pool_sha256": audit.hashlib.sha256(pool.tobytes()).hexdigest(),
        "bank_unique_indices": d["banks"],
        "forecast_prefix_max_abs_error": {"visual": 0.0, "proprio": 0.0},
        "forecast_valid_mask": mask.tolist(),
        "first_stop_or_limit": stop.tolist(),
        "model_mse": error,
        "persistence_mse": error,
        "unique_branch_repeats": repeats,
        "conditions": conditions,
        "continuation_replay_by_repeat": [{"actions": 0.0}] * 3,
    }
    summary = {
        "stage": "pilot",
        "status": "passed",
        "episodes": [{"task": "push-v3", "seed": 29, "steps": 400, "states": [record]}],
    }
    (folder / "summary.json").write_text(json.dumps(summary))
    return folder, cfg, summary


def test_deduplicated_bank_audit_counts_physical_traces_and_retains_new_help(source):
    folder, cfg, _ = source
    rows, traces, replays = audit.validate_stage(
        folder, cfg, "pilot", [("push-v3", 29)]
    )
    assert traces == 15 and replays == 3 and rows[0]["unique_count"] == 5
    assert not rows[0]["banks"]["policy"]["8"]["helpful_opportunity"]
    assert rows[0]["banks"]["mixed"]["8"]["helpful_opportunity"]
    assert rows[0]["banks"]["policy"]["4"] == rows[0]["banks"]["mixed"]["4"]


def test_shared_branch_relabeling_cannot_pass_audit(source):
    folder, cfg, summary = source
    summary["episodes"][0]["states"][0]["conditions"][1]["branch_repeats"][0][0][
        "success"
    ] = True
    (folder / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        audit.validate_stage(folder, cfg, "pilot", [("push-v3", 29)])
