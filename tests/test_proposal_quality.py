import numpy as np
import pytest
from trust_imagination.proposal_quality import (
    action_measurements,
    latent_measurements,
    report_trace_timing,
    episode_mean,
    outcome_class,
)


def test_logical_duplicate_weight_and_channel_separation():
    x = np.zeros((4, 50, 4), np.float32)
    x[3, :, 0] = 1
    x[:, :, 3] = -1
    m = action_measurements(x)
    assert m["translation"]["pair_rms"] == pytest.approx(0.5 / np.sqrt(3))
    assert m["gripper"]["pair_rms"] == 0
    assert m["byte_unique_logical_actions"] == 2
    assert m["exact_bound_by_channel"] == [0.25, 0, 0, 1]
    assert m["translation_participation"] == pytest.approx(1)
    assert action_measurements(np.zeros_like(x))["translation_participation"] == 0


def test_first_success_and_failure_censoring_are_not_horizon_imputations():
    x = np.zeros((36, 3), bool)
    x[-1, 2] = True
    m = report_trace_timing(x, 50)
    assert m["first_success_relative"] == 36 and m["first_success_absolute"] == 86
    assert m["success_within_prefix50"] and not m["failure_censored"]
    failure = report_trace_timing(np.zeros((350, 3), bool), 50)
    assert failure["first_success_relative"] is None and failure["failure_censored"]
    assert failure["stop_reason"] == "horizon"
    x[2, 0] = True
    with pytest.raises(ValueError):
        report_trace_timing(x, 50)
    with pytest.raises(ValueError):
        report_trace_timing(np.zeros((349, 3), bool), 50)


def test_unsupported_cosine_and_episode_weighting():
    m = latent_measurements(np.zeros((4, 16)), np.zeros(16))
    assert (
        m["reference_cosine_distance"] is None
        and m["pair_rms_over_reference_change"] is None
    )
    rows = [{"episode_id": "a", "v": 1}] * 4 + [{"episode_id": "b", "v": 0}]
    assert episode_mean(rows, lambda r: r["v"])["mean"] == 0.5
    assert outcome_class([1, 1, 0]) == "harmful_only"
    assert outcome_class([0, 1, 0]) == "helpful_only"
    assert outcome_class([1 / 3, 1 / 3]) == "tied_partial"
