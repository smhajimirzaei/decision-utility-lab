import numpy as np
import pytest
from trust_imagination.cadence_contract import (
    prefix_bank,
    forecast_valid_mask,
    condition_metrics,
    trace_prefix_success,
)


def test_both_contracts_execute_identical_bounded_nested_actions():
    proposals = np.linspace(-2, 2, 8 * 50 * 4).reshape(8, 50, 4)
    assert np.array_equal(
        prefix_bank(proposals, 10), prefix_bank(proposals, 50)[:, :10]
    )
    assert prefix_bank(proposals, 50).dtype == np.float32
    assert np.max(np.abs(prefix_bank(proposals, 50))) == 1
    with pytest.raises(ValueError):
        prefix_bank(proposals[:, :10], 10)
    with pytest.raises(ValueError):
        prefix_bank(proposals, 20)


def test_forecast_mask_excludes_post_stop_padding_but_keeps_terminal_boundary():
    masks = forecast_valid_mask([3, 5, 49, 50])
    assert masks.sum(1).tolist() == [0, 1, 9, 10]
    assert masks[1, 0] and not masks[1, 1]


def test_matched_outcomes_do_not_use_best_candidate_per_repeat():
    outcomes = [[0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0]]
    m = condition_metrics(outcomes, [False] * 4, 4)
    assert m["best_measured_offline"] == 1 / 3
    assert not m["helpful_opportunity"] and not m["harmful_opportunity"]
    assert (
        m["alternative_help_frequency"] == 1 / 9
        and m["alternative_harm_frequency"] == 1 / 9
    )
    assert m["all_candidates_post_prefix_eligible"]


def test_early_success_is_included_but_separate_from_value_eligibility():
    outcomes = [[0, 1, 0, 0]] * 3
    m = condition_metrics(outcomes, [False, True, False, False], 4)
    assert m["helpful_opportunity"] and m["best_measured_offline"] == 1
    assert (
        m["early_prefix_candidate_fraction"] == 0.25
        and not m["all_candidates_post_prefix_eligible"]
    )
    flags = np.zeros((30, 3), dtype=bool)
    flags[29, 2] = True
    assert not trace_prefix_success(flags, 10) and trace_prefix_success(flags, 50)
