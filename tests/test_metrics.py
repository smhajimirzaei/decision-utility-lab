import pytest
from trust_imagination.metrics import brier_score, candidate_regret, reranking_benefit


def test_regret_is_relative_to_best_tested_candidate_and_handles_ties():
    assert candidate_regret([0.2, 0.7, 0.7], 0) == pytest.approx(0.5)
    assert candidate_regret([0.2, 0.7, 0.7], 2) == 0
    assert candidate_regret([0.3, 0.3], 1) == 0


def test_improvement_and_harm_have_opposite_signs():
    assert reranking_benefit([0.8, 0.2, 0.9], 1) == pytest.approx(-0.6)
    assert reranking_benefit([0.8, 0.2, 0.9], 2) == pytest.approx(0.1)


def test_brier_distinguishes_perfect_uncertain_and_wrong_predictions():
    assert brier_score([0, 1], [0, 1]) == 0
    assert brier_score([0.5, 0.5], [0, 1]) == 0.25
    assert brier_score([1, 0], [0, 1]) == 1


@pytest.mark.parametrize("values,index", [([], 0), ([float("nan")], 0), ([1], 2)])
def test_invalid_candidate_records_fail(values, index):
    with pytest.raises(ValueError):
        candidate_regret(values, index)


@pytest.mark.parametrize(
    "probabilities,labels", [([0.1], [0, 1]), ([1.1], [1]), ([0.2], [2])]
)
def test_invalid_calibration_records_fail(probabilities, labels):
    with pytest.raises(ValueError):
        brier_score(probabilities, labels)
