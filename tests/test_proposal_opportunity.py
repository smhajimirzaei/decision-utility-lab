import numpy as np
import pytest
from trust_imagination.proposal_opportunity import comparison_banks


def test_banks_preserve_base_and_shared_policy_with_deterministic_bounded_changes():
    raw = np.random.default_rng(1).uniform(-2, 2, (8, 50, 4)).astype(np.float32)
    a = comparison_banks(raw, 100)
    b = comparison_banks(raw, 100)
    np.testing.assert_array_equal(a["pool"], b["pool"])
    np.testing.assert_array_equal(a["pool"][:8], np.clip(raw, -1, 1))
    assert a["policy_indices"][0] == a["mixed_indices"][0] == 0
    assert len(set(a["policy_indices"])) == len(set(a["mixed_indices"])) == 8
    assert np.max(np.abs(a["pool"])) <= 1
    for proposal in a["pool"][8:]:
        np.testing.assert_array_equal(proposal[:, 3], a["pool"][0, :, 3])
        assert (
            np.max(np.linalg.norm(proposal[:, :3] - a["pool"][0, :, :3], axis=1))
            <= 0.100001
        )
    np.testing.assert_array_equal(
        raw, np.random.default_rng(1).uniform(-2, 2, (8, 50, 4)).astype(np.float32)
    )


def test_design_rejects_nonfinite_wrong_length_and_unreported_amplitude():
    with pytest.raises(ValueError):
        comparison_banks(np.zeros((8, 10, 4)), 1)
    with pytest.raises(ValueError):
        comparison_banks(np.full((8, 50, 4), np.nan), 1)
    with pytest.raises(ValueError):
        comparison_banks(np.zeros((8, 50, 4)), 1, amplitude=0.2)
