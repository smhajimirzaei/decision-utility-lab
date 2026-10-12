import pytest
from trust_imagination.repeated_outcomes import continuation_seeds, summarize_repeats


def test_repeats_preserve_original_seed_and_are_distinct_across_states():
    original = 500000 + 20 * 1000 + 50
    seeds = continuation_seeds(20, 50, 3)
    assert seeds[0] == original and len(set(seeds)) == 3
    assert not set(seeds) & set(continuation_seeds(21, 50, 3))
    with pytest.raises(ValueError):
        continuation_seeds(20, 50, 0)


def test_repeated_effects_pair_each_candidate_with_its_own_repeat_base():
    outcomes = [
        [{"success": y} for y in row] for row in [[0, 1, 0], [1, 0, 1], [0, 1, 1]]
    ]
    result = summarize_repeats(outcomes)
    assert result["empirical_success_frequency_by_candidate"] == [1 / 3, 2 / 3, 2 / 3]
    assert result["empirical_help_frequency_by_candidate"] == [0, 2 / 3, 1 / 3]
    assert result["empirical_harm_frequency_by_candidate"] == [0, 1 / 3, 0]
    assert result["empirical_delta_vs_base_by_candidate"] == [0, 1 / 3, 1 / 3]
