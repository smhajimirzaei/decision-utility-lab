"""Pure metadata helpers for matched stochastic continuation repeats."""

import numpy as np


def continuation_seeds(episode_seed, state_step, repeats):
    if repeats < 1 or repeats > 100 or episode_seed < 0 or not 0 <= state_step < 400:
        raise ValueError("Invalid repeat seed contract")
    seeds = [
        500000 + episode_seed * 1000 + state_step + repeat * 10000000
        for repeat in range(repeats)
    ]
    if max(seeds) >= 2**31:
        raise ValueError("Repeat seeds exceed contract range")
    return seeds


def summarize_repeats(outcomes):
    values = np.asarray([[int(b["success"]) for b in repeat] for repeat in outcomes])
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("Require repeat-by-candidate outcomes including fixed base")
    base = values[:, :1]
    return {
        "empirical_success_frequency_by_candidate": values.mean(0).tolist(),
        "empirical_help_frequency_by_candidate": ((values - base) > 0).mean(0).tolist(),
        "empirical_harm_frequency_by_candidate": ((values - base) < 0).mean(0).tolist(),
        "empirical_delta_vs_base_by_candidate": (values - base).mean(0).tolist(),
        "repeat_labels_relative_to_candidate0": [
            [
                (
                    "helps"
                    if y > b
                    else (
                        "hurts"
                        if y < b
                        else "tie_both_success" if b else "tie_both_failure"
                    )
                )
                for y in row
            ]
            for row, b in zip(values, values[:, 0])
        ],
        "scope": "Finite matched-seed empirical frequencies, not a proven expected benefit",
    }
