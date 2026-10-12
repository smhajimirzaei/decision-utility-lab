"""Decision metrics with explicit reference and input validation."""

import math


def _finite(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected nonempty finite values.")
    return values


def candidate_regret(realized_values, selected_index: int) -> float:
    """Unnormalized regret relative to the best *tested* candidate."""
    values = _finite(realized_values)
    if not 0 <= selected_index < len(values):
        raise ValueError("Selected index is outside the candidate bank.")
    return max(values) - values[selected_index]


def reranking_benefit(
    realized_values, selected_index: int, base_index: int = 0
) -> float:
    values = _finite(realized_values)
    if not (0 <= selected_index < len(values) and 0 <= base_index < len(values)):
        raise ValueError("Candidate index is outside the bank.")
    return values[selected_index] - values[base_index]


def brier_score(probabilities, labels) -> float:
    probabilities = _finite(probabilities)
    labels = list(labels)
    if len(probabilities) != len(labels) or any(
        label not in (0, 1) for label in labels
    ):
        raise ValueError("Labels must be binary and match the predictions.")
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be in [0, 1].")
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
