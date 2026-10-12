import numpy as np

from trust_imagination.task_evaluator import (
    episode_weights,
    fit_head,
    predict,
    visual_features,
)


def test_spatial_pooling_preserves_coarse_quadrants():
    grid = np.zeros((16, 16, 384), dtype=np.float32)
    grid[:4, 4:8, 0] = 7
    projection = np.zeros((384, 16), dtype=np.float32)
    projection[0, 0] = 1
    result = visual_features(grid, projection).reshape(4, 4, 16)
    assert result[0, 1, 0] == 7
    assert np.count_nonzero(result) == 1


def test_episodes_receive_equal_total_weight_despite_candidate_count():
    rows = [{"episode_id": "a"}] * 2 + [{"episode_id": "b"}] * 8
    weights = episode_weights(rows)
    assert np.isclose(weights[:2].sum(), 0.5)
    assert np.isclose(weights[2:].sum(), 0.5)


def test_head_learns_signal_and_prediction_does_not_change_training_scaling():
    features = np.linspace(-2, 2, 40, dtype=np.float32)[:, None]
    target = (features[:, 0] > 0).astype(np.float32)
    head = fit_head(
        features, target, np.full(40, 1 / 40, np.float32), 0.001, updates=200
    )
    before = head["mean"].copy(), head["scale"].copy()
    assert np.mean((predict(head, features) - target) ** 2) < 0.04
    assert np.isfinite(predict(head, np.asarray([[1000]], np.float32))).all()
    assert np.array_equal(before[0], head["mean"])
    assert np.array_equal(before[1], head["scale"])
