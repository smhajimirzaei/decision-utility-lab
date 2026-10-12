import numpy as np
from trust_imagination.readout_study import (
    fit_linear,
    linear_predict,
    selections,
    variant,
)
from trust_imagination.task_evaluator import episode_weights


def test_linear_generalizes_on_signal_and_preserves_training_scaling():
    rng = np.random.default_rng(15)
    train = rng.normal(size=(160, 3)).astype(np.float32)
    y = (train[:, 0] > 0).astype(np.float32)
    weights = np.full(len(train), 1 / len(train), np.float32)
    head = fit_linear(train, y, weights, 0.001)
    test = rng.normal(size=(80, 3)).astype(np.float32)
    assert np.mean((linear_predict(head, test) - (test[:, 0] > 0)) ** 2) < 0.06
    original = head["mean"].copy()
    linear_predict(head, np.full((1, 3), 1e6, np.float32))
    np.testing.assert_array_equal(original, head["mean"])


def test_global_features_average_spatial_cells_and_keep_proprio_text_horizon():
    x = np.zeros((2, 305), np.float32)
    x[:, :256] = np.tile(np.arange(16, dtype=np.float32), 16)
    x[:, 256:] = 7
    actual = variant(x, "global")
    assert actual.shape == (2, 65)
    np.testing.assert_array_equal(actual[0, :16], np.arange(16))
    np.testing.assert_array_equal(actual[:, 16:], np.full((2, 49), 7))
    assert variant(x, "spatial_no_proprio").shape == (2, 289)


def test_uniform_control_exact_expectation_and_episode_weighting():
    rows = []
    for state, outcomes in [
        ("a/0", [0, 1, 1, 1]),
        ("a/50", [0, 0, 0, 0]),
        ("b/0", [1, 0, 0, 0]),
    ]:
        for c, y in enumerate(outcomes):
            rows.append(
                {
                    "state_id": state,
                    "episode_id": state.split("/")[0],
                    "split": "validation",
                    "candidate_index": c,
                    "success": y,
                    "eligible": True,
                }
            )
    result = selections(rows, "random", 4, "validation")
    assert result["selected_success_states"] == 1
    assert result["helps"] == 0.75 and result["hurts"] == 0.75
    assert result["episode_mean_success"] == 0.3125
    assert selections(rows, "base", 4, "validation")["episode_mean_success"] == 0.5
