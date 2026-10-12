import numpy as np
import pytest
from trust_imagination.baseline_audit import (
    inputs,
    fit_model,
    predict_model,
    strata,
    state_episode_id,
)
from trust_imagination.cadence_value_diagnostic import weights

CFG = {
    "normalization_floor": 0.05,
    "normalization_clip": 10.0,
    "l2": 0.1,
    "random_map_width": 16,
}


def test_summary_identity_uses_state_field_and_rejects_misassociated_labels():
    episode = {"task": "reach-v3", "seed": 24, "states": []}
    state = {"episode_id": "reach-v3_seed24", "task": "reach-v3", "seed": 24}
    assert state_episode_id(episode, state) == "reach-v3_seed24"
    with pytest.raises(ValueError):
        state_episode_id(episode, {**state, "seed": 25})
    with pytest.raises(ValueError):
        state_episode_id(episode, {**state, "episode_id": "push-v3_seed24"})


def test_nonlinear_readout_matches_independent_primal_and_does_not_fit_test_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(11, 5))
    y = rng.uniform(size=11)
    w = np.arange(1, 12, dtype=float)
    w /= w.sum()
    h = fit_model(x, y, w, CFG, 601)
    z = np.clip((x - h["mean"]) / h["scale"], -10, 10)
    phi = np.sqrt(2) * np.cos(z @ h["matrix"] + h["phase"]) - h["center"]
    expected = np.linalg.solve(
        phi.T @ (w[:, None] * phi) + 0.1 * np.eye(16), phi.T @ (w * (y - w @ y))
    )
    np.testing.assert_allclose(h["coef"], expected, atol=1e-12)
    snapshots = {k: v.copy() for k, v in h.items()}
    predict_model(h, np.full((3, 5), 1e8), CFG)
    for k, v in h.items():
        np.testing.assert_array_equal(v, snapshots[k])
    np.testing.assert_allclose(h["mean"], w @ x, atol=1e-12)
    with pytest.raises(ValueError):
        predict_model(h, np.full((1, 5), np.nan), CFG)


def test_nonvisual_inputs_ignore_visual_and_outcome_sentinels():
    blocks = {
        name: np.full((4, 2), i, dtype=float)
        for i, name in enumerate(
            (
                "current",
                "nonvisual_current",
                "actions",
                "predicted_visual",
                "predicted_proprio",
            )
        )
    }
    blocks["actual_future"] = np.full((4, 100), np.nan)
    blocks["terminal_mask"] = np.zeros((4, 8))
    expected = np.c_[
        blocks["nonvisual_current"], blocks["actions"], blocks["predicted_proprio"]
    ]
    np.testing.assert_array_equal(
        inputs(blocks, "nonvisual_direct_predicted"), expected
    )
    blocks["predicted_visual"][:] = np.nan
    blocks["current"][:] = np.nan
    np.testing.assert_array_equal(
        inputs(blocks, "nonvisual_direct_predicted"), expected
    )
    with pytest.raises(ValueError):
        inputs(blocks, "direct_predicted_both")


def test_current_only_has_no_within_state_selection_signal():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(12, 3))
    y = rng.uniform(size=12)
    h = fit_model(x, y, np.ones(12) / 12, CFG)
    scores = predict_model(h, np.broadcast_to(x[0], (8, 3)), CFG)
    np.testing.assert_array_equal(scores, np.repeat(scores[0], 8))


def test_episode_weights_and_descriptive_strata_do_not_multiply_evidence():
    rows = [{"episode_id": "a"}] * 8 + [{"episode_id": "b"}] * 24
    w = weights(rows)
    assert w[:8].sum() == pytest.approx(0.5) and w[8:].sum() == pytest.approx(0.5)
    assert strata([50, 50], [False, False]) == "union_complete_no_early"
    assert strata([29, 50], [True, False]) == "union_early_success"
    assert strata([29, 50], [False, False]) == "union_incomplete_no_early"
