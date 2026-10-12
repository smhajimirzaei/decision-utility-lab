import numpy as np
import pytest
from trust_imagination.cadence_value_diagnostic import (
    complete_support,
    fit,
    metrics,
    pool,
    predict,
)


def test_terminal_mask_and_early_success_do_not_admit_padded_future():
    mask = np.ones((8, 10), dtype=bool)
    early = np.zeros(8, dtype=bool)
    assert complete_support(mask, early)
    mask[6, 9] = False
    assert not complete_support(mask, early)
    mask[6, 9] = True
    early[6] = True
    assert not complete_support(mask, early)


def test_pooling_preserves_position_and_global_agrees_with_spatial_mean():
    x = np.zeros((16, 16, 384), dtype=np.float32)
    x[:4, :4, 0] = 1
    projection = np.eye(384, 16, dtype=np.float32)
    spatial = pool(x, projection, "spatial").reshape(16, 16)
    assert spatial[0, 0] == 1 and spatial[1:, 0].sum() == 0
    np.testing.assert_allclose(spatial.mean(0), pool(x, projection, "global"))


def test_weighted_dual_ridge_matches_primal_and_only_uses_training_normalization():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(9, 4))
    y = rng.uniform(size=9)
    w = np.arange(1, 10, dtype=float)
    w /= w.sum()
    cfg = {"normalization_floor": 0.05, "normalization_clip": 10, "l2": 0.1}
    h = fit(x, y, w, cfg)
    z = (x - h["mean"]) / h["scale"] - h["center"]
    expected = np.linalg.solve(
        z.T @ (w[:, None] * z) + 0.1 * np.eye(4), z.T @ (w * (y - w @ y))
    )
    np.testing.assert_allclose(h["coef"], expected, atol=1e-12)
    before = h["mean"].copy()
    predict(h, np.full((2, 4), 1000.0), cfg)
    np.testing.assert_array_equal(before, h["mean"])
    with pytest.raises(ValueError):
        predict(h, np.full((1, 4), np.nan), cfg)


def test_selection_ties_retain_base_and_brier_uses_each_repeat():
    rows = [
        {"state_id": "s", "episode_id": "e", "candidate_index": i, "outcomes": y}
        for i, y in enumerate(([0, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0]))
    ]
    tie = metrics(rows, np.ones(4) * 0.5, 4)
    assert tie["decisions"][0]["chosen"] == 0 and tie["gain"] == 0
    chosen = metrics(rows, np.asarray([0.1, 0.9, 0.3, 0.1]), 4)
    assert chosen["success"] == 1 and chosen["gain"] == 1 and chosen["help"] == 1
    assert chosen["informative_pair_ordering"] == 1
    p = np.asarray([0.1, 0.9, 0.3, 0.1])
    y = np.asarray([r["outcomes"] for r in rows])
    assert chosen["per_repeat_brier"] == pytest.approx(np.mean((p[:, None] - y) ** 2))
