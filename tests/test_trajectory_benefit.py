"""Synthetic counterexamples for leakage, paired utility and chronological inputs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from trust_imagination.trajectory_benefit import (
    comparative_features,
    labels,
    selection,
    fit_benefit,
    predict_benefit,
)


def fixture():
    rng = np.random.default_rng(812)
    return {
        "executed_unique_pool": rng.uniform(-1, 1, (8, 50, 4)).astype("f"),
        "current_latent": np.zeros((1, 1, 1, 16, 16, 384), "f"),
        "current_proprio_latent": np.zeros((1, 1, 4, 16), "f"),
        "proprio": np.zeros(4, "f"),
        "predicted_latents": np.zeros((8, 10, 1, 16, 16, 384), "f"),
        "predicted_proprio_latents": np.zeros((8, 10, 4, 16), "f"),
    }


def test_physical_labels_cannot_change_deployment_features():
    d = fixture()
    projection = np.eye(384, 16, dtype="f")
    for kind in ("direct", "endpoint", "trajectory"):
        a = comparative_features(
            d, projection, ["task"], "task", 50, list(range(8)), 8, 0.25, kind, True
        )
        d.update(
            valid_mask=np.zeros((8, 10)),
            first_hit=np.arange(8),
            outcomes=np.ones((8, 3)),
            actual_future=np.nan,
        )
        b = comparative_features(
            d, projection, ["task"], "task", 50, list(range(8)), 8, 0.25, kind, True
        )
        np.testing.assert_array_equal(a, b)


def test_earlier_forecast_only_changes_trajectory():
    d = fixture()
    p = np.eye(384, 16, dtype="f")
    args = (p, ["task"], "task", 0, list(range(8)), 8, 0.25)
    a = comparative_features(d, *args, "endpoint", False)
    t = comparative_features(d, *args, "trajectory", False)
    d["predicted_latents"][4, 4] = 1
    np.testing.assert_array_equal(a, comparative_features(d, *args, "endpoint", False))
    assert not np.array_equal(t, comparative_features(d, *args, "trajectory", False))


def test_reference_and_paired_help_harm():
    y = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1], [1, 0, 0]])
    z = labels(y)
    np.testing.assert_allclose(z[0], [1 / 3, 0])
    np.testing.assert_array_equal(z[3], [0, 0])
    np.testing.assert_allclose(z[1], [0, 1 / 3])
    p = np.zeros((8, 2))
    p[4] = [0.5, 0.1]
    p[7] = [0.5, 0.1]
    assert selection(p, 0.2) == 4 and selection(p, 0.4) == 3


def test_two_output_map_and_probability_simplex():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(12, 8))
    t = rng.uniform(size=(12, 2))
    w = np.full(12, 1 / 12)
    heads = fit_benefit(x, t, w)
    p = predict_benefit(heads, x)
    assert p.shape == (12, 2) and np.all(p >= 0) and np.all(p.sum(1) <= 1 + 1e-12)
    assert heads[0]["coef"].shape == heads[1]["coef"].shape == (128,)
    np.testing.assert_array_equal(heads[0]["matrix"], heads[1]["matrix"])


def test_shared_four_bank_metadata_does_not_invent_shift():
    d = fixture()
    p = np.eye(384, 16, dtype="f")
    args = (p, ["task"], "task", 50, list(range(8)), 4)
    a = comparative_features(d, *args, 0, "trajectory", True)
    b = comparative_features(d, *args, 0.25, "trajectory", True)
    np.testing.assert_array_equal(a, b)
