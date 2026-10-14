"""Reference identity, paired direction and unchanged deployment3 queries."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import numpy as np
from test_trajectory_benefit import fixture
from trust_imagination.trajectory_benefit import comparative_features
from trust_imagination.pairwise_benefit_queries import reference_query, paired_labels


def test_reference3_reproduces_all_deployment_features_exactly():
    d = fixture()
    p = np.eye(384, 16, dtype="f")
    for kind in ("direct", "endpoint", "trajectory"):
        for metadata in (False, True):
            x = comparative_features(
                d, p, ["a", "b", "c"], "a", 50, list(range(8)), 8, 0.25, kind, metadata
            )
            np.testing.assert_array_equal(reference_query(x, kind, metadata, 3), x)


def test_reversed_queries_swap_help_and_harm_and_self_is_zero():
    y = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])
    a = paired_labels(y, 0)
    b = paired_labels(y, 1)
    np.testing.assert_array_equal(a[1], b[0][::-1])
    np.testing.assert_array_equal(a[0], [0, 0])


def test_reference_actions_and_geometry_follow_query():
    d = fixture()
    p = np.eye(384, 16, dtype="f")
    x = comparative_features(
        d, p, ["a", "b", "c"], "a", 50, list(range(8)), 8, 0.25, "direct", True
    )
    q = reference_query(x, "direct", True, 4)
    np.testing.assert_array_equal(
        q[:, 240:440], np.broadcast_to(x[4, 40:240], (8, 200))
    )
    assert q[4, -3] == 0 and np.all(q[4, 440:640] == 0)
