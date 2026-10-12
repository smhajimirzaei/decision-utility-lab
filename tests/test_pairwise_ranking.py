import json
import numpy as np
import pytest

from trust_imagination.pairwise_ranking import (
    fit_ranker,
    pairs,
    pair_metrics,
    selection_metrics,
    utility,
)
from trust_imagination.ranking_study import development


def rows_for(states):
    rows = []
    for sid, outcomes in states:
        for c, y in enumerate(outcomes):
            rows.append(
                {
                    "state_id": sid,
                    "episode_id": sid.split("/")[0],
                    "candidate_index": c,
                    "eligible": True,
                    "outcomes": [y] if np.isscalar(y) else y,
                }
            )
    return rows


def test_pairs_retain_ties_and_weight_whole_episodes_then_states():
    rows = rows_for([("a/0", [0, 1, 1]), ("a/50", [0, 0, 0]), ("b/0", [1, 0])])
    data = pairs(rows)
    assert len(data["target"]) == 7
    assert np.count_nonzero(data["target"] == 0) == 4
    assert sum(
        w for w, e in zip(data["weights"], data["episodes"]) if e == "a"
    ) == pytest.approx(0.5)
    assert sum(
        w for w, s in zip(data["weights"], data["states"]) if s == "a/0"
    ) == pytest.approx(0.25)
    assert data["weights"][-1] == pytest.approx(0.5)


def test_pair_error_distinguishes_mean_effect_from_continuation_noise():
    rows = rows_for([("a/0", [[0, 1, 1], [1, 0, 1]])])
    metrics = pair_metrics(rows, np.zeros(2), "null")
    assert metrics["episode_weighted_mean_effect_mse"] == 0
    assert metrics["episode_weighted_repeat_effect_mse"] == pytest.approx(2 / 3)
    assert metrics["informative_pairs"] == 0
    random = selection_metrics(rows, "random", 2)
    assert random["episode_mean_help"] == pytest.approx(1 / 6)
    assert random["episode_mean_harm"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("objective", ["relative_ridge", "pairwise_logistic"])
def test_ranker_learns_order_on_independent_states_without_nuisance_offset(objective):
    rng = np.random.default_rng(121)

    def sample(count):
        rows = []
        features = []
        for state in range(count):
            context = rng.normal(size=2) * 100
            for c in range(4):
                action = rng.uniform(-1, 1)
                features.append([action, *context])
                rows.append(
                    {
                        "state_id": f"{state}/0",
                        "episode_id": str(state),
                        "candidate_index": c,
                        "eligible": True,
                        "outcomes": [int(action > 0)],
                    }
                )
        return rows, np.asarray(features)

    train, x = sample(50)
    test, z = sample(30)
    head = fit_ranker(x, pairs(train), objective, "linear", 0.001)
    before = head["mean"].copy()
    metric = pair_metrics(test, utility(objective, head, z), objective)
    assert metric["informative_pair_order_accuracy"] > 0.95
    assert np.max(np.abs(head["coef"][1:])) < 1e-10
    utility(objective, head, z * 1e5)
    np.testing.assert_array_equal(before, head["mean"])


def test_fixed_nonlinear_features_learn_context_dependent_action_order():
    rng = np.random.default_rng(144)

    def sample(count):
        rows = []
        x = []
        for state in range(count):
            context = rng.choice([-1, 1])
            for c in range(4):
                action = rng.uniform(-1, 1)
                x.append([context, action])
                rows.append(
                    {
                        "state_id": f"{state}/0",
                        "episode_id": str(state),
                        "candidate_index": c,
                        "eligible": True,
                        "outcomes": [int(context * action > 0)],
                    }
                )
        return rows, np.asarray(x)

    train, x = sample(60)
    test, z = sample(30)
    head = fit_ranker(
        x, pairs(train), "relative_ridge", "linear_plus_fixed_tanh", 0.001
    )
    assert (
        pair_metrics(test, utility("relative_ridge", head, z), "relative_ridge")[
            "informative_pair_order_accuracy"
        ]
        > 0.9
    )


def test_old_test_row_is_rejected_before_feature_file_access(tmp_path, monkeypatch):
    (tmp_path / "rows.json").write_text(
        json.dumps([{"seed": 10, "split": "test", "success": 1}])
    )
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(args)
        raise AssertionError("Feature access must not occur")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="episode split"):
        development(
            tmp_path,
            {
                "train_seeds": [2, 3],
                "validation_seeds": [8, 9],
                "excluded_source_seeds": [10, 11],
            },
        )
    assert opened == []
