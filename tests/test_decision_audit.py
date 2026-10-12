"""Audit guards and weighting under informative, noisy matched interventions."""

from copy import deepcopy
import pytest
from trust_imagination.decision_audit import (
    SCHEMA,
    validate,
    audit_state,
    aggregate,
    audit,
)


def fixture():
    s = {
        "cohort": "development",
        "dataset": "main",
        "split": "development",
        "episode_id": "episode1",
        "task": "toy",
        "seed": 1,
        "step": 0,
        "bank": "policy",
        "reference_index": 0,
        "repeat_ids": [10, 20, 30],
        "remaining_horizon": 20,
        "outcomes": [[1, 0], [0, 1], [0, 1]],
        "first_hit": [[1, None], [None, 2], [None, 3]],
        "forecast_supported": [False, True],
        "candidates": [
            {"logical_id": i, "physical_id": i, "action_sha256": str(i) * 64}
            for i in range(2)
        ],
        "scores": {"head": [0.2, 0.8]},
    }
    return {
        "schema": SCHEMA,
        "episode_splits": {"episode1": "development"},
        "selectors": {"head": {"input_scope": "deployment"}},
        "states": [s],
    }


def test_matched_help_harm_and_regret_with_noisy_repeats():
    d = fixture()
    validate(d)
    r = audit_state(d["states"][0], 2, "head")
    assert r["gain"] == pytest.approx(1 / 3)
    assert r["help"] == pytest.approx(2 / 3) and r["harm"] == pytest.approx(1 / 3)
    assert r["regret"] == 0 and r["opportunity"] == pytest.approx(1 / 3)
    assert r["selected_first_hit"] == [None, 2, 3]
    uniform = audit_state(d["states"][0], 2, "uniform")
    assert uniform["success"] == 0.5 and uniform["switch_rate"] == 0.5


@pytest.mark.parametrize(
    "mutation",
    ["stream", "reference", "physical", "split", "offline", "timing", "nonfinite"],
)
def test_rejects_misaligned_or_leaking_inputs(mutation):
    d = fixture()
    s = d["states"][0]
    t = deepcopy(s)
    t["bank"] = "mixed"
    d["states"].append(t)
    if mutation == "stream":
        t["repeat_ids"][0] = 11
    elif mutation == "reference":
        t["candidates"][0]["action_sha256"] = "a" * 64
    elif mutation == "physical":
        t["outcomes"][0][1] = 1
        t["first_hit"][0][1] = 4
    elif mutation == "split":
        t["split"] = "test"
    elif mutation == "offline":
        d["selectors"]["head"]["input_scope"] = "actual_future"
    elif mutation == "timing":
        t["first_hit"][0][1] = 1
    elif mutation == "nonfinite":
        t["scores"]["head"][1] = float("nan")
    with pytest.raises(ValueError):
        validate(d)


def test_episode_weighting_and_logical_duplicate_weight_are_preserved():
    rows = [
        dict(
            episode_id="a",
            **{
                k: 1.0
                for k in (
                    "success",
                    "base",
                    "gain",
                    "help",
                    "harm",
                    "regret",
                    "opportunity",
                    "switch_rate",
                )
            },
        )
        for _ in range(3)
    ]
    rows.append(
        dict(
            episode_id="b",
            **{
                k: 0.0
                for k in (
                    "success",
                    "base",
                    "gain",
                    "help",
                    "harm",
                    "regret",
                    "opportunity",
                    "switch_rate",
                )
            },
        )
    )
    assert aggregate(rows)["success"] == 0.5
    d = fixture()
    s = d["states"][0]
    s["candidates"].append({**s["candidates"][1], "logical_id": 2})
    for matrix in ("outcomes", "first_hit"):
        for row in s[matrix]:
            row.append(row[1])
    s["scores"]["head"].append(0.8)
    s["forecast_supported"].append(True)
    validate(d)
    r = audit_state(s, 3, "uniform")
    assert r["success"] == pytest.approx(5 / 9)
    assert audit_state(s, 3, "head")["chosen"] == 1


def test_zero_opportunity_can_still_have_harm_and_selection_never_uses_repeat_view():
    d = fixture()
    s = d["states"][0]
    s["outcomes"] = [[1, 0]] * 3
    s["first_hit"] = [[1, None]] * 3
    r = audit(d, sizes=(2,))
    head = next(v for v in r["decisions"] if v["selector"] == "head")
    assert head["opportunity"] == 0 and head["gain"] == -1 and head["regret"] == 1
    assert head["class_name"] == "harmful" and head["chosen"] == 1


def test_external_training_and_test_are_not_pooled():
    d = fixture()
    first = d["states"][0]
    first["split"] = "training"
    first["cohort"] = "train"
    d["episode_splits"]["episode1"] = "training"
    second = deepcopy(first)
    second.update(episode_id="episode2", cohort="heldout", split="test", seed=2)
    d["states"].append(second)
    d["episode_splits"]["episode2"] = "test"
    result = audit(d, sizes=(2,))
    groups = {s["cohort"] for s in result["summaries"]}
    assert (
        "all_training" in groups
        and "all_test" in groups
        and "all_development" not in groups
    )
    second["cohort"] = "train"
    with pytest.raises(ValueError, match="multiple episode splits"):
        validate(d)


def test_reserved_selector_names_cannot_replace_controls():
    d = fixture()
    d["selectors"]["offline_best"] = {"input_scope": "deployment"}
    with pytest.raises(ValueError, match="collides"):
        validate(d)
