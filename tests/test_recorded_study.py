"""Public records must preserve pairing, episode weights, and the full grid."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trust_imagination.recorded_study import (
    _aggregate,
    _outcome,
    audit,
    validate,
    METRICS,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recorded():
    return json.loads((ROOT / "docs/data/trajectory_study.json").read_text())


def test_paired_help_and_harm_do_not_cancel_out_of_the_report():
    state = {
        "outcomes": {
            "policy": [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0],
            ]
        }
    }
    result = _outcome(state, "policy", 4, 0)
    assert result["success"] == Fraction(2, 3)
    assert result["gain3"] == 0
    assert result["help3"] == result["harm3"] == Fraction(1, 3)
    assert result["regret"] == Fraction(1, 3)


def test_episodes_receive_equal_weight_despite_different_state_counts():
    high = {key: Fraction(1) for key in METRICS}
    low = {key: Fraction(0) for key in METRICS}
    rows = [({"episode_id": "a", "seed": 56}, high)]
    rows += [({"episode_id": "b", "seed": 57}, low)] * 3
    result = _aggregate(rows)
    assert result["success"] == 0.5
    assert result["episodes"] == 2 and result["states"] == 4


def test_complete_recording_preserves_the_negative_primary_result(recorded):
    result = audit(recorded)
    assert len(result["records"]) == 108 and len(result["controls"]) == 72
    assert result["advance_condition_met"] is False
    assert result["primary_contrasts"][0]["mean_gain_difference"] == pytest.approx(
        -1 / 108
    )
    control = next(
        r
        for r in result["controls"]
        if r["control"] == "3" and r["bank"] == "repair" and r["size"] == 8
    )
    assert control["metrics"]["success"] == pytest.approx(5 / 9)


@pytest.mark.parametrize(
    "mutation",
    [
        "reserved",
        "duplicate",
        "missing_model",
        "missing_choice",
        "bounds",
        "binary",
        "pairing",
        "episode",
    ],
)
def test_rejects_corrupted_recordings(recorded, mutation):
    data = deepcopy(recorded)
    if mutation == "reserved":
        data["states"][0]["seed"] = 32
    elif mutation == "duplicate":
        data["states"].append(data["states"][0])
    elif mutation == "missing_model":
        data["selections"].pop()
    elif mutation == "missing_choice":
        data["selections"][0]["choices"].pop()
    elif mutation == "bounds":
        data["selections"][0]["choices"][0]["choice"] = data["selections"][0]["size"]
    elif mutation == "binary":
        data["states"][0]["outcomes"]["policy"][0][7] = 2
    elif mutation == "pairing":
        data["states"][0]["outcomes"]["mixed"][0][0] ^= 1
    else:
        first = data["states"][0]
        other = next(s for s in data["states"] if s["task"] != first["task"])
        other["episode_id"] = first["episode_id"]
        other["state_id"] = f"{other['episode_id']}/{other['step']}"
    with pytest.raises(ValueError):
        validate(data)


def test_command_works_without_site_packages_and_refuses_overwrite(tmp_path):
    output = tmp_path / "replay"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/summarize_trajectory_study.py"),
        "--input",
        str(ROOT / "docs/data/trajectory_study.json"),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    assert (output / "summary.json").read_bytes() == (
        ROOT / "docs/data/trajectory_summary.json"
    ).read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
