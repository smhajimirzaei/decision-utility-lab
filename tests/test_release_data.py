"""The public data reproduces the recorded report without raw model archives."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from trust_imagination.decision_audit import audit

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_scores_and_outcomes_reproduce_the_report():
    data = json.loads((ROOT / "demo/inputs.json").read_text())
    expected = json.loads((ROOT / "demo/report.json").read_text())
    assert audit(data) == expected
    assert {s["seed"] for s in data["states"]} == {27, 28, 30, 31, 41, 42}


def test_packaged_media_matches_the_recorded_hashes():
    proof = json.loads((ROOT / "demo/verification.json").read_text())
    for record in proof["assets"]:
        actual = hashlib.sha256(
            (ROOT / "demo" / record["path"]).read_bytes()
        ).hexdigest()
        assert actual == record["sha256"], record["path"]


def test_cli_runs_without_site_packages_and_preserves_existing_output(tmp_path):
    destination = tmp_path / "audit"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/audit_decisions.py"),
        "--input",
        str(ROOT / "examples/two_candidates.json"),
        "--sizes",
        "2",
        "--output",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    original = (destination / "report.json").read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert (
        result.returncode != 0
        and (destination / "report.json").read_bytes() == original
    )
