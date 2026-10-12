"""Build the recorded HTML demo from a decision-audit document and its media."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trust_imagination.decision_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    report = audit(data)
    if report["inventory"] != {
        "states": 54,
        "bank_rows": 108,
        "episodes": 18,
        "selectors": 42,
    }:
        raise ValueError(
            "This presentation template is for the packaged study; use audit_decisions.py for other documents"
        )
    media = sorted(
        {
            value
            for state in data["states"]
            for value in [*state["images"].values(), state["video"]]
        }
    )
    base = args.input.parent.resolve()
    for name in media:
        source = (base / name).resolve()
        source.relative_to(base)
        if not source.is_file():
            raise FileNotFoundError(source)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in media:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
    for name, value in [("inputs.json", data), ("report.json", report)]:
        (args.output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    fields = [
        "cohort",
        "episode_id",
        "task",
        "seed",
        "step",
        "bank",
        "size",
        "selector",
        "chosen",
        "success",
        "base",
        "gain",
        "help",
        "harm",
        "regret",
        "opportunity",
        "switch_rate",
        "class_name",
    ]
    with (args.output / "decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(report["decisions"])
    payload = json.dumps(
        {"inputs": data, "summaries": report["summaries"]},
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    template = (ROOT / "scripts/demo_template.html").read_text(encoding="utf-8")
    if template.count("__AUDIT_DATA__") != 1:
        raise ValueError("Demo template marker missing or duplicated")
    (args.output / "index.html").write_text(
        template.replace("__AUDIT_DATA__", payload), encoding="utf-8", newline="\n"
    )
    manifest = {
        "status": "built_from_cached_inputs",
        "source_inputs_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "files": [
            {
                "path": p.relative_to(args.output).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        ],
    }
    (args.output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Demo built: {args.output}")


if __name__ == "__main__":
    main()
