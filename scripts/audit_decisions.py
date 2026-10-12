"""Audit any decision-audit/1 JSON document; standard library only."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trust_imagination.decision_audit import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    a = p.parse_args()
    data = json.loads(a.input.read_text(encoding="utf-8-sig"))
    report = audit(data, sizes=a.sizes)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    columns = [
        "cohort",
        "dataset",
        "split",
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
    with (a.output / "decisions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(report["decisions"])
    print(json.dumps(report["inventory"]))
    print("No models loaded; no fitting; no CUDA dependency.")


if __name__ == "__main__":
    main()
