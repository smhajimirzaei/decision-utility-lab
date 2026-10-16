"""Replay the completed study's recorded choices and paired outcomes; no models."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trust_imagination.recorded_study import METRICS, audit, markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output / "results.md").write_text(
        markdown(report), encoding="utf-8", newline="\n"
    )
    columns = ["spec", "control", "bank", "size", "mode", "threshold", *METRICS]
    for kind in ("records", "controls"):
        with (args.output / f"{kind}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in report[kind]:
                writer.writerow(
                    {
                        **{key: row.get(key) for key in columns if key not in METRICS},
                        **{key: row["metrics"][key] for key in METRICS},
                    }
                )
    print(
        f"Recomputed {len(report['records'])} readout rows and {len(report['controls'])} controls; advancement={report['advance_condition_met']}"
    )


if __name__ == "__main__":
    main()
