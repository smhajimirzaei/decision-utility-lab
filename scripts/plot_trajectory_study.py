"""Plot mean and per-seed readout utility from the published summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New filename prefix"
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    names = [
        "direct__metadata",
        "endpoint__metadata",
        "trajectory__metadata",
        "direct__plain",
    ]
    labels = [
        "Direct + metadata",
        "Endpoint + metadata",
        "Trajectory + metadata (primary)",
        "Direct, plain (secondary)",
    ]
    colors = ["#35657b", "#7e8992", "#bf5337", "#296d59"]
    rows = {
        r["spec"]: r["metrics"]
        for r in report["records"]
        if r["bank"] == "repair" and r["size"] == 8 and r["mode"] == "calibrated"
    }
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.hashsalt": "decision-utility-lab",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    for i, (name, color) in enumerate(zip(names, colors)):
        value = 100 * rows[name]["gain3"]
        left.barh(i, value, height=0.52, color=color)
        left.text(
            value + (0.18 if value >= 0 else -0.18),
            i,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
        )
        right.plot(
            [56, 57, 58],
            [100 * rows[name]["by_seed"][str(s)]["gain3"] for s in (56, 57, 58)],
            marker="o",
            markersize=5,
            color=color,
            label=labels[i],
            linewidth=1.5,
            linestyle="--" if name == "direct__plain" else "-",
        )
    left.set_yticks(range(4), labels)
    left.invert_yaxis()
    left.axvline(0, color="#343f46", linewidth=0.8)
    left.set_xlim(-7.4, 6.5)
    left.set_xlabel("Mean gain over reference (percentage points)")
    left.set_title("Episode-weighted mean", loc="left", pad=16, fontweight="bold")
    right.axhline(0, color="#343f46", linewidth=0.8)
    right.set_xticks([56, 57, 58])
    right.set_ylim(-19, 17)
    right.set_xlabel("Development seed group")
    right.set_ylabel("Gain over reference (percentage points)")
    right.set_title(
        "Three groups, shown individually", loc="left", pad=16, fontweight="bold"
    )
    right.grid(axis="y", color="#e7e9eb", linewidth=0.6)
    right.set_axisbelow(True)
    fig.suptitle(
        "Readout utility on held-out development episodes",
        x=0.03,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.03,
        "Repair bank, K=8. Nine episodes in three seed groups. Single-intervention outcomes; no controller claim.",
        fontsize=9,
        color="#505a61",
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.78, bottom=0.19, wspace=0.48)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        target = args.output.with_suffix("." + extension)
        if target.exists():
            raise FileExistsError(target)
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {"Date": None} if extension == "svg" else None
        )
        fig.savefig(target, dpi=180, metadata=metadata, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
