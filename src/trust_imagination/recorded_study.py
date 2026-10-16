"""Recompute intervention utility from recorded choices and paired outcomes.

This replays the published trajectory study's arithmetic. It does not fit a
model, regenerate forecasts, or treat candidate branches as independent episodes.
"""

from collections import defaultdict
from fractions import Fraction
from itertools import product
import math

SCHEMA = "trajectory-study/1"
BANKS = ("policy", "mixed", "repair")
SIZES = (4, 6, 8)
MODES = ("ungated", "calibrated")
METRICS = (
    "success",
    "reference3",
    "gain3",
    "help3",
    "harm3",
    "intervention",
    "reference0",
    "gain0",
    "help0",
    "harm0",
    "regret",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mean(values):
    values = list(values)
    return sum(values, Fraction()) / len(values)


def validate(data):
    _require(data.get("schema") == SCHEMA, "Unknown recorded-study schema")
    study = data["study"]
    seeds = study["development_seed_groups"]
    _require(seeds == [56, 57, 58], "Development seed groups differ from the protocol")
    _require(
        len(study["tasks"]) == 3
        and set(study["tasks"]) == {"reach-v3", "push-v3", "door-open-v3"},
        "Task grid differs",
    )
    _require(study["reference_index"] == 3, "Expected the frozen reference index3")
    names = [m["name"] for m in data["models"]]
    expected_names = {
        f"{kind}__{meta}"
        for kind, meta in product(
            ("direct", "endpoint", "trajectory"), ("plain", "metadata")
        )
    }
    _require(
        len(names) == 6 and set(names) == expected_names, "Incomplete readout grid"
    )
    for model in data["models"]:
        _require(
            type(model["threshold"]) in (int, float)
            and math.isfinite(model["threshold"])
            and model["threshold"] >= 0,
            "Invalid frozen threshold",
        )
    states = {}
    episodes = set()
    episode_identities = {}
    for state in data["states"]:
        identity = state["state_id"]
        _require(identity not in states, "Duplicate source state")
        _require(
            state["seed"] in seeds and state["task"] in study["tasks"],
            "Non-development source state",
        )
        _require(
            type(state["step"]) is int and state["step"] in (0, 50, 100, 150),
            "Unexpected decision step",
        )
        episode_identity = (state["task"], state["seed"])
        _require(
            episode_identities.get(state["episode_id"], episode_identity)
            == episode_identity,
            "Episode identity reused across tasks or seeds",
        )
        episode_identities[state["episode_id"]] = episode_identity
        _require(
            identity == f"{state['episode_id']}/{state['step']}",
            "State identity differs",
        )
        _require(set(state["outcomes"]) == set(BANKS), "Missing candidate bank")
        for outcomes in state["outcomes"].values():
            _require(
                len(outcomes) == 3 and all(len(row) == 8 for row in outcomes),
                "Expected three matched repeats and eight logical candidates",
            )
            _require(
                all(type(v) is int and v in (0, 1) for row in outcomes for v in row),
                "Outcomes must be binary",
            )
        for bank in BANKS[1:]:
            _require(
                [r[:4] for r in state["outcomes"][bank]]
                == [r[:4] for r in state["outcomes"]["policy"]],
                "Shared first-four outcomes are not matched",
            )
        states[identity] = state
        episodes.add((state["task"], state["seed"], state["episode_id"]))
    _require(
        len(states) == study["development_states"], "Source-state inventory differs"
    )
    _require(
        len(episodes) == 9
        and {(t, s) for t, s, _ in episodes} == set(product(study["tasks"], seeds)),
        "Incomplete development episode grid",
    )
    configurations = set()
    for selection in data["selections"]:
        key = tuple(selection[k] for k in ("spec", "bank", "size", "mode"))
        _require(key not in configurations, "Duplicate selection configuration")
        configurations.add(key)
        _require(selection["size"] in SIZES, "Unexpected candidate budget")
        choices = selection["choices"]
        _require(
            len(choices) == len(states)
            and {c["state_id"] for c in choices} == set(states),
            "Incomplete selection grid",
        )
        _require(
            all(
                type(c["choice"]) is int and 0 <= c["choice"] < selection["size"]
                for c in choices
            ),
            "Selected candidate outside its bank",
        )
    _require(
        configurations == set(product(names, BANKS, SIZES, MODES)),
        "Incomplete model/bank/budget/mode grid",
    )
    return states


def _outcome(state, bank, size, choice):
    repeats = state["outcomes"][bank]
    frequencies = [_mean(row[i] for row in repeats) for i in range(size)]
    chosen = range(size) if choice is None else [choice]
    success = _mean(frequencies[i] for i in chosen)
    result = {
        "success": success,
        "regret": max(frequencies) - success,
        "intervention": (
            Fraction(size - 1, size) if choice is None else Fraction(choice != 3)
        ),
    }
    for reference in (0, 3):
        deltas = [row[i] - row[reference] for i in chosen for row in repeats]
        result.update(
            {
                f"reference{reference}": frequencies[reference],
                f"gain{reference}": success - frequencies[reference],
                f"help{reference}": _mean(d > 0 for d in deltas),
                f"harm{reference}": _mean(d < 0 for d in deltas),
            }
        )
    return result


def _aggregate(rows):
    episodes = defaultdict(list)
    groups = defaultdict(lambda: defaultdict(list))
    for state, values in rows:
        episodes[state["episode_id"]].append(values)
        groups[state["seed"]][state["episode_id"]].append(values)

    def average(collection):
        return {
            key: float(
                _mean(_mean(row[key] for row in group) for group in collection.values())
            )
            for key in METRICS
        }

    return {
        **average(episodes),
        "episodes": len(episodes),
        "states": len(rows),
        "by_seed": {
            str(seed): average(group) for seed, group in sorted(groups.items())
        },
    }


def audit(data):
    states = validate(data)
    models = {m["name"]: m for m in data["models"]}
    records, controls = [], []
    for selection in data["selections"]:
        spec, bank, size, mode = (
            selection[k] for k in ("spec", "bank", "size", "mode")
        )
        rows = [
            (
                states[c["state_id"]],
                _outcome(states[c["state_id"]], bank, size, c["choice"]),
            )
            for c in selection["choices"]
        ]
        records.append(
            {
                "spec": spec,
                "bank": bank,
                "size": size,
                "mode": mode,
                "threshold": 0 if mode == "ungated" else models[spec]["threshold"],
                "metrics": _aggregate(rows),
            }
        )
    for bank, size in product(BANKS, SIZES):
        for control in [*range(size), "uniform", "offline_best"]:
            rows = []
            for state in states.values():
                choice = None if control == "uniform" else control
                if control == "offline_best":
                    frequencies = [
                        _mean(row[i] for row in state["outcomes"][bank])
                        for i in range(size)
                    ]
                    choice = max(range(size), key=lambda i: frequencies[i])
                rows.append((state, _outcome(state, bank, size, choice)))
            controls.append(
                {
                    "control": str(control),
                    "bank": bank,
                    "size": size,
                    "metrics": _aggregate(rows),
                }
            )
    primary = {
        r["spec"]: r["metrics"]
        for r in records
        if r["bank"] == "repair" and r["size"] == 8 and r["mode"] == "calibrated"
    }
    contrasts = []
    for baseline in ("direct__metadata", "endpoint__metadata"):
        differences = {
            str(seed): primary["trajectory__metadata"]["by_seed"][str(seed)]["gain3"]
            - primary[baseline]["by_seed"][str(seed)]["gain3"]
            for seed in (56, 57, 58)
        }
        contrasts.append(
            {
                "baseline": baseline,
                "mean_gain_difference": math.fsum(differences.values()) / 3,
                "seed_deltas": differences,
            }
        )
    return {
        "schema": "trajectory-summary/1",
        "study": data["study"],
        "records": records,
        "controls": controls,
        "primary_contrasts": contrasts,
        "advance_condition_met": contrasts[0]["mean_gain_difference"] > 0,
    }


def markdown(report):
    rows = [
        (r["spec"], r["metrics"])
        for r in report["records"]
        if r["bank"] == "repair" and r["size"] == 8 and r["mode"] == "calibrated"
    ]
    rows += [
        ("control:" + r["control"], r["metrics"])
        for r in report["controls"]
        if r["bank"] == "repair" and r["size"] == 8
    ]
    lines = [
        "# Recorded trajectory study",
        "",
        "Repair bank, eight candidates. Rates weight source episodes equally.",
        "Nine development episodes in three seed groups; these are single-intervention outcomes, not controller success.",
        "",
        "| Readout / control | Success % | Gain vs reference (pp) | Help % | Harm % | Switch % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in rows:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{100 * values[k]:.2f}"
                for k in ("success", "gain3", "help3", "harm3", "intervention")
            )
            + " |"
        )
    lines += [
        "",
        f"Predeclared advancement condition met: {report['advance_condition_met']}.",
        "Training is data-limited. The secondary direct result is concentrated in seed58.",
        "The offline best uses observed outcomes and is an optimistic control, not a deployment method.",
    ]
    return "\n".join(lines) + "\n"
