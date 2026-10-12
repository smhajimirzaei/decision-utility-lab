"""CPU-only, model-independent audit of matched candidate decisions.

Outcomes are offline audit instrumentation. This module never constructs a
deployment selector from outcome labels, fits a model or imports torch.
"""

from collections import defaultdict
import math
import re
from statistics import mean

SCHEMA = "decision-audit/1"
METRICS = (
    "success",
    "base",
    "gain",
    "help",
    "harm",
    "regret",
    "opportunity",
    "switch_rate",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(data):
    require(data.get("schema") == SCHEMA, "Unsupported decision-audit schema")
    require(bool(data.get("states")), "No decision states")
    seen = set()
    episodes = {}
    references = {}
    physical = {}
    cohort_splits = {}
    require(
        not any(
            name in ("reference", "uniform", "offline_best")
            or name.startswith("fixed:")
            for name in data["selectors"]
        ),
        "Selector collides with built-in control",
    )
    for s in data["states"]:
        key = (s["cohort"], s["dataset"], s["episode_id"], s["step"], s["bank"])
        require(key not in seen, "Duplicate decision bank")
        seen.add(key)
        episode = s["episode_id"]
        split = s["split"]
        require(
            data["episode_splits"].get(episode) == split,
            "Episode split declaration differs",
        )
        require(
            not s["cohort"].startswith("all_"),
            "Cohort name reserved for pooled reporting",
        )
        require(
            cohort_splits.setdefault(s["cohort"], split) == split,
            "Cohort contains multiple episode splits",
        )
        assignment = (split, s["cohort"], s["dataset"], s["task"], s["seed"])
        require(
            episodes.setdefault(episode, assignment) == assignment,
            "Episode crosses cohort/dataset/split",
        )
        n = len(s["candidates"])
        r = len(s["repeat_ids"])
        require(
            n >= 2 and r > 0 and len(set(s["repeat_ids"])) == r,
            "Invalid matched repeat IDs",
        )
        require(
            s.get("reference_index") == 0,
            "Candidate zero must be the declared reference",
        )
        y = s["outcomes"]
        t = s["first_hit"]
        support = s["forecast_supported"]
        require(
            len(y) == len(t) == r and len(support) == n,
            "Outcome/timing/support shape differs",
        )
        require(all(type(v) is bool for v in support), "Invalid forecast support")
        require(
            all(
                len(row) == n and all(type(v) is int and v in (0, 1) for v in row)
                for row in y
            ),
            "Outcomes must be a binary repeat-by-candidate matrix",
        )
        require(all(len(row) == n for row in t), "Timing matrix shape differs")
        require(
            len(set(c["logical_id"] for c in s["candidates"])) == n,
            "Duplicate logical candidate IDs",
        )
        for i, c in enumerate(s["candidates"]):
            require(
                bool(re.fullmatch("[a-f0-9]{64}", c["action_sha256"])),
                "Invalid action checksum",
            )
            for j in range(r):
                hit = t[j][i]
                require(
                    (hit is None and y[j][i] == 0)
                    or (
                        type(hit) is int
                        and 1 <= hit <= s["remaining_horizon"]
                        and y[j][i] == 1
                    ),
                    "Success and first-hit/censor status differ",
                )
            signature = (
                c["action_sha256"],
                tuple(row[i] for row in y),
                tuple(row[i] for row in t),
                support[i],
            )
            context = (s["cohort"], episode, s["step"], tuple(s["repeat_ids"]))
            identity = context + (c["physical_id"],)
            require(
                physical.setdefault(identity, signature) == signature,
                "Shared physical candidate differs between banks",
            )
            byte_identity = context + (c["action_sha256"],)
            require(
                physical.setdefault(byte_identity, signature) == signature,
                "Byte-identical candidate has different matched outcome/support",
            )
        refkey = (s["cohort"], episode, s["step"])
        ref = (
            tuple(s["repeat_ids"]),
            s["candidates"][0]["action_sha256"],
            tuple(row[0] for row in y),
            tuple(row[0] for row in t),
            support[0],
        )
        require(
            references.setdefault(refkey, ref) == ref,
            "Shared reference or continuation stream differs",
        )
        names = set(s["scores"])
        require(
            names == set(data["selectors"]),
            "Selector availability differs from declaration",
        )
        for name, scores in s["scores"].items():
            require(
                data["selectors"][name]["input_scope"] == "deployment",
                "Offline target/actual future cannot be a deployment selector",
            )
            require(
                len(scores) == n
                and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in scores
                ),
                "Invalid deployment scores",
            )
    return {
        "states": len(references),
        "bank_rows": len(seen),
        "episodes": len(episodes),
        "selectors": len(data["selectors"]),
    }


def choose(scores):
    top = max(scores)
    return next(i for i, v in enumerate(scores) if v >= top - 1e-10)


def _sequential_sum(values):
    """Preserve the recorded float arithmetic across Python versions.

    Python 3.12 changed built-in float summation. The frozen report used
    sequential addition, so keep that order explicit for exact reproduction.
    """
    total = 0.0
    for value in values:
        total += value
    return total


def audit_state(s, size, selector):
    require(2 <= size <= len(s["candidates"]), "Candidate budget outside bank")
    y = [row[:size] for row in s["outcomes"]]
    means = [mean(row[i] for row in y) for i in range(size)]
    base = means[0]
    best = max(means)
    scores = None
    if selector == "reference":
        chosen = 0
    elif selector == "uniform":
        chosen = None
    elif selector == "offline_best":
        chosen = means.index(best)
    elif selector.startswith("fixed:"):
        chosen = int(selector.split(":")[1])
        require(0 <= chosen < size, "Fixed choice outside bank")
    else:
        scores = s["scores"][selector][:size]
        chosen = choose(scores)
    weights = (
        [1 / size] * size
        if chosen is None
        else [float(i == chosen) for i in range(size)]
    )
    success = _sequential_sum(w * v for w, v in zip(weights, means))
    help = mean(
        _sequential_sum(weights[i] * int(row[i] > row[0]) for i in range(size))
        for row in y
    )
    harm = mean(
        _sequential_sum(weights[i] * int(row[i] < row[0]) for i in range(size))
        for row in y
    )
    opportunity = best - base
    regret = best - success
    result = {
        k: s[k]
        for k in (
            "cohort",
            "dataset",
            "split",
            "episode_id",
            "task",
            "seed",
            "step",
            "bank",
        )
    }
    result.update(
        size=size,
        selector=selector,
        chosen=chosen,
        success=success,
        base=base,
        gain=success - base,
        help=help,
        harm=harm,
        regret=regret,
        opportunity=opportunity,
        switch_rate=1 - weights[0],
        candidate_means=means,
        best_indices=[i for i, v in enumerate(means) if abs(v - best) <= 1e-12],
        class_name=(
            "helpful_and_harmful"
            if best > base and min(means) < base
            else (
                "helpful"
                if best > base
                else (
                    "harmful"
                    if min(means) < base
                    else (
                        "tied_success"
                        if base == 1
                        else "tied_failure" if base == 0 else "tied_partial"
                    )
                )
            )
        ),
        selected_first_hit=(
            None if chosen is None else [row[chosen] for row in s["first_hit"]]
        ),
        reference_first_hit=[row[0] for row in s["first_hit"]],
        selected_forecast_supported=(
            None if chosen is None else s["forecast_supported"][chosen]
        ),
        scores=scores,
        offline_control=selector == "offline_best",
    )
    require(
        abs(result["gain"] - (opportunity - regret)) < 1e-12,
        "Opportunity/regret identity failed",
    )
    require(
        abs(result["gain"] - (help - harm)) < 1e-12,
        "Paired benefit/harm identity failed",
    )
    return result


def aggregate(rows):
    require(bool(rows), "Empty aggregate population")
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(row)
    return {
        "episodes": len(groups),
        "states": len(rows),
        **{
            k: mean(mean(row[k] for row in values) for values in groups.values())
            for k in METRICS
        },
        "helpful_states": sum(row["opportunity"] > 1e-12 for row in rows),
        "weighting": "repeats within candidate; decisions within episode; episodes equally",
    }


def audit(data, sizes=(4, 8)):
    inventory = validate(data)
    rows = []
    for s in data["states"]:
        for size in sizes:
            for name in [
                "reference",
                "uniform",
                "offline_best",
                *[f"fixed:{i}" for i in range(size)],
                *data["selectors"],
            ]:
                rows.append(audit_state(s, size, name))
    grouped = defaultdict(list)
    for row in rows:
        key = (row["cohort"], row["dataset"], row["bank"], row["size"], row["selector"])
        grouped[key].append(row)
        grouped[
            (
                "all_" + row["split"],
                row["dataset"],
                row["bank"],
                row["size"],
                row["selector"],
            )
        ].append(row)
    summary = []
    for key, values in grouped.items():
        cohort, dataset, bank, size, selector = key
        summary.append(
            dict(
                cohort=cohort,
                dataset=dataset,
                bank=bank,
                size=size,
                selector=selector,
                **aggregate(values),
            )
        )
    return {
        "schema": SCHEMA,
        "inventory": inventory,
        "fits": 0,
        "selected_model": None,
        "scope": (
            "Descriptive inspected development data; offline best is optimistic; no controller/final-test claim"
            if all(s["split"] == "development" for s in data["states"])
            else "Descriptive matched-intervention audit, pooled within split; offline best is optimistic; no controller/statistical-confirmation claim"
        ),
        "summaries": summary,
        "decisions": rows,
    }
