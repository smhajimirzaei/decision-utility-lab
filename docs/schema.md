# `decision-audit/1`

The generic command reads a JSON document with four required top-level fields:

| Field | Meaning |
|---|---|
| `schema` | Exactly `decision-audit/1` |
| `episode_splits` | Episode ID → split name |
| `selectors` | Named scorers, each declaring `input_scope: deployment` |
| `states` | Decision-bank records |

Each state declares `cohort`, `dataset`, `split`, `episode_id`, `task`, `seed`, `step`, `bank`, `reference_index`, `repeat_ids`, and `remaining_horizon`.

Candidate/outcome fields:

| Field | Shape / contract |
|---|---|
| `candidates` | Logical candidates, each with `logical_id`, `physical_id`, and executed-action `action_sha256` |
| `outcomes` | Binary repeat × candidate matrix |
| `first_hit` | Same shape; positive relative action count for success, `null` for failure/censoring |
| `forecast_supported` | One boolean per candidate |
| `scores` | Selector name → finite score vector aligned with candidates |

Candidate zero must be the reference. Repeat IDs must be unique and identical across matched banks. Shared physical or byte-identical actions must share outcomes and support. An episode cannot cross cohort, dataset, task/seed identity, or split. A cohort must have one declared split; pooled aggregates remain separate for training and test.

Control names `reference`, `uniform`, `offline_best`, and `fixed:*` are reserved. Actual-future targets cannot be declared as deployment scorers. Optional images, video paths, instructions, and input diagnostics serve the demo; the generic audit does not require them.

See [`examples/two_candidates.json`](../examples/two_candidates.json) for a synthetic two-choice fixture, or [`demo/inputs.json`](../demo/inputs.json) for the recorded study.

```bash
python -S scripts/audit_decisions.py --input examples/two_candidates.json --sizes 2 --output runs/toy
```
