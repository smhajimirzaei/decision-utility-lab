# Recorded trajectory study data

`docs/data/trajectory_study.json` uses schema `trajectory-study/1`. It contains development choices and matched binary outcomes, without encoder inputs, fitted weights, raw traces, or cloud metadata.

| Field | Meaning |
|---|---|
| `study` | Frozen splits, task grid, reference index, counts, and training-coverage flag |
| `models` | Six readout specifications and their frozen thresholds and parameter budgets |
| `states` | Unique source states, with task, seed, source step, and outcomes |
| `selections` | Recorded model choices for every bank/budget/mode combination |
| `provenance` | Original contract/model/evaluation hashes and recording scope |

Each state's `outcomes[bank]` has shape **three repeats × eight logical candidates**. A value is 1 for observed success and 0 otherwise. Repeat positions are paired within a state across candidates and banks. The shared first four candidates have identical outcomes across banks. Logical choices retain their weights even when physical candidates were deduplicated during collection.

A state ID is `episode_id/step`; it is an identifier, not a filesystem path. Each selection row has a model `spec`, `bank`, `size` in {4, 6, 8}, `mode` in {ungated, calibrated}, and one recorded choice per source state. Reference 3 is available in every bank. Both modes remain separate even when a zero threshold makes their choices identical.

The auditor checks the nine-episode grid, unique identities, binary outcome shapes, shared-four matching, all 108 selection configurations, and candidate bounds. It computes success, paired help/harm, switching, and regret, averaging states within episode and episodes equally. Per-seed values use the same weighting.

Uniform selection averages all logical choices exactly. Offline-best selects the largest observed repeat mean, breaking ties by lowest index. It uses outcomes for both selection and evaluation and is optimistic.

The returned `trajectory-summary/1` report contains 108 readout rows, 72 controls, two primary contrasts, and the advancement decision. JSON/CSV rates are fractions; Markdown displays percentages and percentage-point differences.

These files replay selection arithmetic. They do not independently verify neural predictions, renderer behavior, or physical traces; those were checked against retained research exports before publication. The earlier recorded demo uses the separate [decision-audit schema](schema.md).
