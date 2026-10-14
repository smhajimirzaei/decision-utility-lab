# Reproduction

## Cached decision audit

Python 3.10+ is sufficient for the generic audit. It loads the public cached scores and matched outcomes, without initializing a model:

```bash
python -S scripts/audit_decisions.py --input demo/inputs.json --output runs/reproduction
```

The output report should match `demo/report.json` exactly. Weighted candidate metrics use explicit sequential float addition to preserve the recorded arithmetic across Python versions. This avoids the last-bit differences introduced by Python 3.12's change to built-in `sum`.

The recorded media sits in `demo/assets/`. Rebuilding the portable presentation uses the same inputs:

```bash
python scripts/build_demo.py --input demo/inputs.json --output runs/demo
```

The builder copies the referenced media and writes a fresh report, CSV, and HTML. It refuses existing output directories. It does not run simulation or inference.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The default suite covers CPU contracts, proposal construction, matched repeat metrics, synthetic trace auditing, and readouts. GPU/native-renderer checks require the separate research environment. CI does not download pretrained weights or execute robot control.

The comparative feature and query tests can also be run on their own:

```bash
python -m pytest -q tests/test_trajectory_benefit.py tests/test_pairwise_benefit_queries.py
```

These tests check paired help/harm labels, reference identity, probability normalization, shared candidate geometry, outcome leakage, and sensitivity to earlier forecast frames. The public feature/query implementations preserve the research source semantics; formatting changes do not change their Python syntax trees.

## Comparative readout records

[`docs/data/fixed_reference_readouts.csv`](data/fixed_reference_readouts.csv) and [`docs/data/all_reference_readouts.csv`](data/all_reference_readouts.csv) retain all 432 aggregate records from each historical CPU round. Rates are fractions; `gain3`, `help3`, and `harm3` compare to fixed logical index 3. Each row identifies its cohort, readout, bank, budget, and threshold. `selected_brier` is the selected candidate's three-class help/harm/tie Brier score, averaged over matched repeats, then states within episode, then episodes. These are descriptive results on inspected cohorts.

[`repair_frozen_heads.csv`](data/repair_frozen_heads.csv) includes all 42 unchanged heads for main and timing-pilot groups at K4/K8 across three banks. Its `gain`, `help`, and `harm` columns use reference 0; the separately named `*_vs_fixed3` columns use reference 3. Keep pilot rows separate from main results.

The [comparison figure](figures/comparative_readouts.png) is available as PNG, [PDF](figures/comparative_readouts.pdf), and [SVG](figures/comparative_readouts.svg). Its plotted model values were checked against the two readout CSVs before publication. The public tables support inspection of recorded aggregates; they do not supply the raw encoder inputs required to reproduce model fitting. The dated collection, fitting declarations, cloud orchestration, and raw exports remain in the private research workspace. The public release provides reusable feature/readout modules, synthetic tests, and the declared scientific method.

## Simulator/model experiments

The research modules remain under `src/trust_imagination/`. Simulation uses Meta-World/MuJoCo; frozen policy and dynamics adapters use LeRobot/SmolVLA and JEPA-WMs. Exact upstream revisions, checkpoint hashes, cameras, preprocessing, and action conventions are recorded in the two model-manifest JSON files under `configs/`.

Collection was developed for Linux/CUDA with Python 3.12.11. The [recorded package inventory](runtime-packages.txt) gives the versions from the last successful source run, including Meta-World 3.1.1 and MuJoCo 3.3.0; the early policy manifest also records its original 3.0.0 reference environment. Model weights and upstream source must be obtained separately under their licenses. The package's `research` extra supplies simulator/analysis dependencies, not the complete upstream model stack.

Raw latent and full simulator-trace archives are not included. The public data supports the decision audit, first-hit reporting, and recorded demonstration. Reproducing encoder features, renderer replay, or model forecasts requires new inference and the pinned environment. A recorded source-policy video cannot replace a forked continuation.

## Provenance

`demo/inputs.json` retains source hashes and collection identities. Original byte-level verification records are under `demo/provenance/`. Public-release checks are in `demo/verification.json`, and `demo/release_manifest.json` records the final packaged hashes. Text uses LF line endings, and the presentation was formatted for release; the recorded input values and outcomes are unchanged. The original frozen experiment source and local archives remain separate from this checkout.
