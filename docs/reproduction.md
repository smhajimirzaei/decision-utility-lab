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

## Simulator/model experiments

The research modules remain under `src/trust_imagination/`. Simulation uses Meta-World/MuJoCo; frozen policy and dynamics adapters use LeRobot/SmolVLA and JEPA-WMs. Exact upstream revisions, checkpoint hashes, cameras, preprocessing, and action conventions are recorded in the two model-manifest JSON files under `configs/`.

Collection was developed for Linux/CUDA with Python 3.12.11. The [recorded package inventory](runtime-packages.txt) gives the versions from the last successful source run, including Meta-World 3.1.1 and MuJoCo 3.3.0; the early policy manifest also records its original 3.0.0 reference environment. Model weights and upstream source must be obtained separately under their licenses. The package's `research` extra supplies simulator/analysis dependencies, not the complete upstream model stack.

Raw latent and full simulator-trace archives are not included. The public data supports the decision audit, first-hit reporting, and recorded demonstration. Reproducing encoder features, renderer replay, or model forecasts requires new inference and the pinned environment. A recorded source-policy video cannot replace a forked continuation.

## Provenance

`demo/inputs.json` retains source hashes and collection identities. Original byte-level verification records are under `demo/provenance/`. Public-release checks are in `demo/verification.json`, and `demo/release_manifest.json` records the final packaged hashes. Text uses LF line endings, and the presentation was formatted for release; the recorded input values and outcomes are unchanged. The original frozen experiment source and local archives remain separate from this checkout.
