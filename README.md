# Decision Utility Lab

Tools and experiments for checking whether world-model forecasts improve a robot's action choices.

[![CPU tests](https://github.com/smhajimirzaei/decision-utility-lab/actions/workflows/test.yml/badge.svg)](https://github.com/smhajimirzaei/decision-utility-lab/actions/workflows/test.yml)

[Latest study](docs/trajectory-results.md) · [Recorded demo](demo/) · [Methods](docs/methodology.md) · [Reproduction](docs/reproduction.md)

A selector can make mistakes because it misreads the future, or because none of its candidate actions would help. This project separates those cases using matched simulator continuations. A frozen SmolVLA policy supplies actions; a latent world model supplies forecasts. Direct/action controls test whether those forecasts add anything useful.

## Latest result

The completed trajectory study contains **36 source episodes** across reach, push, and door opening: 18 for training, nine for calibration, and nine for development evaluation. Models and thresholds were frozen before development scoring.

| Repair bank, eight candidates | Intervention success |
|---|---:|
| Fixed reference action | 55.56% |
| Direct/action + metadata | 52.78% |
| Endpoint forecast + metadata | 50.00% |
| **Trajectory forecast + metadata — predeclared primary** | **51.85%** |
| Direct/action without metadata — secondary ablation | 60.19% |
| Best measured offline candidate — optimistic control | 60.19% |

**The primary method did not meet the advancement criterion.** Its gain was 0.93 percentage points below the matched direct control and 3.70 points below the fixed reference. The secondary action-only result is interesting, but all its improvement came from one development seed group.

![Readout gains and their variation across development seed groups](docs/figures/trajectory_study.png)

These are single-intervention outcomes on nine development episodes in three seed groups, not full-controller success rates. Training was data-limited: only two of 54 reached training states had helpful alternatives to the reference. The study does not establish a reliable intervention gate, a formal harm guarantee, or a general limitation of world models.

The [results report](docs/trajectory-results.md) retains all six readouts, fixed-action controls, per-seed differences, and the stopping decision.

## Reproduce the numbers

Python 3.10 or later is enough. The recorded-study auditor uses only the standard library:

```bash
git clone https://github.com/smhajimirzaei/decision-utility-lab.git
cd decision-utility-lab
python -S scripts/summarize_trajectory_study.py --input docs/data/trajectory_study.json --output runs/trajectory
```

This recomputes 108 readout rows, 72 controls, and the primary contrasts from saved choices and paired binary outcomes. It writes JSON, CSV, and a Markdown report. Use a new output directory for each run. It does not rerun model fitting, forecast inference, or simulation.

For the CPU tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See [reproduction](docs/reproduction.md) for figure generation, the earlier audit, and the separate simulator/model environment.

## Recorded demo

![Recorded action-selection audit](demo/desktop.png)

Open [`demo/index.html`](demo/index.html) locally, keeping `demo/assets/` beside it. Download or clone the repo first; GitHub's file preview shows HTML source. No server, GPU, or model download is required.

The demo shows an **earlier 18-episode audit**, with native observations, source-policy videos, 42 frozen scorers, and helpful, harmful, and tied cases. It is a separate cohort from the completed trajectory study above. The future model predicts latent features; the videos are recorded policy execution, not generated futures.

```bash
python -S scripts/audit_decisions.py --input demo/inputs.json --output runs/demo-audit
```

## Repository guide

| Path | Purpose |
|---|---|
| [`decision_audit.py`](src/trust_imagination/decision_audit.py) | General audit of candidate scores and matched outcomes |
| [`recorded_study.py`](src/trust_imagination/recorded_study.py) | Exact episode-weighted replay of the completed trajectory study |
| [`trajectory_benefit.py`](src/trust_imagination/trajectory_benefit.py) | Direct, endpoint, and chronological forecast features; help/harm heads |
| [`configs/`](configs/) | Experiment settings and pinned upstream model manifests |
| [`docs/data/`](docs/data/) | Recorded choices, paired outcomes, and complete tables |
| [`tests/`](tests/) | CPU checks for contracts, weighting, leakage, and readouts |
| [`demo/`](demo/) | Portable recorded audit and its media |

The [original audit](docs/results.md), [repair experiments](docs/comparative-readouts.md), and [frozen trajectory protocol](docs/trajectory-study.md) document how the question developed. Related work and the contribution's limits are discussed in [research notes](docs/research.md).

## Dependencies and scope

The experiments use [Meta-World](https://github.com/Farama-Foundation/Metaworld), [SmolVLA / LeRobot](https://github.com/huggingface/lerobot), and [JEPA-WMs](https://github.com/facebookresearch/jepa-wms). Full collection requires separate upstream downloads and a compatible research environment. Foundation-model weights, raw latent archives, and cloud configuration are not included.

A project license has not been assigned. Third-party software and assets retain their upstream terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
