# Research direction and related work

The next question is whether a deployment-available estimate of benefit over a fixed base action stays calibrated when proposal construction or search budget changes. That requires useful alternatives in the first place. The current inspected population supplies too few helpful cases to establish a reliable learned gate.

A prospective study should hold out whole episodes and candidate configurations, retain harmful/tied outcomes, compare capacity-matched direct/action controls, and measure the eventual closed-loop controller at a fixed cadence. Calibration thresholds and model-selection rules must be fixed before confirmation data is opened. Cached one-intervention performance and controller episode success are separate outcomes.

Several close works constrain the contribution:

- [Pre-VLA](https://arxiv.org/abs/2605.22446) studies action verification and advantage-based selection. An advantage monitor is a relevant comparator.
- [When to Trust Imagination](https://arxiv.org/abs/2605.06222v3) uses future–reality verification for adaptive world-action-model execution. Generic verification/replanning is established work.
- [The Planning Limits of Latent World Models](https://arxiv.org/abs/2609.39235v1) studies planning range, rollout horizons, subgoals, and VLA proposal selection. Short-horizon failure alone is not a new contribution.
- [AD-WM](https://arxiv.org/abs/2609.30264v2) trains action-discriminative dynamics and reports planning-specific diagnostics. Action separation and prediction/control mismatch have direct precedent.

The [completed 36-episode study](trajectory-results.md) did not meet its predeclared advancement criterion: trajectory + metadata underperformed the matched direct control and the fixed reference on repair K8. A secondary action-only readout matched the measured offline ceiling, with all gain in one development group. Training contained only two helpful policy/mixed states, both on push.

This release is an evaluation toolkit and development case study. A method claim would need prospective transfer across candidate configurations and another model/policy pairing, with gains over strong direct controls. The current result supplies no evidence of a trajectory advantage in this fixed readout/data regime; it does not establish that world models are generally unhelpful. Novelty and conference readiness remain open.
