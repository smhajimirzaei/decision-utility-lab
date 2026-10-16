# Frozen fresh trajectory study

This is the scientific protocol declared on October 13, 2026, before fresh collection. It compares paired benefit estimates from direct/action inputs, predicted endpoints, and full latent trajectories under proposal-construction and search-budget shifts. The initial tranche is a development diagnostic, not a final conference confirmation.

## Data split and intervention

| Role | Seeds | Source episodes |
|---|---|---:|
| Training | 47–52 | 18 |
| Calibration | 53–55 | 9 |
| Development evaluation | 56–58 | 9 |
| Technical timing pilot | 59 | Excluded |
| Reserved | 32–39 | Unopened |

Each seed spans reach-v3, push-v3, and door-open-v3. Use an absolute horizon of 400 steps and source decisions at 0, 50, 100, and 150 if reached. Execute a 50-action candidate prefix, followed by a 50-action VLA continuation, with three matched repeats. Stop physical traces at first success or termination. Preserve failures, ties, early successes, and incomplete technical records.

The 16-candidate physical union is deduplicated while retaining logical sampling weights. Policy uses indices 0–7; mixed uses [0, 1, 2, 3, 8, 9, 10, 11] at radius 0.1; repair uses [0, 1, 2, 3, 12, 13, 14, 15] at radius 0.25. All banks share their first four logical choices. Physical base-policy replay uses candidate 0. The primary comparative reference is logical index 3, selected using previously inspected development evidence; candidate 0 is a secondary reference.

Targets are matched-repeat frequencies of helping and harming relative to 3. Select the largest predicted help-minus-harm, breaking utility ties within 1e-10 by lowest logical index, then abstain to 3 unless benefit exceeds the threshold. Reference 3 has forced zero help/harm. Retain every fixed-index control, uniform selection, candidate 0, and the measured offline ceiling. Report source-policy success separately from intervention utility.

## Features and capacity

Every model uses current pooled visual features, encoded and raw proprioception, task identity, remaining horizon, candidate/reference actions, and action differences. The frozen spatial-channel projection is 384 × 16. Endpoint models add the final predicted visual/proprioceptive features and reference differences. Trajectory models add all ten predicted time points in chronological order, spaced five steps apart. Plain direct, endpoint, and trajectory feature dimensions are 640, 736, and 1600.

The metadata variants add five columns: effective bank radius, log candidate count, candidate/reference action RMS, bank mean RMS, and action-bound fraction. Radius is zero for the shared K4 and policy bank. No physical future, outcome flag, goal coordinate, support mask, or first-hit time enters deployment features. Predictions beyond a hypothetical early success remain forecasts, not physical endpoints.

All six specifications use train-only normalization with floor 0.05 and clipping at 10, a deterministic cosine map of width 128 and seed 701, ridge penalty 0.1, and two help/harm outputs. Each fits 256 coefficients plus two intercepts. Clip probabilities to [0, 1], then divide by max(1, help + harm). No hyperparameter sweep or training early stopping. Report the differing input dimensions and feature-access costs.

## Fitting, calibration, and evaluation

Fit and calibrate only on policy/mixed configurations at K4/K8. Weight seeds equally, tasks/episodes equally within seed, states equally within episode, configurations/budgets equally, and logical candidates equally. Repair construction and K6 are held out; K6 uses saved logical candidates, without new simulation. Report all six specifications across all three banks and K4/K6/K8. Shared K4 banks are controls, not independent populations.

Trajectory + metadata is the predeclared primary model; direct + metadata and endpoint + metadata are matched primary ablations. Fit on seeds 47–52 only. Calibrate on 53–55 using selections from the frozen fitted models. The threshold grid is [0, 0.025, 0.05, 0.1, 0.2, 0.4, 1.01]. Choose the lowest threshold with empirical paired mean harm at most 0.05 and positive gain over 3 on calibration policy/mixed configurations. If none qualifies, use 1.01 and always abstain. Retain every threshold curve, including rejected thresholds.

Freeze models and thresholds before opening evaluation outcomes from 56–58. Primary contrasts are trajectory + metadata against direct + metadata and endpoint + metadata on repair K8, using paired gain/help/harm and intervention coverage. Show construction transfer and budget transfer separately, plus per-seed/task decisions and latency. Use episode/seed cluster uncertainty rather than branch-level standard errors.

Three calibration seed groups do not establish a formal 5% risk guarantee. The all-reference rehearsal is separately exploratory and cannot replace the predeclared fresh method after observing labels.

## Progression rules

Require a complete, verified training/calibration grid before fitting. Keep partial records and report missing cells; do not silently replace seeds. If fewer than three training seed groups contain helpful alternatives relative to 3, label the comparative fit data-limited and retain the planned ablations.

If primary trajectory gain over direct inputs is nonpositive on repair K8 evaluation, retain the negative result and stop short of prospective controller claims. A positive result requires a separately declared sample-size and controller protocol, per-group consistency checks, and an adequate independent confirmation before opening reserved seeds. No extra scientific tranche or final reserved collection is part of this initial protocol.

The study completed on October 16, 2026. The primary method failed its advancement criterion; see the [complete results](trajectory-results.md). The protocol above preserves the design declared before collection.
