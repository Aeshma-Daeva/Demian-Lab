# Local-flow scope and curvature

Records: `DEMIAN_UNCERTAINTY_JACOBIAN_RESULTS.md` and immutable directories
`data/diagnostics/demian_uncertainty_horizon_v1_20261013/`,
`data/diagnostics/demian_uncertainty_curvature_v1_20261013/`.
Astra reviewed these as evaluable scope/model-check diagnostics. Registry
entries `R-UNC-HORIZON-001` and `R-UNC-CURVATURE-001` are registered;
the narrow `H-UNC-001` status is unchanged.

| Model | Question | Outcome |
| --- | --- | --- |
| Fixed-centroid Jacobian | How far does one local-linear account extend? | All-seed threshold holds at 2 steps; first sampled failure at 4; all cells fail by 8. |
| Directional second order | Does ordinary curvature explain that error? | Substantial median reduction, but seed/horizon-dependent; all-seed criterion still fails. |

Horizon scope uses the same four seeds and cached source states, not new
replications. Residuals use the fixed A-cloud scatter denominator. Identity and
centroid translation are worse throughout. Largest derivative-vs-native
rollout discrepancy is 4.32e-4; all 384 cached two-step anchors match exactly.

Curvature model-check: h4 median same/new residual drops 0.104/0.112 to 0.042/0.047;
h8 drops 0.237/0.261 to 0.096/0.104. Seed 994 exceeds 0.1 at h4 and worsens at h16.
Directional Hessian finite-difference maximum discrepancy 0.000191 meets the
declared 0.001 validation threshold. No higher-order search follows.

Interpretation: ordinary nonlinear curvature accounts for part of finite-window
forecast error; neither a global linear nor universally adequate quadratic
description is established. This is not chaos, information loss or memory use.

Descriptive channel audit: gate contributes the largest median share of raw
residual energy, with slow/carrier also substantial. State scaling prevents
interpreting this as causal localization or as evidence ownership.

Falsifier for local adequacy: precision-valid perturbations around independently
selected reachable states exceed the declared local response-error threshold.
Reachable-anchor radius gate completed: see `DEMIAN_UNCERTAINTY_RADIUS_RESULTS.md`.
Current gate: natural-history chords, not further Taylor-order or radius tuning.
