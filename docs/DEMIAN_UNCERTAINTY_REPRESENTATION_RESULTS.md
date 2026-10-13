# Uncertainty Representation Probe Results

Status: superseded as unevaluable, 2026-10-13. This is not retained-evidence,
uncertainty, causal, routing, utility, learning, or adaptation evidence.

## Observation

At the final pre-resolution delay frame, per-model-seed frozen ridge probes
decoded public-evidence log odds from full internal state on held-out complete
public histories.

| delay | full state mean held-out R² | positive model seeds | current cue R² | shuffled full-state R² |
| --- | ---: | ---: | ---: | ---: |
| 8 | 0.964 | 8 / 8 | -0.022 | -6.049 |
| 32 | 0.901 | 8 / 8 | -0.022 | -44.772 |

The latest-observation baseline was negative at both delays. The secondary
linear entropy probe was negative for every model seed at both delays.

Artifact: `data/diagnostics/demian_uncertainty_representation_v1_20261012/`.

## Correction

The generator paired schedule index with hidden-target parity. Since schedule
can therefore carry evaluator-side label structure, the held-out probe cannot
distinguish schedule association from retained accumulated evidence. Preserve
the artifact as a measurement record; do not interpret its scores as support.

## Limits

- The corrective gate balances target within every reliability schedule,
  audits schedule-only prediction, and uses stratified complete-episode splits.
