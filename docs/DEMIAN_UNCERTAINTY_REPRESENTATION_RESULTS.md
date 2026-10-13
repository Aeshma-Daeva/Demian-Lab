# Uncertainty Representation Probe Results

Status: preliminary support after Astra scientific review, 2026-10-12. This is not a causal,
routing, utility, learning, or adaptation result.

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

## Interpretation

This supports held-out decodability of accumulated public-evidence log odds
from the full state after delay in these fixed model realizations. It does not
establish a calibrated internal belief, an uncertainty code, causal channel
contribution, routing utility, learning, or adaptation.

## Limits

- This is a discovery split over fresh public histories, not a fresh model-seed
  confirmation.
- Full state has substantially more probe features than individual channels.
- Entropy is nonlinear in log odds; its linear-probe failure is not evidence of
  absent information.
- The next gate is repeated randomized episode splits, schedule/length transfer,
  probe-capacity sensitivity, then matched channel reset/restore.
