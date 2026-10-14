# Matched-target transport descriptor gate

Protocol: `DEMIAN_UNCERTAINTY_MATCHED_TARGET_DESCRIPTOR_PROTOCOL.json`.
Artifacts: `data/diagnostics/demian_uncertainty_matched_target_descriptor_matrix_v1_20261014/`, `data/diagnostics/demian_uncertainty_matched_target_descriptor_tracking_v1_20261014/`, and `data/diagnostics/demian_uncertainty_matched_target_descriptor_v1_20261014/`.

## Observation

Across four fresh seeds and three fixed projections, strict matched-target
heldout transport passes `0/12` delay2 cells and `2/12` delay8 cells. The
delay2 stratum has no pass cells; delay8 has two passes from separate seeds.

No descriptor meets the preregistered nomination criterion: a predeclared
pass/fail separation in at least three of four seeds. Therefore no projected
residual, channel residual-energy fraction, carrier-to-slow residual relation,
or gate-to-message residual relation is nominated.

## Interpretation

This fresh batch does not reproduce stable frozen portability and cannot
support a relationship explanation for its failures. It is a descriptive
failure of the stated gate, not evidence of information erasure, route
causality, adaptation, or a channel mechanism.
