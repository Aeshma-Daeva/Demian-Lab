# Matched-target transport descriptor gate

Corrected protocol: `DEMIAN_UNCERTAINTY_MATCHED_TARGET_DESCRIPTOR_CORRECTED_PROTOCOL.json`.
Corrected artifact: `data/diagnostics/demian_uncertainty_matched_target_descriptor_corrected_v1_20261014/`.

The prior descriptor artifact is unevaluable for association: it fit channel
maps and relationship predictors using heldout-order train states. The separate
frozen tracking strata remain valid.

## Observation

Across four fresh seeds and three fixed projections, the valid frozen tracking
strata have `0/12` delay2 and `2/12` delay8 strict heldout transport passes.
The corrected analysis fits and saves every channel map and relationship
predictor on base-order train32 only before evaluation on heldout eval32.

No descriptor can meet the preregistered nomination criterion: delay2 has no
pass cells and delay8 has only two pass cells. No projected residual, channel
residual-energy fraction, carrier-to-slow residual relation, or gate-to-message
residual relation is nominated.

## Interpretation

The fresh batch does not reproduce stable frozen portability and cannot yet
support a relationship explanation for its failures. This is not evidence of
information erasure, route causality, adaptation, or a channel mechanism.
