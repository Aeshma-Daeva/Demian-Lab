# Matched-target order tracking

Protocol: `DEMIAN_UNCERTAINTY_MATCHED_TARGET_TRACKING_PROTOCOL.json`.
Artifacts: `data/diagnostics/demian_uncertainty_matched_target_matrix_v1_20261014/` and `data/diagnostics/demian_uncertainty_matched_target_tracking_v1_20261014/`.

## Observation

For every paired binary history, the heldout sequence contains the same
symbol–reliability pairs and exactly the same public log-odds target. With four
fresh seeds and split109, within-order orthogonal tracking clears 12/12 cells
at delay2 and delay8. Strict matched-target heldout-order tracking clears 9/12
at delay2 and 7/12 at delay8.

Identity has strongly negative R2 and scalar-scale correction remains negative
on average. Orthogonal tracking improves the heldout median R2 to `.596` at
delay2 and `.487` at delay8, but misses the declared all-cell gate.

## Interpretation

The earlier changed-target order design overstated the apparent cross-order
failure: when target mismatch is removed, frozen tracking transfers in a
majority of cells. Remaining failures show that target mismatch is not the only
cause. Translation, scale and orientation changes are all relevant, but this
measurement does not identify their channel-level cause.

## Limits and next gate

The 12 projection cells are dependent within four seeds. A proper orthogonal
map can improve transport without being an internal mechanism. The next package
should characterize the remaining failed cells with preregistered geometric
residuals and channel/route relationship measurements, then confirm candidates
on fresh seeds before interventions.
