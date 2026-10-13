# Availability versus portability

Artifacts:
`data/diagnostics/demian_uncertainty_history_matrix_v1_20261013/`,
`data/diagnostics/demian_uncertainty_history_matrix_replication_v1_20261013/`.
Protocols: `DEMIAN_UNCERTAINTY_HISTORY_MATRIX_PROTOCOL.json` and replication.
Astra reviewed both as evaluable: `R-UNC-MATRIX-001` and
`R-UNC-MATRIX-REP-001`. `H-UNC-001` remains preliminary.

Observation: each independently allocated batch beats analytic latest and
matched conditional shuffle in every within-order, matched-time full-state
projection cell. Fits use32 histories; disjoint32 test histories evaluate.

| Checkpoint | First batch median R² | Replication median R² |
| --- | ---: | ---: |
| cue6 |0.886|0.895|
| delay2 |0.695|0.852|
| delay8 |0.634|0.767|
| delay32 |0.507|0.734|
| delay64 |0.342|0.668|

Controls: fixed16-dimensional projections with seeds17/29/43; ridge alpha1;
train-only coefficients/standardization and conditional shuffle; analytic
current/latest single-cue semantics. Base-order label/sign stratification fixes
the disjoint history split; evaluation-order label distributions may differ.

Portability: cross-order comparator wins at cue6/d2/d8/d32/d64 are
0/12,0/12,2/12,1/12,8/12 in batch1 and2/12,2/12,8/12,11/12,11/12 in batch2.
Later transfer improves in batch2; neither meets the declared all-cell rule.
Cue6-frozen probes fail every delayed within-order comparison in both batches.

Interpretation: accumulated public evidence is available through64 blank ticks
in the tested fixed-order, matched-time regimes. A single frozen cue6 coordinate
system is not portable through delay. This is neither erasure nor calibrated
uncertainty, autonomous memory use, route ownership, learning or adaptation.

Scope: eight fresh parameter seeds in two separate batches with distinct order
pairs and32/32 allocations. Projection/time cells are dependent sensitivities,
not additional independent replications. Architecture advantage is untested.

Falsifier: fresh matched-time heldout histories fail fair public/shuffle controls.
Temporal/order portability must be tested separately, not inferred from availability.

Research-direction fork: characterize the existing order/time-specific code,
or define a functional objective for pursuing portable representation.
No further geometry tuning or automatic architecture modification.
