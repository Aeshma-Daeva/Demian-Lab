# Two-axis geometry and fixed Jacobian

Artifact: `data/diagnostics/demian_uncertainty_jacobian_v1_20261013/`.
Protocol: `DEMIAN_UNCERTAINTY_JACOBIAN_PROTOCOL.json`.
Astra reviewed evaluability; registered `R-UNC-JACOBIAN-001`.
`H-UNC-001` unchanged.

Scope: four fresh seeds; train32 paired histories; heldout32 histories in the
same order and exhaustive64 histories in a new order. 1,024 states, 48 metrics,
512 exact float32 continuation checks. Maximum double-rollout discrepancy:
2.72e-5 in full-state norm.

| Map | Same-order median relative residual | New-order median relative residual |
| --- | ---: | ---: |
| Translation | 0.256 | 0.275 |
| Similarity | 0.026 | 0.107 |
| Rank8 Jacobian | 0.046 | 0.110 |
| Full Jacobian | 0.046 | 0.050 |

Observation: full Jacobian meets the <=0.1 criterion in every seed/axis cell;
similarity meets it for same-order histories but fails strict new-order transfer.
The fixed local derivative is calculated at the training A centroid, without
fitted correction or evaluation-state relinearization.

Interpretation: ordinary local recurrent flow suffices for the measured
two-blank-step geometry. Fitted similarity is more accurate on the same-order
cloud; the derivative generalizes better across the new order. No distinctive
memory mechanism, decoder, causal channel or functional use is established.

Coverage boundary: cloud-centered excluded variance (<0.003) is not coverage of
displacements from the training centroid. Derived from the saved source states
and basis, 58.4–92.9% of the new-order centroid displacement lies outside rank8;
10.8–19.2% of all reference-centered direction energy lies outside it.
Reproduction function: `_reference_coverage` in
`development/run_demian_uncertainty_horizon.py`.

Falsifier: the same fixed-centroid derivative loses accuracy over longer blank
continuations or broader sampled neighborhoods. Next: declared horizons
2/4/8/16/32 on these cached cases; this extends scope, not replication.
