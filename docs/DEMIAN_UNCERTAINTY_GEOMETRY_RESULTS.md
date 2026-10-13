# Uncertainty geometry: measurement record

Registry promotion pending. Completed artifacts remain immutable; hypothesis
status remains the reviewed narrow `H-UNC-001` preliminary support.

| Gate | Observation | Boundary |
| --- | --- | --- |
| Paired delay | B median frozen R2 -26.35; C -29.68 | Two extra blank steps suffice for calibration failure. |
| Global L2 | B median R2 -11.32; C -16.18 | Total magnitude contributes but does not restore transfer. |
| Paired geometry | B distance correlation 0.99977–0.99998; C 0.99889–0.99995 | Distances permit common translation/rotation; no stable evidence-axis conclusion. |

Evidence directories under `data/diagnostics/`:

- `demian_uncertainty_length_invariance_v1_20261013/`
- `demian_uncertainty_l2_invariance_v1_20261013/`
- `demian_uncertainty_geometry_v1_20261013/`

Interpretation: nearly preserved between-history distances can coexist with
failed fixed affine decoding. Raw state cosine is dominated by the state
baseline; it does not establish preservation of smaller evidence differences.
No channel norm or displacement identifies a causal evidence carrier.

Falsifier for the radial-scale explanation: fresh channelwise radial
normalization fails frozen transfer under the prescribed seed/sensitivity gate.
Protocol: `DEMIAN_UNCERTAINTY_CHANNEL_INVARIANCE_PROTOCOL.json`.

Review: Astra accepted the paired and L2 measurement boundaries; L2 summary
metadata was corrected in `7cbc0e5`. Geometry review confirmed translation and
rotation ambiguity. Formal registry entries for these three gates are pending.
