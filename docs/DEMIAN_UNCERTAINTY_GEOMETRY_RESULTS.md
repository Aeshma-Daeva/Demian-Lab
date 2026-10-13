# Uncertainty geometry: measurement record

Fresh channelwise gate registered as `R-UNC-CHANNEL-001` following Astra review.
Earlier paired/L2/geometry entries remain pending. Hypothesis status remains
the reviewed narrow `H-UNC-001` preliminary support.

State dimension at hidden width 256 is 1,312: five 256-coordinate channels
and a 32-coordinate control channel. Prior 1,536-coordinate prose was wrong;
executed probes flatten the actual tensors. Corrective training splits contain
176 episodes, not the previously described 88.

| Gate | Observation | Boundary |
| --- | --- | --- |
| Paired delay | B median frozen R2 -26.35; C -29.68 | Two extra blank steps suffice for calibration failure. |
| Global L2 | B median R2 -11.32; C -16.18 | Total magnitude contributes but does not restore transfer. |
| Paired geometry | B distance correlation 0.99977–0.99998; C 0.99889–0.99995 | Distances permit common translation/rotation; no stable evidence-axis conclusion. |
| Fresh channelwise L2 | B median R2 -36.56; C -33.77; control wins 5/72 and 14/288 | Every seed fails the prescribed gate; radial normalization search ends. |

Evidence directories under `data/diagnostics/`:

- `demian_uncertainty_length_invariance_v1_20261013/`
- `demian_uncertainty_l2_invariance_v1_20261013/`
- `demian_uncertainty_geometry_v1_20261013/`
- `demian_uncertainty_channel_invariance_v1_20261013/`

Interpretation: nearly preserved between-history distances can coexist with
failed fixed affine decoding. Raw state cosine is dominated by the state
baseline; it does not establish preservation of smaller evidence differences.
No channel norm or displacement identifies a causal evidence carrier.

Falsifier for the radial-scale explanation: fresh channelwise radial
normalization fails frozen transfer under the prescribed seed/sensitivity gate.
Protocol: `DEMIAN_UNCERTAINTY_CHANNEL_INVARIANCE_PROTOCOL.json`.

Fresh-gate scope: seeds 494–497 and 594–597; new heldout reliability order;
128 train-base histories and 64 heldout base histories, with all four neutral
suffixes paired within each base history. The 72/288 cells are dependent
sensitivity settings, not independent sample sizes. Heldout A also fails the
strict all-sensitivity criterion (51/72 control wins; median R2 0.484).

Observation: B/C median correlations remain 0.746/0.734 despite poor R2.
Interpretation: a correlation can survive baseline/calibration error; it does
not establish a stable native evidence code. Frozen transfer under the tested
global and channelwise radial transforms is unsupported. No causal escalation.

Next question: how much paired movement is shared translation, rotation, or
history-dependent deformation? Answer with descriptive centered geometry and
heldout transformation residuals, rather than selecting more normalizations.

Review: Astra accepted the paired and L2 measurement boundaries; L2 summary
metadata was corrected in `7cbc0e5`. Geometry review confirmed translation and
rotation ambiguity. Formal registry entries for these three gates are pending.
