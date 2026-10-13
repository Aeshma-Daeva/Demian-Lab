# Paired similarity geometry

Artifact: `data/diagnostics/demian_uncertainty_similarity_v1_20261013/`.
Protocol: `DEMIAN_UNCERTAINTY_SIMILARITY_PROTOCOL.json`.
Astra-reviewed evaluable diagnostics, registered as `R-UNC-SIMILARITY-001`;
`H-UNC-001` unchanged.

Scope: four fresh model seeds; 64 paired training histories and 64 histories
under a fresh heldout order; A/B/four C variants; 3,072 saved states.
Rank8 basis and similarity maps fit only training clouds. Metrics use heldout
A-cloud RMS as denominator; histories/variants are dependent within seed.

| Extra-delay B, seed median | Relative full residual | Relative centered residual |
| --- | ---: | ---: |
| Identity | 3.711 | 0.224 |
| Train-mean translation | 0.265 | 0.224 |
| Paired similarity | 0.144 | 0.048 |
| Randomized-pair similarity | 1.208 | 1.170 |

Observation: similarity full residual ranges 0.111–0.271 across all seeds and
B/C variants, missing the prescribed <=0.1 criterion in every cell. Centered
residual ranges 0.038–0.064. Heldout excluded variance is below 0.006, satisfying
the <=0.01 coverage criterion. Most remaining error is a shared centroid offset.

Interpretation: training-derived similarity approximates the heldout cloud's
centered arrangement; it does not fully predict its location. The large
displacement fraction explained is dominated by common translation and is
insufficient alone to establish preserved shape. No stable evidence axis,
decoder recovery, causal carrier, or memory use is demonstrated.

Falsifier: same-order heldout histories also have large centered residuals,
or a generic local linearization explains the apparent structure equally well.
Next: B-only two-axis gate, distinguishing same-order heldout histories from
unseen-order offset, with a fixed model-derived two-delay Jacobian baseline.

Reproduction: `states.npz` plus `state_metadata.jsonl` preserve exact states and
pairing. `transforms.npz` contains `<seed>_basis` and `<seed>_<variant>_*`
means/rotation/scale; manifest pins source commit, protocol, and file hashes.
Allocation seed 20261015 is an RNG identifier; the run date is 2026-10-13.
