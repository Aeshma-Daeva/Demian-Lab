# Matched-target geometry and relations

Protocol: `DEMIAN_UNCERTAINTY_MATCHED_TARGET_RELATIONS_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_uncertainty_matched_target_relations_v1_20261014/`.

## Observation

Matched-target cross-order clouds have median scale ratio `1.00` at cue6,
delay2 and delay8, with high pairwise-distance alignment (`.957/.968/.957`).
They are not raw-coordinate identical: normalized Procrustes residual is
`.256/.220/.217`; rank-8 history-space overlap declines `.967/.852/.805`.

Within order, cue6-to-delay evolution expands strongly: median centroid shift
is `8.48` at delay2 and `51.87` at delay8; scale ratio is `1.16` then `3.04`.
Gate/non-gate change-norm correlations span nearly `-1` to `+1` across channels
and seeds at both delays.

## Interpretation

The remaining matched-target transport failures occur despite near-preserved
pairwise geometry and uniform global scale. The clouds have partial orientation
or deformation differences, but no stable gate-to-channel co-change direction
identifies a candidate mechanism. This narrows the next work to a preregistered
projection-level relationship test rather than route intervention.
