# Representation geometry and channel relations

Protocol: `DEMIAN_UNCERTAINTY_REPRESENTATION_RELATIONS_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_uncertainty_representation_relations_v3_20261013/`.
This is an analysis-only read of the two preserved matrix batches. Source
manifests, states and metadata are hash-verified before analysis. Astra review
found no status-promotion basis.

## Observation

Within each reliability order, cue6-to-delay evolution expands the state cloud
strongly in both batches. Median scale ratio at delay2/d8/d32/d64 is
approximately `1.18/3.04/39.9/220`. Pairwise-history geometry remains high
but declines from about `.955` at delay2 to `.895` at delay64. Rank-8
history-space subspace overlap declines from about `.94` to `.45`; normalized
Procrustes residual rises from about `.26` to `.55`.

At a matched checkpoint, train-order and heldout-order clouds are near-isometric
after translation, uniform scaling and an unconstrained orthogonal map. Across
batches, pairwise-distance alignment is at least `.9992`; normalized residual
is at most `.033`; scale ratios remain near one. This is relational geometry,
not raw-coordinate identity.

Gate-to-channel change-norm correlations vary sharply by seed, interval and
order, commonly spanning negative to positive values. No relation has a stable
direction across the eight parameter seeds.

## Interpretation

The moving within-order representation combines large common expansion with
partial deformation of history relations. The near-isometry across cue orders
shows that frozen-decoder portability can fail despite similar pairwise
geometry, because a fixed decoder is sensitive to translation and rotation.
The analysis does not establish that rotation causes portability variation.

The gate co-change measurements do not nominate a reproducible gate-to-channel
mechanism. They remain descriptive candidates for a later preregistered test.

## Limits and next gate

Timepoints, orders and channels are dependent observations within a parameter
seed. No causal intervention, task behavior, route ownership, memory use,
learning, adaptation or hypothesis-status change follows.

If continuing descriptive work, preregister one heldout test that decomposes
late cross-order transfer into identity versus orthogonal-map alignment and
evaluates the corresponding frozen decoder transfer. Do not resume broad
geometry sweeps.
