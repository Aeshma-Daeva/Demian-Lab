# Cross-seed relational geometry

Protocol: `DEMIAN_CROSS_SEED_GEOMETRY_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_cross_seed_relational_geometry_v1_audited_20261016/`.
Review: Astra; descriptive result only. No hypothesis-status change.

## Observation

For 64 canonically matched binary histories under one fixed reliability order,
pairwise-distance matrices are highly similar across the four frozen parameter
seeds. Median paired distance correlation and normalized discrepancy:

| Checkpoint | Correlation | Discrepancy | Paired minus shuffled |
| --- | ---: | ---: | ---: |
| cue6 | 1.000 | .014 | 1.008 |
| delay2 | .999 | .018 | 1.001 |
| delay8 | .999 | .022 | 1.003 |
| delay32 | .994 | .065 | .993 |
| delay64 | .949 | .176 | .965 |

The six seed-pair values are dependent comparisons over four seeds. The
shuffled comparator uses 32 deterministic history permutations per cell.
Unaligned same-history raw-state cosine is near zero at every checkpoint.

## Interpretation

Across this corpus, different parameter seeds instantiate highly similar
**relational** history geometry despite nonshared raw axes. This is a useful
cross-seed baseline for later perturbation and routing comparisons.

## Boundary

The raw-history comparison uses ordinal symbol sequences, not actual encoder
features. It does not exclude an encoder-geometry explanation. Near-zero raw
cosine does not exclude a common coordinate system after alignment. This is
not cross-seed decoder transfer, a mechanism result, causal evidence, or an
architecture comparison.

Falsifier: a new seed batch with matched histories fails the predeclared
paired-versus-shuffled relational contrast under the same source selection.
