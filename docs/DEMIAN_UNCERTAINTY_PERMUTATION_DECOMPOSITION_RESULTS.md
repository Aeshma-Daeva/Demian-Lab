# Matched-target cue-pair permutation decomposition

Protocols: `DEMIAN_UNCERTAINTY_PERMUTATION_DECOMPOSITION*_PROTOCOL.json`.
Artifacts: `data/diagnostics/demian_uncertainty_permutation_decomposition*20261014/`.

## Observation

Discovery evaluated identity plus five adjacent cue-pair swaps on four seeds.
The frozen rule selected `swap_56 - identity` at delay8. Fresh confirmation
repeated that signed continuous-margin contrast in 4/4 seeds: mean `+0.2363`.

| fresh delay8 seed-median mean | identity | `swap_56` |
| --- | ---: | ---: |
| true tracked R² | 0.2369 | 0.2792 |
| analytic latest-cue R² | 0.4062 | -0.0069 |
| continuous transport margin | -0.1813 | 0.0550 |

`swap_56` changes the final cue from reliability `.95` to `.5`. Its margin
gain is therefore substantially driven by the analytic recency comparator,
not a demonstrated increase in transported evidence.

## Geometry

Fresh delay8 `swap_56` versus base: centroid displacement `0.08374`, centered
spread `0.76674 -> 0.76694`, centered 16D subspace overlap `0.97613`.
These are descriptive full-state cloud measures, not a channel explanation.

## Interpretation

Cue-pair order can reproducibly change the declared frozen transport margin.
This package does not establish a recency mechanism, geometric cause, channel
relation, or routing effect.

Astra review: the margin replication is valid but comparator-driven; geometry
does not nominate a relation or intervention.

## Next gate

Freeze two upstream permutations with the exact same final cue-pair. Compare
true R² and margin under that matched analytic baseline; retain identity only
as a recency reference. Analyze geometry or channels only after a residual
true-R² effect replicates.
