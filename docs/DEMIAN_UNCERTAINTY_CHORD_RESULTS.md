# Natural-history chord response

Protocol: `DEMIAN_UNCERTAINTY_CHORD_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_uncertainty_chord_v1_20261013/`.
Astra review: evaluable; registered `R-UNC-CHORD-001`. No hypothesis promotion.

Observation: all13,824 cases satisfy the precision margin. All768 source
continuations and1,536 full-chord endpoint continuations replay exactly.
Quarter-chord responses satisfy the0.1 relative linearity bound through h8.
Full-chord h4 and half-chord h8 include natural-only threshold failures.

Full-chord h8 median relative errors:

| Seed | Early cue: natural / random | Final cue: natural / random |
| --- | --- | --- |
|1294|0.085 /0.076|0.036 /0.027|
|1295|0.212 /0.136|0.081 /0.045|
|1394|0.049 /0.046|0.023 /0.022|
|1395|0.080 /0.084|0.026 /0.030|

Control: random disturbances match requested full-state L2 norm at the same
source, with two fixed isotropic directions. Paired contrast uses the mean of
those controls per source; seed1395 reverses the contrast for both cue flips.
Overall failures:183/4,608 natural versus130/9,216 control cases. These unequal
denominators and dependent cases are not independent replication counts.

Interpretation: finite history-created displacements can exceed local-linear
adequacy. Excess relative to equal-sized controls is seed-dependent.
The result does not establish a seed-general special memory geometry.

Scope: four fresh parameter seeds, exhaustive32 history pairs per flip, both
orientations. Early cue reliability0.95 and final cue reliability0.5 confound
position with reliability. Interpolants need not be reachable; endpoints are.
Requested radii span0.18–4.47% of anchor norm, exceeding the previous radius1
study. No causal channel, uncertainty calibration, memory use or adaptation.

Reproduction: `summarize_metrics` in the chord runner; `paired_analysis.json`
pins the original manifest hash and analysis commit. Original393MiB NPZ remains
local;14 lossless publication parts reconstruct every named array and dtype.
Use `prediction_parts/partition_manifest.json`; original run hashes unchanged.

Falsifier of universal local adequacy: valid natural chord responses exceed0.1;
observed in the declared finite windows. No architectural modification follows.

Decision: stop local-geometry gates. Natural-only failures are concentrated in
seed1295, not a replicated direction exception. Next: fresh time-resolved frozen
calibration across cue boundaries and delays. Channel breakdown is bookkeeping,
not a causal gate.
