# Cross-seed anchor transfer

Protocol: `DEMIAN_CROSS_SEED_TRANSFER_PROTOCOL.json`, frozen at `8ec1ecf`.
Artifact: `data/diagnostics/demian_cross_seed_anchor_transfer_v1_recovered_20261017/`.
Scope: four fitting seeds, six fresh evaluation seeds, 16 anchor / 24 fitting /
24 evaluation histories; one reliability order; ten checkpoints. Separate
external decoder at each checkpoint. Demian architecture and weights frozen.

## Observation

Median evaluation-seed R² and passing seeds under the declared criterion:

| Delay | Demian anchor | Demian permutation q95 | Demian passes | Reservoir anchor | Reservoir passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | .296 | .475 | 0/6 | .951 | 6/6 |
| 8 | .305 | .462 | 0/6 | .852 | 6/6 |
| 32 | .417 | .489 | 0/6 | .689 | 6/6 |
| 64 | .406 | .466 | 0/6 | .516 | 4/6 |

Neither system passes the overall four-delay gate. Demian exceeds the
latest-cue and shuffled-label controls, but not the required margin over the
95th percentile of 32 fixed anchor permutations. Some wrong anchor assignments
match or exceed the decoder; this does not describe the typical permutation.

Demian median cross-seed distance correlation is `.9992/.9995/.9915/.9751`
at delays2/8/32/64. These dependent seed-pair comparisons describe same-time
geometry, not temporal invariance or decoding quality.

Median PC1 fraction is `.956` at delay8 and `.960` at delay32; participation
rank is approximately `1.09/1.08`. A train-history affine six-bit state model
explains `.999893/.985486` of heldout state variation at these delays, declining
to `.810693` at delay64. The anchor-fitted affine surrogate's decoding closely
matches the original through32. Residual geometry remains cross-seed correlated;
the affine explanation is incomplete, especially later.

## Interpretation

The shared geometry replicates, while the frozen distance readout fails its
correspondence-specificity criterion. Positive transfer R² remains observed.
Within-seed anchor decoding is similarly limited (`.314/.344/.374/.381`),
so the primary failure cannot be assigned solely to differences between seeds.
Low-dimensional input filtering and readout conditioning remain alternatives.

The reservoir preserves broader history geometry (participation rank roughly
4–5) and performs better on this decoder test. State width, public frames and
probe rules match; parameters, initialization, input mixing and regime differ
(Demian 1,545,056 versus reservoir 2,058,528 parameters). No architecture ranking.

## Integrity and review

The original run completed collection, fits and predictions, then failed JSON
export on a NumPy scalar. `5ad71b6` repairs export and adds hash-verified recovery.
Original failure, source and observations remain preserved. All 1,084 recovered
prediction arrays exactly match the original; no primary decoder was refitted.

Astra independently verified manifests, frozen-fit hashes, predictions, splits
and gates. Review: `review.json`. No hypothesis-status change.
Raw state/parameter archives remain local and hash-listed; compact evidence,
fitted readouts, calibrations and source snapshots are versioned.

Verification: 38 focused tests pass; full suite 560 pass / 4 fail. Failures are
missing historical artifacts: current-artifact loader, manifest validation,
v10 summary digest and v10 summary rebuild.

## Bounded continuation

Diagnose distance-feature conditioning against squared-distance contrasts and
train-only whitening on these cached states, explicitly as exploratory analysis.
Freeze any selected explanation before fresh-seed confirmation. Channel or
routing changes remain downstream of predictive evidence.

Falsifier: a prospective fresh-seed run fails to reproduce the stated geometry,
linear-surrogate adequacy or decoder/control contrasts under this protocol.
No information-erasure, autonomous readout, adaptation, AFP or causal claim.
