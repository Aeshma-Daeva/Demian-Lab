# AFP Lineage Probe

This experiment maps surface/latent dynamics across selected Demian ancestors:

```text
native v2 -> native v3 -> native v8 -> native v9
                                  \
                                   -> v9 five-channel -> Demian v1
```

The selected line deliberately avoids the native versions whose primary
experimental object was changing model weights. The first pass therefore keeps
the comparison centered on explicit recurrent state.

## Question

Where in the lineage does the following regime appear?

```text
surface velocity -> approximately zero
latent/recurrent-state velocity -> measurably non-zero
```

This is called an accumulating fixed-point (AFP) candidate here. It is a
**projected/observable fixed point**, not a mathematical fixed point of the
complete dynamical system.

## Why this experiment exists

The historical lab used a broader fixed-point interior classifier. In that
classifier, a run could be labelled `accumulating_fixed_point` when the exposed
trajectory was classified as `FIXED_POINT` while slow/message statistics
continued to show accumulation.

The new probe records that historical label **and** direct measurements:

- exposed-surface RMS step size;
- full recurrent-state-tuple RMS step size;
- relative velocity after dividing movement by state scale;
- absolute latent/surface RMS movement ratio;
- surface and latent norm/delta slopes over the tail;
- surface dimension / recurrent-state dimension;
- clean and perturbed arms;
- separate relative and absolute threshold-sensitivity grids.

The two definitions are not forced to agree. Disagreement is useful evidence
about what the older classifier was actually detecting.

### Exploratory threshold warning

The first lineage batch showed that normalizing latent movement by the latent
state norm can obscure an accumulating state whose norm has itself become very
large. A second, absolute-RMS sensitivity view was therefore added **after
inspection of the first batch**. Its default threshold is exploratory/post-hoc
and must not be treated as confirmatory evidence. Any publication-grade run
should preregister the chosen criterion before collecting the confirmation
seeds.

## Variants

- `native_v2`: recruitment-biased explicit route substrate.
- `native_v3`: v2 plus endogenous tightness state.
- `native_v8`: compressed seven-channel comparison.
- `native_v9`: minimal `fast/slow/control` scaffold.
- `v9_5ch_default`: v9 plus `message/carrier`.
- `v9_5ch_accumulator`: the historical tuned accumulator configuration.
- `v9_5ch_accumulator_zero_init`: accumulator with zero-initialized
  message/carrier, closer to the later evolutionary setup.
- `demian_v1`: explicit six-channel gate-state synthesis.

The exact C16 evolutionary archive used evolved v9 five-channel genomes. Those
raw archive candidates are not currently present in the public workbench
branch, so this script does **not** claim to reproduce C16 exactly. It tests the
publicly available architectural ancestors of that result.

## Run

```bash
python -m development.afp_lineage_probe \
  --hidden-size 32 \
  --steps 384 \
  --tail-steps 96 \
  --seeds 94,95,96,97,98,99,100,101 \
  --perturb-step 192 \
  --perturb-scale 0.35 \
  --output-dir outputs/afp-lineage
```

Outputs:

- `afp_lineage_results.json`: full per-seed metrics and sensitivity grid.
- `afp_lineage_summary.csv`: compact lineage-level summary.

## Interpretation boundary

A useful outcome can be any of the following:

1. historical and strict AFP labels agree in one or more ancestors;
2. the old classifier reports AFP but direct latent/surface velocity does not;
3. AFP exists only in evolved v9 five-channel candidates not recoverable from
   the public scaffold;
4. Demian v1 replaces surface-fixed accumulation with another form of latent
   path dependence.

The experiment should report whichever outcome occurs. It should not tune the
thresholds after seeing the result in order to force an AFP claim.


## Historical classifier audit

The older attractor classifier calls a trajectory `FIXED_POINT` when its most
recent exposed RMS step is below an absolute threshold corresponding to
`velocity_magnitude < 0.02`. In the current implementation,
`velocity_magnitude = residual_delta / 3`, so the effective surface threshold
is approximately `residual_delta < 0.06`.

The historical interior classifier then promotes a fixed-point run to
`accumulating_fixed_point` when slow/message norms or contraction ratios cross
additional thresholds.

This is a useful historical heuristic, but it is substantially looser than the
new direct convergence criterion. The lineage study therefore reports the old
label as **historical classifier output**, not as proof of a mathematical fixed
point.
