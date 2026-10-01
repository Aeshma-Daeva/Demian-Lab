# AFP Lineage Audit — 2026-10-01

Status: exploratory lineage result. The confirmatory criterion at the end of
this note is frozen before running new held-out seeds.

## Setup

The first direct lineage batch used:

- hidden size: 32;
- 384 autonomous recurrent steps;
- final analysis window: 96 steps;
- seeds: 94–101;
- clean and perturbation arms;
- eight selected variants from native v2 through Demian v1.

The experiment measured the exposed surface and the **complete recurrent state
tuple** separately.

## Main result

The historical classifier labelled every run in every tested variant as:

```text
FIXED_POINT -> accumulating_fixed_point
```

That label is too coarse to serve as the modern AFP definition.

The historical `FIXED_POINT` test uses the most recent exposed
`residual_delta` through:

```text
velocity_magnitude = residual_delta / 3
FIXED_POINT if velocity_magnitude < 0.02
```

which is equivalent to an exposed RMS-step threshold of approximately:

```text
residual_delta < 0.06
```

The interior classifier then calls that fixed point accumulating when slow or
message norm/contraction statistics cross additional thresholds.

This was useful as an exploratory basin-interior heuristic, but it does not
establish convergence of the observed surface.

## Direct surface/latent measurement

Using RMS movement normalized per dimension, the clean-arm mean
latent/surface movement ratios were:

| Variant | Mean latent/surface RMS movement ratio |
| --- | ---: |
| native v2 | 12.94x |
| native v3 | 25.86x |
| native v8 | 25.86x |
| native v9 | 44.35x |
| v9 five-channel default | 19.47x |
| v9 five-channel accumulator | 10.43x |
| v9 five-channel accumulator, zero-init | 10.82x |
| Demian v1 | 28.85x |

So the broad phenomenon survives the audit:

> movement in the complete recurrent state can be much larger than movement in
> the exposed surface.

What does **not** survive is the claim that every historical
`accumulating_fixed_point` label was already a strict fixed point of the
surface.

## Closest AFP-like cases

A post-hoc exploratory screen required:

```text
mean surface RMS step <= 1e-3
mean latent RMS step  >= 1e-2
latent/surface RMS ratio >= 5
```

Only a small subset passed.

### native v2, seed 100

```text
surface RMS step       0.000438
latent RMS step        0.011295
latent/surface ratio  25.81x
surface-delta slope   -0.000005
latent-norm slope     +0.001191
```

### native v3, seed 100

```text
surface RMS step       0.000503
latent RMS step        0.041272
latent/surface ratio  82.08x
surface-delta slope   -0.000009
latent-norm slope     +0.031706
```

native v8 produced the same seed-100 trajectory under this configuration,
consistent with its shared/default v3-like recurrence parameters in this
comparison.

These are substantially closer to the intended AFP definition than the broad
historical label: the surface is very quiet, the full recurrent state is still
moving, surface motion continues to decrease, and latent norm is increasing.

### native v9

native v9 showed the strongest **mean** separation across seeds
(44.35x), but did not pass the post-hoc screen because no clean seed combined
both the `1e-3` surface threshold and the `1e-2` latent-motion floor.

Seed 96 was closest on surface convergence:

```text
surface RMS step       0.000265
latent RMS step        0.004816
latent/surface ratio  18.18x
```

Other v9 seeds retained much larger latent motion but had surface movement in
the roughly `0.001–0.005` range.

## v9 five-channel and Demian v1

The public v9 five-channel scaffold did **not** reproduce a strict AFP in this
batch. Its surface remained more active.

That does not falsify historical C16. C16 came from evolved v9 five-channel
archive genomes, and those exact archive candidate files are not present in the
current public workbench branch. The available scaffold and tuned accumulator
configuration are architectural relatives, not the archived evolved
individuals.

Demian v1 also remained surface-active:

```text
mean surface RMS step ≈ 0.0130
mean latent RMS step  ≈ 0.3379
mean latent/surface movement ratio ≈ 28.85x
```

This matches the separate v1 baseline experiment: v1 currently looks more like
**surface-compressed latent path dependence** than a surface-fixed accumulator.

## Interpretation

The original research intuition remains useful, but the terminology needs to
be sharpened.

A safer hierarchy is:

1. **surface/latent separation** — directly measured and common across the
   lineage;
2. **surface-quiescent latent dynamics** — a stronger subset;
3. **accumulating fixed-point candidate** — requires explicit convergence-like
   surface behavior plus continuing latent motion and evidence of accumulation;
4. **causally consequential AFP** — additionally requires matched-future or
   state-surgery evidence that the hidden difference changes continuation.

The project should no longer use the old heuristic label alone as evidence for
level 3 or 4.

## Confirmatory criterion frozen after this exploratory batch

Before viewing any new model seeds, the next held-out run will use the
following predeclared AFP-candidate criterion:

```text
A. mean surface RMS step over final 96 steps <= 1e-3
B. mean latent RMS step over final 96 steps  >= 1e-2
C. latent/surface RMS movement ratio         >= 10
D. surface-delta linear slope                <= 0
E. latent-state norm linear slope            > 0
```

Rationale:

- A requires a genuinely quiet exposed trajectory;
- B prevents a numerically frozen whole system from passing;
- C demands substantial separation between hidden and exposed motion;
- D requires the surface to be stable or still converging rather than drifting
  away;
- E operationalizes the word **accumulating** rather than merely hidden
  oscillation.

This criterion was chosen after inspecting seeds 94–101, so those seeds are
exploratory only. Confirmation must use unseen seeds.

## Next held-out test

Use fresh model/initialization seeds 102–129 with the criterion above frozen.

For every passing candidate:

1. preserve its complete recurrent state;
2. construct a surface-matched control where possible;
3. apply identical future perturbations;
4. measure future surface divergence;
5. ablate/swap candidate internal channels;
6. report failures as well as passes.

The result can therefore falsify AFP prevalence even if the exploratory
examples looked promising.
