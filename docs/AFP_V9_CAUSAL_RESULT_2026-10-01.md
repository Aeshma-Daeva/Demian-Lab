# Native-v9 AFP Causal State Surgery — 2026-10-01

Status: candidate-specific causal follow-up completed after
`AFP_V9_CAUSAL_PREREG_2026-10-01.md` was committed.

## Question

The held-out lineage study identified native-v9 operational AFP candidates at
seeds 104, 112, 123 and 126.

Native v9 exposes:

```text
surface = fast
hidden context = slow + control
```

This allows an exact intervention:

```text
same fast surface
different slow/control
same deterministic future
```

If the future surface changes, the hidden state is causally relevant even
though the current exposed surface is unchanged.

## Sanity control

For all 28 held-out seeds, both future conditions and every exact full-state
clone:

```text
initial surface gap = 0
full-clone future gap = 0
```

The intervention therefore does not begin by changing the measured surface.

## Candidate aggregate

Mean 64-step surface gap from the unmodified continuation across confirmed AFP
seeds 104, 112, 123 and 126:

| Hidden intervention | Autonomous future | Same shared impulse |
| --- | ---: | ---: |
| full clone | 0.000000 | 0.000000 |
| surface-only: reset slow + control | **0.102147** | **0.102156** |
| reset slow only | **0.096817** | **0.096733** |
| reset control only | 0.003389 | 0.003381 |
| stale hidden state, 16 steps | 0.012626 | 0.012672 |
| stale hidden state, 32 steps | 0.011469 | 0.011474 |
| stale hidden state, 64 steps | 0.015072 | 0.015132 |
| stale hidden state, 96 steps | 0.018188 | 0.018185 |

The shared external impulse barely changes the aggregate effect. The latent
context remains consequential under the same future perturbation.

## Channel localization

The strongest result is the difference between slow and control ablation:

```text
reset slow:    mean future gap ~= 0.0968
reset control: mean future gap ~= 0.00339
```

Under this probe, native-v9 continuation is therefore much more strongly
anchored in `slow` than in `control`.

This is consistent with the earlier capsule-continuity line in which canonical
v9 continuity was concentrated strongly in slow state.

## History-preserving interventions

The stale-state arms are more informative than zero resets because they replace
the current hidden context with a **real earlier hidden state from the same
trajectory** while preserving the current fast surface exactly.

Across AFP candidates, replacing slow/control with their values from 16–96
steps earlier produces a nonzero future surface divergence.

Candidate autonomous means:

```text
stale 16: 0.01263
stale 32: 0.01147
stale 64: 0.01507
stale 96: 0.01819
```

So at a surface-quiescent checkpoint, the exact hidden trajectory history is
not interchangeable.

## Candidate examples

### Seed 104

```text
surface-only mean future gap  0.130885
slow-reset mean future gap    0.129067
stale-96 mean future gap      0.017701
```

### Seed 112

```text
surface-only mean future gap  0.079081
slow-reset mean future gap    0.071824
stale-96 mean future gap      0.002669
```

Seed 112 shows that stale-history sensitivity is not equally large in every AFP
candidate even though complete hidden-state deletion still changes the future.

### Seed 123

```text
surface-only mean future gap  0.116848
slow-reset mean future gap    0.113731
stale-16 mean future gap      0.046089
stale-96 mean future gap      0.039468
```

### Seed 126

```text
surface-only mean future gap  0.081773
slow-reset mean future gap    0.072648
stale-96 mean future gap      0.012916
```

## Noncandidate control

Latent surgery also affects held-out native-v9 runs that did not satisfy the
AFP criterion.

In fact, stale-state replacement often produced **larger** mean gaps in the
noncandidate group than in the AFP-candidate group.

Therefore the causal role of slow/control is **not unique to AFP**.

The stronger combined result is narrower:

> native v9 generally carries future-relevant hidden state, and in a subset of
> held-out runs that hidden state remains causally relevant while the exposed
> fast surface simultaneously satisfies the preregistered surface-quiescent,
> latent-accumulating AFP criterion.

This distinction prevents the experiment from confusing general recurrence
with AFP-specific dynamics.

## Strongest current statement

For native v9, held-out operational AFP candidates have now satisfied three
separate layers:

1. **surface quiescence**
   - exposed fast movement below the preregistered threshold;

2. **latent accumulation/dynamics**
   - complete-state motion at least 10x surface motion plus increasing latent
     norm;

3. **causal future relevance**
   - altering only hidden slow/control state at exactly the same current surface
     changes the deterministic future surface trajectory.

That supports the term:

**causally consequential operational accumulating fixed-point candidate**

with the continuing boundary that this is an observable/projected fixed point,
not a full-state mathematical fixed point.

## Next experiments

The next scientific steps are:

1. recover the exact historical C16 evolved five-channel genomes if a private
   archive still exists;
2. rerun the same direct criterion on those candidates;
3. repeat native-v9 confirmation with additional held-out seed blocks;
4. vary hidden size and state gain to map the AFP region rather than one
   parameter point;
5. construct a simple controlled dynamical-system coupling demo for external
   presentation;
6. package the lineage + state-surgery result as the first Centec research
   demonstration.
