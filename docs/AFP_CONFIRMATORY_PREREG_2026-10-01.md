# AFP Confirmatory Preregistration — 2026-10-01

This file is committed **before** running the held-out seed set.

## Frozen discovery/confirmation split

Exploratory seeds:

```text
94–101
```

Confirmatory seeds:

```text
102–129
```

No confirmatory threshold may be changed after inspecting seeds 102–129.

## Architecture set

The held-out run will report all of the following, not only variants that looked
promising during exploration:

- native v2;
- native v3;
- native v8;
- native v9;
- v9 five-channel default;
- v9 five-channel tuned accumulator;
- v9 five-channel tuned accumulator with zero-initialized message/carrier;
- Demian v1.

The public v9 five-channel variants are not substitutes for the unavailable C16
evolved archive genomes.

## Fixed run shape

```text
hidden_size = 32
steps = 384
analysis tail = final 96 steps
clean arm = primary AFP classification arm
perturbation arm = secondary robustness description
```

## Primary AFP-candidate rule

A clean run is a confirmatory AFP candidate only if **all five** conditions hold:

```text
mean surface RMS step <= 0.001
mean latent RMS step  >= 0.010
latent/surface RMS movement ratio >= 10
surface-delta linear slope <= 0
latent-state norm linear slope > 0
```

The complete latent vector is the flattened recurrent state tuple itself, not a
named-channel alias and not the exposed state vector.

## Primary outputs

For each architecture report:

- number/fraction of 28 seeds passing all five conditions;
- mean and distribution of surface RMS step;
- mean and distribution of latent RMS step;
- latent/surface movement ratio;
- surface-delta slope;
- latent-norm slope;
- surface dimension / complete-state dimension.

## Secondary outputs

The historical `FIXED_POINT/accumulating_fixed_point` labels remain in the
report for comparison only. They are not part of the primary confirmation rule.

Perturbation-arm results are secondary. They cannot rescue a failed clean-arm
classification.

## Causal follow-up

Passing the five-condition rule establishes an **AFP candidate**, not causal
memory.

Any candidate promoted to a stronger claim must subsequently pass a separate
matched-future intervention:

```text
similar/controlled surface
different latent state
same future input
-> reproducibly different future trajectory
```

with state surgery or capsule controls where the architecture permits it.

## Decision rule

The study is allowed to conclude that no tested architecture confirms AFP under
this definition. Thresholds will not be loosened after seeing the held-out
results.
