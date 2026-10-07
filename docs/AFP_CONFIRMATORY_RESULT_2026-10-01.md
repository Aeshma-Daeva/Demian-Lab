# AFP Held-Out Confirmation — 2026-10-01

Status: held-out confirmation completed after the criterion in
`AFP_CONFIRMATORY_PREREG_2026-10-01.md` was committed.

## Frozen rule

A clean run was counted as an AFP candidate only if all five preregistered
conditions held over the final 96 steps:

```text
mean surface RMS step <= 0.001
mean latent RMS step  >= 0.010
latent/surface RMS movement ratio >= 10
surface-delta linear slope <= 0
latent-state norm linear slope > 0
```

Exploratory seeds 94–101 were excluded. Confirmation used fresh seeds 102–129
(28 model/initialization seeds).

## Result

| Variant | Clean AFP candidates | Perturbed AFP candidates |
| --- | ---: | ---: |
| native v2 | 0 / 28 | 0 / 28 |
| native v3 | 1 / 28 | 0 / 28 |
| native v8 | 1 / 28 | 0 / 28 |
| native v9 | **4 / 28** | **3 / 28** |
| v9 five-channel default | 0 / 28 | 0 / 28 |
| v9 five-channel accumulator | 0 / 28 | 0 / 28 |
| v9 five-channel accumulator, zero-init | 0 / 28 | 0 / 28 |
| Demian v1 | 0 / 28 | 0 / 28 |

native v3 and native v8 produce the same trajectory under the tested default
configuration for the passing seed, so their seed-126 result should not be
counted as two independent confirmations.

The strongest independent concentration is therefore in **canonical native
v9**.

## Confirmed native-v9 clean candidates

| Seed | Surface RMS step | Latent RMS step | Latent/surface | Surface-delta slope | Latent-norm slope |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 104 | 0.000683 | 0.016713 | 24.46x | -0.000018 | +0.002782 |
| 112 | 0.000184 | 0.010278 | 55.85x | -0.000006 | +0.000106 |
| 123 | 0.000888 | 0.090557 | 102.00x | -0.000021 | +0.014991 |
| 126 | 0.000994 | 0.106461 | 107.12x | -0.000006 | +0.012931 |

Under the perturbation arm, seeds 104, 123 and 126 still passed. Seed 112 did
not.

## What this supports

Under the preregistered **operational** definition, held-out native-v9 runs can
enter a regime with all of the following simultaneously:

1. very low exposed-surface movement;
2. substantially larger complete-state movement;
3. a decreasing/stable surface-motion trend;
4. increasing latent-state norm.

That is substantially stronger evidence than the historical heuristic label.

It supports calling these runs **operational AFP candidates** or
**surface-quiescent latent-accumulating regimes**.

It does not prove a mathematical full-state fixed point, because by definition
the full state continues to move.

## What this falsified or narrowed

The held-out run did **not** show the preregistered AFP regime in:

- native v2;
- the available public v9 five-channel scaffolds;
- Demian v1.

Therefore AFP should not be described as a universal property of Demian.

Demian v1 remains better described, on current evidence, by latent-path
dependence and strong surface/latent separation while its exposed surface is
still dynamically active.

The exact historical C16 evolved five-channel candidates are still unavailable
in the public workbench, so the failure of the public v9 five-channel scaffold
does not constitute a direct rerun of C16.

## Historical-classifier comparison

The old classifier labelled **28/28** held-out runs in **every architecture** as
`accumulating_fixed_point`.

The preregistered direct rule selected only a small architecture/seed subset.

This demonstrates that the old label is useful as historical provenance but is
not specific enough for the present research claim.

## Next question

AFP presence is now a measurement result. The remaining stronger question is
causal:

> when native v9 exposes essentially the same current surface, does the latent
> state that is still moving determine what the surface does next?

The next experiment therefore performs candidate-specific latent-state surgery
while keeping the native-v9 exposed `fast` surface exactly unchanged.
