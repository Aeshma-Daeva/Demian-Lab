# Capsule Continuity

Last updated: 2026-05-11

## Purpose

This document records two continuation controls:

- `full_capsule` versus `body_surface`: model weights are restored in both
  arms; the comparison isolates continuation relevance of omitted hidden state.
- `full_capsule` versus `surface_only`: the latter uses a fresh body and
  empty hidden state; the comparison tests the body-plus-state package and
  cannot isolate hidden-state causality.

Neither control establishes AFP-v2 or a novel mechanism.

## Current Result

The first v9 replication probe is:

- code: [development/probe_v9_capsule_continuity.py](/home/xenith/demian/development/probe_v9_capsule_continuity.py)
- test: [tests/test_v9_capsule_continuity.py](/home/xenith/demian/tests/test_v9_capsule_continuity.py)
- artifact: [data/substrate_lab/v9_capsule_continuity_20260511/summary.json](/home/xenith/demian/data/substrate_lab/v9_capsule_continuity_20260511/summary.json)

It compares uninterrupted continuation with:

- full capsule: restored model body and full internal state
- body + surface: restored model body, empty hidden state, exposed surface rewritten
- surface-only: fresh model body, empty hidden state, exposed surface rewritten
- body-only: restored model body with empty state
- component-only: one internal component restored into an empty state

## Compact Readout

![Capsule continuity sweep](assets/capsule_continuity.svg)

| Substrate | Full capsule cosine | Body + surface cosine | Surface-only cosine |
| --- | ---: | ---: | ---: |
| `demian_native_v9` | `0.99999994` | `0.96098512` | `0.09100710` |
| `v9_five_channel` | `1.0` | `0.75564301` | `-0.03007574` |

Mean trajectory gap also separates the arms:

| Substrate | Full capsule gap | Body + surface gap | Surface-only gap |
| --- | ---: | ---: | ---: |
| `demian_native_v9` | `0.0` | `0.10775166` | `0.25749409` |
| `v9_five_channel` | `0.0` | `0.24032078` | `0.30981059` |

The tables above are the representative seed-94, 24:24 run. The current smoke sweep uses seeds `94,95,96` and pause/resume windows `16:16`
and `24:24`. In all 6 runs per substrate, full capsule resume remains exact
or near-exact (`min cosine > 0.999`, `max mean gap = 0.0`), and surface-only
replay is worse than full capsule resume.

## Interpretation

The representative fixed-body comparison shows that omitted hidden state
changes continuation. The broader `surface_only` comparison also changes the
model body and therefore tests package portability, not hidden-state causality.

The saved sweep does not aggregate `body_surface`; sweep-wide fixed-body
confirmation remains open. These controls do not establish AFP-v2, compression,
or a distinct stability regime.

## Next Checks

- aggregate `body_surface` across seeds and longer pause/resume windows
- test compressed capsules, not only full internal state restore
- test whether `slow`, `message`, and `carrier` can be reduced to a small
  structured code while preserving resume quality
- keep this separate from v7.4 until the v9/v9-five-channel behavior is stable

## Reproduce

```bash
./venv/bin/python development/probe_v9_capsule_continuity.py --hidden-size 16 --pause-steps 24 --resume-steps 24 --seeds 94,95,96 --windows 16:16,24:24
./venv/bin/python -m pytest tests/test_v9_capsule_continuity.py -q
```
