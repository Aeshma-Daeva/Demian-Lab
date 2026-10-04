# Capsule Continuity

Last updated: 2026-05-11

## Purpose

This is a continuation-state sufficiency control. It asks whether the exposed
readout is sufficient to resume a paused deterministic trajectory.

The full capsule restores the full internal tensor state. The surface-only arm
starts from a zeroed internal state with only the exposed surface rewritten.
Because the full restore contains more information, divergence is expected.
The control does not establish AFP dynamics or a novel mechanism.

## Current Result

The first v9 replication probe is:

- code: [development/probe_v9_capsule_continuity.py](/home/xenith/demian/development/probe_v9_capsule_continuity.py)
- test: [tests/test_v9_capsule_continuity.py](/home/xenith/demian/tests/test_v9_capsule_continuity.py)
- artifact: [data/substrate_lab/v9_capsule_continuity_20260511/summary.json](/home/xenith/demian/data/substrate_lab/v9_capsule_continuity_20260511/summary.json)

It compares uninterrupted continuation with:

- full capsule: restored model body and full internal state
- surface-only: empty state with only the exposed surface rewritten
- body-only: restored model body with empty state
- component-only: one internal component restored into an empty state

## Compact Readout

![Capsule continuity sweep](assets/capsule_continuity.svg)

| Substrate | Full capsule cosine | Surface-only cosine | Best component-only read |
| --- | ---: | ---: | --- |
| `demian_native_v9` | `0.99999994` | `0.09100710` | `slow_only`: `0.99973315` |
| `v9_five_channel` | `1.0` | `-0.03007574` | partial continuity spreads across `carrier`, `message`, and `slow` |

Mean trajectory gap also separates the arms:

| Substrate | Full capsule gap | Surface-only gap |
| --- | ---: | ---: |
| `demian_native_v9` | `0.0` | `0.25749409` |
| `v9_five_channel` | `0.0` | `0.30981059` |

The current smoke sweep uses seeds `94,95,96` and pause/resume windows `16:16`
and `24:24`. In all 6 runs per substrate, full capsule resume remains exact
or near-exact (`min cosine > 0.999`, `max mean gap = 0.0`), and surface-only
replay is worse than full capsule resume.

## Interpretation

The exposed surface is not a sufficient continuation state in these probes.
The result is a sanity/control result, not the main discovery. It does not show
that every hidden variable is meaningful, that the capsule is compressed, or
that the observed internal dynamics form a distinct stability regime.

## Next Checks

- repeat across seeds and longer pause/resume windows
- test compressed capsules, not only full internal state restore
- test whether `slow`, `message`, and `carrier` can be reduced to a small
  structured code while preserving resume quality
- keep this separate from v7.4 until the v9/v9-five-channel behavior is stable

## Reproduce

```bash
./venv/bin/python development/probe_v9_capsule_continuity.py --hidden-size 16 --pause-steps 24 --resume-steps 24 --seeds 94,95,96 --windows 16:16,24:24
./venv/bin/python -m pytest tests/test_v9_capsule_continuity.py -q
```
