# Demian Lab

Demian Lab publishes research notes and evidence about recurrent state: what persists after interruption, which internal channels affect continuation, and how an observer changes when its input sequence is perturbed. This branch contains the research website. The runtime is maintained in [Demian-Substrate](https://github.com/Aeshma-Daeva/Demian-Substrate); the Python research workbench is on the [`refactor-current-substrate-workbench`](https://github.com/Aeshma-Daeva/Demian-Lab/tree/refactor-current-substrate-workbench) branch.

## What a reviewer can inspect

- The public Astro site and its dated research posts, feeds, and claim boundaries.
- The [Demian-Substrate runtime](https://github.com/Aeshma-Daeva/Demian-Substrate), including explicit recurrent channels, checkpoint/restore controls, and deterministic probes.
- A bounded [case-study index](docs/CASE_STUDIES.md) that separates public source from local experimental work and historical reports.

## Practical questions

- Does a full checkpoint preserve the next recurrent state differently from a surface-only restore?
- Does changing the order or a small perturbation of an EEG-derived input sequence change an observer trajectory?
- Can ordered well-log measurements be adapted into recurrent diagnostics without treating the observer route as a physical drill path?

## Evidence at a glance

| Case | Public evidence | Status | Boundary |
| --- | --- | --- | --- |
| Restore controls | [Demian-Substrate source and tests](https://github.com/Aeshma-Daeva/Demian-Substrate) | Inspectable public runtime | A control, not an intelligence benchmark. |
| Acoustic probe | [Deterministic virtual-audio validation](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/docs/VIRTUAL_AUDIO_VALIDATION_2026-09-12.md) | Dated public validation record | Software behavior; not microphone acceptance. |
| EEG observer | [Sanitized aggregate](public/evidence/eeg-perturbation-summary.json) | Historical local-report excerpt | Not clinical validation, decoding, or independent replication. |
| Wellbore Geo | [Case boundary](docs/CASE_STUDIES.md#wellbore-geo) | Local experimental adapter | No predictive-lift or public-submission claim. |
| Zenith | [Case boundary](docs/CASE_STUDIES.md#zenith) | Local research direction | Not public source, production qualification, or real-target performance. |

## Run the site locally

```bash
npm install
npm run dev
```

Build the deployable site with:

```bash
npm run build
```

## Repository boundary

This repository is the publication site, not a distribution of every experiment referenced here. EEG and wellbore adapters, their datasets, and full artifacts remain local. The aggregate EEG excerpt preserves only documented summary values and its source digest. The research branch is linked above; its history and findings should be read as research material, not as a release of this website branch.

## Further reading

- [Case studies](docs/CASE_STUDIES.md)
- [Research workbench claims](https://github.com/Aeshma-Daeva/Demian-Lab/blob/refactor-current-substrate-workbench/docs/CLAIMS.md)
- [Public blackboard](https://aeshma-daeva.github.io/Demian-Lab/blackboard)
- [JSON updates feed](https://aeshma-daeva.github.io/Demian-Lab/updates.json)
- [RSS feed](https://aeshma-daeva.github.io/Demian-Lab/rss.xml)

## Publishing notes

Posts live in `src/content/posts/`. The public blackboard is at `/blackboard`, while `/updates.json` and `/rss.xml` are feed endpoints. Deployment is performed by the existing GitHub Actions workflow when a maintainer chooses to publish; this documentation update does not change deployment settings.
