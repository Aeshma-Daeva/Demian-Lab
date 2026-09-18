---
title: "Lyapunov-style perturbation probe: v3 results"
date: 2026-06-23
description: "A saved 250-stream run measures the observer's response to a small input perturbation."
tags: ["perturbation", "v3", "eeg", "probe"]
metrics:
  - label: "recordings"
    value: 250
  - label: "positive_growth_fraction"
    value: 1.0
  - label: "bounded_sensitive_reconvergent"
    value: 101
artifacts:
  - label: "Published aggregate summary"
    path: "evidence/eeg-perturbation-summary.json"
---

## Scope

This saved run measures the observer's response to a small input perturbation. The source implementation, scripts, and full artifacts remain local; this public post links only the bounded aggregate summary.

## Historical saved-run results

The 2026-06-23 report records 250 saved seed-0 coupling streams:

| metric | value |
|---|---|
| recordings | 250 |
| mean_log_growth_slope | 0.0892305 |
| positive_growth_fraction | 1.0 |
| mean_bounded_separation_ratio | 0.000105279 |
| bounded_sensitive_fraction | 1.0 |
| bounded_sensitive | 149 |
| bounded_sensitive_reconvergent | 101 |

## Published artifact

- [Sanitized aggregate summary](/Demian-Lab/evidence/eeg-perturbation-summary.json)

## Interpretation

In this saved probe, a tiny first-step perturbation grows across the observer route in every sampled recording while remaining bounded relative to that route's radius. In 101/250 recordings it later reconverges.

The report is an observer diagnostic for this fixed experiment. It does not establish a formal Lyapunov exponent or mathematical chaos.

Not claiming: brain-state decoding, clinical validity, or biomarkers.

## Verification

```
Historical verification recorded with the 2026-06-23 run: 60 tests passed in 10.35s. This suite was not rerun for this documentation update.
```
