# Case studies and evidence boundaries

This page is an index of the concrete experiments connected to Demian. It distinguishes the public `Demian-Substrate` v1 runtime from local experimental adapters, including the `demian_v1_trisequence_tension_v3` EEG variant. A named channel, route, or class is an implementation detail; it does not establish cognition, autonomy, medical value, or predictive superiority.

| Case | Input and transformation | Output | Evidence status | Limit |
| --- | --- | --- | --- | --- |
| [Demian-Substrate restore controls](#demian-substrate-restore-controls) | Explicit recurrent state; full or surface-only checkpoint restore | Next-state and diagnostic comparison | Public runtime source and focused tests | A control experiment, not an intelligence measure. |
| [Acoustic probe](#acoustic-probe) | Synthetic or virtual-audio features drive a fixed recurrent runtime | Derived traces and step diagnostics | Public deterministic validation record | Not a microphone, speech, or semantic-understanding result. |
| [EEG observer](#eeg-observer) | EEG-derived coupling sequence through a fixed local adapter | Observer-route comparisons under resume, shuffle, and perturbation | Sanitized aggregate from a saved local report | Not brain-state decoding, clinical validation, or a formal Lyapunov result. |
| [Wellbore Geo](#wellbore-geo) | Per-well depth-ordered log measurements become fixed-width coupling | Keyed sidecar diagnostics | Local source and tests inspected during documentation planning | No public source, Kaggle result, or predictive-lift claim. |
| [Zenith](#zenith) | Typed Red/Blue proposals in a local simulation | Referee records and replay comparisons | Local research direction only | Not public code, production qualification, or real-target performance. |

## Demian-Substrate restore controls

[Demian-Substrate](https://github.com/Aeshma-Daeva/Demian-Substrate) is the public runtime authority. Its checkpoint controls compare restoration of full internal state with restoration of a visible surface. The practical question is whether saving only the visible surface preserves the next state. Source and focused tests are public, including [`test_demian_v1_public_api.py`](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/tests/test_demian_v1_public_api.py) and [`test_demian_v1_gate_state.py`](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/tests/test_demian_v1_gate_state.py).

## Acoustic probe

The same public runtime includes deterministic offline and virtual-audio probes: non-semantic acoustic features are supplied to the recurrent state machine and derived traces are recorded. The dated [virtual-audio validation record](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/docs/VIRTUAL_AUDIO_VALIDATION_2026-09-12.md) is the relevant evidence. It validates the documented software path, not physical microphone capture or an acoustic research campaign.

## EEG observer

The local EEG experiment uses an EEG-derived sequence as input to a recurrent observer. Its useful question is whether input order, checkpoint state, and a controlled small perturbation alter the observer's internal trajectory:

`EEG-derived sequence → fixed experimental adapter → recurrent state → resume / shuffle / perturbation comparison`

The published [aggregate excerpt](../public/evidence/eeg-perturbation-summary.json) reports a historical 2026-06-23 saved run over 250 seed-0 coupling streams. It reports a mean log-growth slope of `0.0892305`, a mean bounded separation ratio of `0.000105279`, and the probe-defined regimes `bounded_sensitive` (149) and `bounded_sensitive_reconvergent` (101). These are observer diagnostics. The source implementation, data provenance, and full artifacts are local and were not rerun for this documentation update; the aggregate is not an independent replication. A finite-stream perturbation slope is not presented as a formal Lyapunov exponent or mathematical chaos.

## Wellbore Geo

Demian Geo is a local experimental adapter for well-log sequences from the ROGII wellbore-geology task. It orders measurements by depth within each well, resets recurrent state between wells, and emits auxiliary `demian_*` diagnostics that can be joined to conventional baselines by row ID.

Concrete flow: `one well's ordered measurements → conventional normalized features → fixed-width coupling → one recurrent step per row → keyed sidecar columns`.

Physical wellbore trajectory and internal observer-route trajectory are different quantities: observer-route tortuosity is not drill-path tortuosity. The local implementation and controls were inspected under `demian_geo/features.py`, `demian_geo/demian.py:generate_demian_geo_features`, `tests/test_geo_features.py:test_generate_demian_geo_features_resets_between_wells_and_preserves_rows`, and `docs/baseline_integration.md`; they are not included in this publication branch. No externally verified submission, matched held-out comparison, or non-leaky baseline result is published here. Any lift claim would require an unchanged baseline, identical split, adapter comparison, shuffled-depth control, and per-well/tail-error stability.

## Zenith

Zenith is a separate local research direction for Red/Blue simulation with private role state, typed proposals, a deterministic referee, and replay-based evaluation. A useful conceptual flow is `typed Red proposal → engine validates action → local enterprise transition → role-limited observation → independent referee record → replay fingerprint comparison`.

Its code and full evidence package are not part of this repository or a confirmed public branch. This is not a claim of production qualification, network-isolation attestation, or real-target performance.
