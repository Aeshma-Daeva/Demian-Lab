# Case studies and evidence boundaries

This page is an index of the concrete experiments connected to Demian. It distinguishes the public `Demian-Substrate` v1 runtime from local experimental adapters, including the `demian_v1_trisequence_tension_v3` EEG variant. A named channel, route, or class is an implementation detail; it does not establish cognition, autonomy, medical value, or predictive superiority.

| Case | Input and transformation | Output | Evidence status | Limit |
| --- | --- | --- | --- | --- |
| [Demian-Substrate restore controls](#demian-substrate-restore-controls) | Explicit recurrent state; full or surface-only checkpoint restore | Next-state and diagnostic comparison | Public runtime source and focused tests | A control experiment, not an intelligence measure. |
| [Acoustic probe](#acoustic-probe) | Synthetic or virtual-audio features drive a fixed recurrent runtime | Derived traces and step diagnostics | Public deterministic validation record | Not a microphone, speech, or semantic-understanding result. |
| [EEG observer](#eeg-observer) | EEG-like coupling sequence through a fixed adapter | Order and restore comparisons; historical perturbation aggregate | Public synthetic source plus sanitized saved-report aggregate | Not brain-state decoding, clinical validation, or a formal Lyapunov result. |
| [Wellbore Geo](#wellbore-geo) | Per-well depth-ordered log measurements become fixed-width coupling | Keyed recurrent diagnostics | Public synthetic source and tests | No Kaggle result or predictive-lift claim. |
| [Zenith](#zenith) | Evidence and validity events update a typed epistemic position | Belief, justification, tension, history, and action-authority transitions | Public minimal extraction and tests | Not a complete agent, truth-discovery system, or production qualification. |

## Demian-Substrate restore controls

[Demian-Substrate](https://github.com/Aeshma-Daeva/Demian-Substrate) is the public runtime authority. Its checkpoint controls compare restoration of full internal state with restoration of a visible surface. The practical question is whether saving only the visible surface preserves the next state. Source and focused tests are public, including [`test_demian_v1_public_api.py`](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/tests/test_demian_v1_public_api.py) and [`test_demian_v1_gate_state.py`](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/tests/test_demian_v1_gate_state.py).

## Acoustic probe

The same public runtime includes deterministic offline and virtual-audio probes: non-semantic acoustic features are supplied to the recurrent state machine and derived traces are recorded. The dated [virtual-audio validation record](https://github.com/Aeshma-Daeva/Demian-Substrate/blob/main/docs/VIRTUAL_AUDIO_VALIDATION_2026-09-12.md) is the relevant evidence. It validates the documented software path, not physical microphone capture or an acoustic research campaign.

## EEG observer

The public [Demian-EEG](https://github.com/Aeshma-Daeva/Demian-EEG) extraction uses a deterministic synthetic EEG-like sequence as input to a recurrent observer. Its useful question is whether input order and checkpoint completeness alter the observer's internal trajectory:

`EEG-derived sequence → fixed experimental adapter → recurrent state → resume / shuffle / perturbation comparison`

The published [aggregate excerpt](../public/evidence/eeg-perturbation-summary.json) reports a historical 2026-06-23 saved run over 250 seed-0 coupling streams. It reports a mean log-growth slope of `0.0892305`, a mean bounded separation ratio of `0.000105279`, and the probe-defined regimes `bounded_sensitive` (149) and `bounded_sensitive_reconvergent` (101). These are observer diagnostics. The public repository makes the core order/restore mechanism runnable but does not reproduce that raw-data campaign; the aggregate is not an independent replication. A finite-stream perturbation slope is not presented as a formal Lyapunov exponent or mathematical chaos.

## Wellbore Geo

[Demian-Geo](https://github.com/Aeshma-Daeva/Demian-Geo) is a public experimental adapter for well-log sequences. It orders measurements by depth within each well, resets recurrent state between wells, and emits auxiliary `demian_*` diagnostics that can be joined to conventional baselines by row ID.

Concrete flow: `one well's ordered measurements → conventional normalized features → fixed-width coupling → one recurrent step per row → keyed sidecar columns`.

Physical wellbore trajectory and internal observer-route trajectory are different quantities: observer-route tortuosity is not drill-path tortuosity. The public repository includes the adapter, synthetic wells, focused tests, and a group-aware comparison helper. It excludes private competition files and does not publish an externally verified submission or predictive-lift result. Any lift claim would require an unchanged baseline, identical split, adapter comparison, shuffled-depth control, and per-well/tail-error stability.

## Zenith

[Zenith Epistemic Runtime](https://github.com/Aeshma-Daeva/Zenith-Epistemic-Runtime) extracts the domain-neutral state machine that proved most relevant to this research line: `evidence reference → justification standing → tension state → authority assessment → commitment or abstention`.

The extraction distinguishes evidence expiry from contradiction, retains append-only transition history, rejects non-monotonic updates, protects checkpoints with a digest, and prevents a denied authority assessment from becoming a selected commitment. The broader Red/Blue simulator and operational Abraxas material remain private. This is not a claim of truth discovery, complete agency, production qualification, or real-target performance.
