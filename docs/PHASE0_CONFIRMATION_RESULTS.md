# Phase 0 confirmation

Status: post-run audit passed; Phase 0b development calibration admitted.
Untrained, state-budget-matched models; fixed symbolic cue/delay/query drive;
float64, 128-tick burn-in, 100 fresh seeds (194–293), three directions per seed.

| Model | 384-tick mean rate | 1920-tick mean rate | Sampled classification |
| --- | ---: | ---: | --- |
| RNN | −0.53480 | −0.53052 | Contracting: 100/100 seeds |
| GRU | −0.44397 | −0.44032 | Contracting: 100/100 seeds |
| Gate-disabled Demian | +0.03801 | +0.03784 | Expanding: 100/100 seeds |
| Demian | +0.03904 | +0.03813 | Expanding: 100/100 seeds |

All 2400 unique cells passed precision, finiteness, operating-point coverage,
state-budget and frozen direction/horizon checks. Archived source snapshots,
original development decision and test-log digest were verified.
Directions share a seed trajectory; they are not independent population samples.

- Observation: default RNN/GRU and Demian occupy different sampled regimes.
- Control: expansion also occurs without gate modulation.
- Interpretation: default-baseline persistence differences do not isolate architectural effects.
- Unestablished: asymptotic chaos, AFP-v2, memory, channel roles, architectural superiority.
- Next: recurrent-only RNN gain calibration on seeds 94–96; fresh confirmation required.

Evidence: `regime_phase0_frozen_h256_s100_20261008_v2` under `data/diagnostics/`;
[decision](PHASE0_DECISION.json), [gain protocol](REGIME_GAIN_PROTOCOL.json).
