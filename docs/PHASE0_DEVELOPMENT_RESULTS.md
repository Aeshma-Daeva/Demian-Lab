# Phase 0 development evidence

Status: exploratory directional measurements; confirmation pending.
Untrained models, common symbolic cue/delay/query drive, noop acknowledgements,
state-budget matching (1312 float64 elements), reference Demian width 256.
Seeds 94–96, directions 1701–1703, burn-in 128. Rates are per tick.

| System | 384 measured ticks | 1920 measured ticks | Observation |
| --- | ---: | ---: | --- |
| RNN | −0.53177 | −0.52761 | Sampled contraction |
| GRU | −0.44459 | −0.44132 | Sampled contraction |
| Demian gate-disabled control | +0.03715 | +0.03849 | Sampled expansion |
| Demian | +0.04004 | +0.03867 | Sampled expansion |

72/72 records passed endpoint finite-difference checks and state/tangent finiteness.
Not independent population estimates: directions share each model trajectory.
These are not largest-singular-value FTLEs, asymptotic exponents or chaos diagnoses.

- Observation: sampled regime separation persists at the longer horizon.
- Control: gate-disabled Demian also expands; gate-specific advantage is unestablished.
- Hypothesis: regime matching may change persistence comparisons; untested.
- AFP-v2: unconfirmed; neither surface convergence nor functional latent state is established here.
- Plasticity: deferred until regime and visibility/decodability gates.

Pilot directories: `regime_phase0_development_h256_t512_20261008` and
`regime_phase0_development_h256_t2048_20261008`, under `data/diagnostics/`.
These pilots predate exact source snapshots. Renewed development validation is required;
the frozen campaign archives source, protocol, tests, audit and decision hashes.
