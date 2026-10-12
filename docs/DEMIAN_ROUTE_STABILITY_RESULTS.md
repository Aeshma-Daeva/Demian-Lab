# Demian fixed-forcing route-stability v1

Status: 144/144 declared cells collected. Untrained Demian v1; fixed
cue/delay/query input with noop acknowledgements; width 256. Four development
and four held-out parameter seeds; carrier pulse at ticks 32 and 96; amplitudes
0.003, 0.01, and 0.03; three directions.

| Split | Arm | Mean final full-state separation | Mean peak surface separation |
| --- | --- | ---: | ---: |
| Development | lesion-only | 264.97 | 2.28 |
| Development | pulse + lesion | 268.05 | 2.34 |
| Development | pulse + sham | 65.26 | 0.85 |
| Development | random-direction pulse | 71.75 | 0.89 |
| Held-out | lesion-only | 254.23 | 2.28 |
| Held-out | pulse + lesion | 258.71 | 2.32 |
| Held-out | pulse + sham | 66.45 | 0.80 |
| Held-out | random-direction pulse | 62.83 | 0.81 |

- Observation: gain-zero `carrier_to_slow` lesion changes the subsequent full-state
  trajectory strongly under fixed forcing. The result replicates in held-out seeds.
- Control: lesion-only is already near pulse-plus-lesion. The lesion itself explains
  most of that branch's separation.
- Interpretation: `carrier_to_slow` is dynamically consequential in this regime.
- Untested: route-specific mediation of a carrier pulse; memory, AFP-v2, learning,
  adaptation, and chaos.
- Recovery: no arm reached the declared threshold for four consecutive sampled
  transitions within 128 steps. These windows are censored, not an AFP decision.

Evidence: [protocol](DEMIAN_ROUTE_STABILITY_PROTOCOL.json),
[summary](../data/diagnostics/demian_route_stability_v1_20261011/summary.json).
