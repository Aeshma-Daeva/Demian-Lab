# Demian fixed-forcing route-interaction v1

Status: 96/96 declared cells collected. Untrained Demian v1; fixed
cue/delay/query input with noop acknowledgements; width 256. Four development
and four held-out parameter seeds; carrier pulse at ticks 32 and 96; route
multiplier sweep 1.0 to 0.0.

| Split | Multiplier | Mean peak state interaction |
| --- | ---: | ---: |
| Development | 1.00 | 0.00 |
| Development | 0.90 | 30.81 |
| Development | 0.75 | 44.94 |
| Development | 0.50 | 49.24 |
| Development | 0.25 | 55.81 |
| Development | 0.00 | 48.59 |
| Held-out | 1.00 | 0.00 |
| Held-out | 0.90 | 56.64 |
| Held-out | 0.75 | 62.98 |
| Held-out | 0.50 | 69.09 |
| Held-out | 0.25 | 69.86 |
| Held-out | 0.00 | 71.50 |

- Observation: reducing `carrier_to_slow` changes the carrier-pulse response
  after subtraction of the route-only trajectory effect.
- Control: an intact multiplier is zero by construction; route-only branches
  remain separate from pulse branches.
- Interpretation: this supports route-sensitive internal propagation under
  fixed forcing.
- Untested: carrier specificity, route-mediated memory, AFP-v2, learning,
  adaptation, and chaos. A matched random-direction gain sweep is next.

Evidence: [protocol](DEMIAN_ROUTE_INTERACTION_PROTOCOL.json),
[summary](../data/diagnostics/demian_route_interaction_v1_20261011/summary.json).
