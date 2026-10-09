# Demian channel visibility pilot

Status: 54/54 declared cells collected. Untrained Demian v1; fixed cue/delay/query input;
three seeds (294–296), three paired directions, width 256, relative channel pulse 0.01.

| Channel | Visible cells | Mean delay | Mean peak paired surface separation |
| --- | ---: | ---: | ---: |
| fast | 9/9 | 0 | 0.03357 |
| slow | 9/9 | 1 | 0.30547 |
| control | 9/9 | 1 | 0.01308 |
| message | 9/9 | 0 | 1.06403 |
| carrier | 9/9 | 0 | 1.00140 |
| gate | 9/9 | 0 | 0.90960 |

- Observation: `slow` and `control` have no sampled direct readout response; all their
  sampled responses cross the threshold after one transition. The other four channels cross at direct readout.
- Interpretation: this is evidence of different local paths from explicit state channels to the exposed surface.
- Untested: learning, adaptation, long-horizon retention, AFP-v2, and chaos.

Evidence: local artifact `data/diagnostics/demian_channel_visibility_pilot_20261009/`;
[protocol](DEMIAN_CHANNEL_VISIBILITY_PROTOCOL.json).
