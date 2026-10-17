# Dynamic-maze baseline — 2026-10-17

Status: execution and independent Astra review complete. No architecture,
encoder or weight changes; only external ridge action heads were fitted.

## Protocol

Seven-by-seven partially observed world; local occupancy, goal bearing, energy
and action acknowledgements. Five actions: cardinal moves and wait. Open,
closed, periodic and switched door conditions; 96-action budget. Direct and
safe-detour paths take four and eight actions. Door tapes are matched across
policies. Switch onset is tick2; elapsed switch is not necessarily an observed
obstruction. Runtime resets between episodes; no experience-dependent learning.

Width256; four development seeds and six reserved confirmation seeds.
Teacher-imitation fits use even D4 orientations; evaluation uses odd orientations.
Each seed receives24 worlds under five policies:1,200 evaluation episodes,
55,480 ticks. Worlds/checkpoints repeat within seeds and are not independent
replications. Protocol: `DEMIAN_DYNAMIC_MAZE_PROTOCOL.json`.

## Observation

Confirmation means across six parameter seeds; failures cost96 actions:

| Policy | Success | Cost |
| --- | ---: | ---: |
| Recurrent, fitted head |58.3%|51.54|
| Per-tick reset, separately fitted head |100%|8.00|
| Acute reset, intact recurrent head |2.1%|95.48|
| Current observation only |100%|8.00|
| Random |58.3%|69.65|

The frozen recurrence-advantage gate fails in all four development and all six
confirmation seeds. Confirmation recurrent success ranges25–91.7%; acute reset
0–12.5%. Acute-reset failure alone is not useful-memory evidence: separately
fitting a reset-state head removes that performance deficit.

A prespecified post-pilot diagnostic reuses the four development seeds and
unchanged fitted heads/tapes, replacing held-out orientations with familiar
ones. Recurrent and matched-reset policies both reach100% success at6.458
actions; observation-only reaches100% at6.708. This is not fresh-seed confirmation.

The scripted matched-observation history probe produces distinct recurrent
states, but neither history pair receives both correct actions from the fitted
heads in any seed. Those return-to-junction histories are outside the teacher
curriculum. State-history influence is distinct from demonstrated useful recall.

## Interpretation and limits

- The whole encoder–dynamics–action-head pipeline has orientation-dependent
  portability. This does not isolate the head or establish memory erasure.
- The safe detour solves every evaluated world without recurrence. At100%
  control success, the declared additional ten-point success gain is impossible.
  Preserve the failed gate; cost-only or memory-essential tests need new protocols.
- Familiar-orientation recovery does not establish recurrence advantage.
- Route norms are instrumentation, not route causality. No adaptation, learning,
  smart routing, AFP or architecture-ranking claim follows.

## Integrity and preservation

Astra independently verified238 baseline manifest entries,30 frozen heads,
teacher logits,38,710 saved policy actions and aggregate gates. Twelve selected
episodes reproduced694 full states/public vectors bitwise, with exact world
acknowledgements; norm summaries differed by at most6.1e-5. The diagnostic's43
manifest entries, unchanged heads, paired tapes and1,884 actions also verified.
No invalidating implementation bug found; review returned, not pending.

Verification:16 maze tests and56 focused tests pass. Full suite:579 pass,
four failures from absent historical v10/evolution artifacts; unrelated to maze.

Artifacts: `data/diagnostics/demian_dynamic_maze_baseline_v1_20261017/`
and `data/diagnostics/demian_maze_orientation_diagnostic_v1_20261017/`.
Compact metrics, protocols, heads, logits, probes, source and manifests are
versioned. Bulky raw states, parameters, traces and training matrices remain
local and hash-listed; compact checkout alone is not the full raw archive.

## Open scientific decision

Before changing internal routing, separate action/orientation transfer from
history usefulness. A new task must make remembered observations consequential
for declared cost or success, with matched memoryless controls. The current
world validates a closed loop, not pressure to learn internal regulation.
