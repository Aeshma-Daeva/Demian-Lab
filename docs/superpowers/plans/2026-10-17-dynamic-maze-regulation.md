# Dynamic Maze and Experience-Dependent Regulation Plan

**Goal:** Determine whether experience improves Demian's regulation of internal
processing and environmental behavior under changing conditions.
**Architecture:** First characterize frozen Demian in a reproducible dynamic
maze. Subsequently compare an explicit internal routing-learning mechanism
against matched frozen and disrupted-experience controls in a versioned branch.
**Stack:** Existing Python/NumPy/PyTorch and pytest; no simulation download required.
**Spec:** User's 2026-10-17 research discussion; `docs/EXPERIMENT_RULES.md`.
**Execution:** Native implementation; independent scientific review at package
boundaries. Routine verification and runs need no approval checkpoint.

## Constraints

- First package preserves Demian architecture and weights. New world encoder
  and action interface are explicitly versioned; do not overwrite cue-task ones.
- Diagnostic/readout fitting conditions are recorded separately. External action
  head fitting does not establish learning inside Demian.
- Environment/connector executes actions; it supplies no path planner, hidden
  state, door clock, global tick or external memory to Demian.
- Later learning variables and update rules must be specified before their run.
  Keep experience inside the system; investigate slow/gate traces as candidates,
  without assigning channel functions by name alone.
- Distinguish navigation, feedback regulation and improvement through experience.
- Raw geometry, gate changes and criterion passage alone establish none of these.
- Influence budgets and actual computation budgets are separate experiments.
- Preserve existing untracked experiments. Registry status remains manually reviewed.

## Scientific sequence

| Phase | Deliverable | Required evidence |
| --- | --- | --- |
| 1 World | Small dynamic maze, snapshots and valid actions | Exact replay, hidden-state isolation, solvability, deterministic matched disturbance tapes |
| 2 Frozen baseline | Closed-loop trajectories and disclosed action-head fitting | Performance relative to random and observation-only policies; recurrence contribution measured |
| 3 Internal organization | Channel/route timing around decisions and disturbances | Fresh histories predict outcomes beyond observable task state, recent actions, time, norms and seed identity |
| 4 Internal learning | Explicit persistent routing variables, update rule and bounded influence budget | Experience improves later held-out probes relative to matched frozen learning and reset controls |
| 5 Causality | Selective trace/route interventions and restore | Predicted behavioral loss and rescue beyond generic perturbation |
| 6 Transfer | New layouts, schedules and interference | Benefit persists under a frozen learning rule and equal interaction budgets |

Failure at a phase remains evidence. Diagnose the failed condition; do not
substitute a different success metric or introduce routing as a repair by default.

## First package: world and frozen baseline

World: bounded7×7 grid with two routes to a visible task goal, one intermittent
door on the shorter route and a reliable detour. Layout, start, goal, door cell
and full event tape are saved per episode. Initial template must have detour
length greater than the open-door shortest path; validate by oracle search.

Observations: egocentric3×3 occupancy, locally visible door state, goal bearing
`(sign(goal_x-x), sign(goal_y-y))`, energy remaining and previous-action acknowledgement. No full map,
door phase, hidden target, score accumulator or experiment condition flag.
Energy reveals within-episode elapsed cost; record this timing proxy explicitly.
Randomize initial door phase on saved tapes; energy must not uniquely reveal it.
Actions: north/south/east/west/wait. One environment action per recurrent tick;
blocked moves consume the same movement unit as successful moves. Wait also
consumes one unit. Budget96units per episode; goal or exhaustion terminates.
Each tick observes the current tape-defined door state, executes one action,
then advances the world clock. A closed door blocks entry; an occupant may
leave after closure. Reject layouts without a route when the door is closed.

Conditions: static-open reference, static-closed reference, recurring door
schedule, and one schedule change at tick2 within an episode. Initial fit/evaluation
layouts use disjoint D4 orientations; broader topology transfer remains later.
Tick2 can occur after direct-route entry, so elapsed switch time alone does not
establish an experienced obstruction. Log actual visibility and observed changes.
Require a validated pair of equal-length observation histories reaching the
same junction with identical current observations, energy and acknowledgement,
but different earlier observed door conditions and different optimal route
choices. If the template lacks such a case, it cannot test useful memory.

Primary outcomes: goal success within96ticks and paired goal-reaching cost;
report failures at full budget rather than dropping them from cost comparisons.
Secondary: blocked moves, recovery after closure, repeated mistakes and route
choice. Plot these alongside compact channel/route summaries, not raw vectors.

Fitting: collect teacher trajectories on development layouts only. Fit one
stateless ridge action head on frozen Demian states per parameter seed, alpha1;
freeze it before evaluation. Record normalization, coefficients, labels, sample
counts and training layouts. Fit an observation-only head on identical labels.
Each confirmation seed receives its own head fit on the same permitted training
layouts; fresh seeds do not imply a shared decoder or zero-shot policy transfer.
Teacher may know the static map and observed history, but no future door tape
or currently hidden door state. Identical complete observation histories must
not acquire different labels through privileged information. A separate oracle
may use privileged state for solvability/reference scores only. No test-layout fitting.
Teacher waits once at a visibly closed entry before detouring, supplying actual
wait targets. Forced training coverage steps retain the desired teacher labels.

Pilot: width16 smoke checks, then width256; four development seeds84001–84004.
Reserve six fresh confirmation seeds85001–85006, not used to revise the pilot.
Use24total episodes per policy per seed: six per condition,96ticks maximum.
Analyze paired contrasts at parameter-seed level. Full Demian state and connector
reset at each episode start; no between-episode learning in the frozen baseline.
Freeze exact layouts, episode allocation,
schedule tapes and effect criterion before collecting scientific outcomes.
Benchmark pilot wall time and storage before any larger campaign.

## Implementation tasks

- [x] **World:** create `development/demian_maze_world.py` and
  `tests/test_demian_maze_world.py`. Interfaces: `MazeConfig`, `MazeObservation`,
  `MazeAction`, `MazeSnapshot`; `reset(config)`, `observe()`, `step(action)`,
  `snapshot()`, `restore(snapshot)`. Test motion, blocked/wait cost, termination,
  door timing, exact replay and invalid configuration/action rejection.
- [x] **Boundary:** create `development/demian_maze_runner.py` and
  `tests/test_demian_maze_runner.py`. Consume world observations; produce
  `MazeStepRecord` with observation, action, acknowledgement, state and route
  trace. Add stateless encoder and independently fitted/frozen action heads.
  Test no privileged-field influence, fitting isolation, saved-fit prediction
  replay, one-action timing and unchanged Demian parameter hashes.
- [x] **Frozen controls:** intact recurrence, per-tick recurrence reset,
  observation-only head, random policy and prerecorded-observation replay.
  Test acute resets with the intact head and a separately fitted reset-state
  head trained on the same allowed trajectories/labels and fitting budget.
  Attribute recurrence benefits using the latter matched-fit comparison;
  acute reset failure alone may reflect action-head distribution shift.
  Reset is a recurrence/state control, not a learning-ablation result.
  Replay validates trajectories; it cannot supply a fair navigation score once
  actions and observations become inconsistent. Match exogenous world tapes;
  closed-loop trajectories may diverge because actions differ.
- [x] **Protocol/run:** create `docs/DEMIAN_DYNAMIC_MAZE_PROTOCOL.json` and
  `development/run_demian_maze_baseline.py`. Verify seed novelty, fit/eval layout
  disjointness and snapshot completeness before execution. Save states,
  controller fits, world tapes, source/dependencies, predictions and hashes.
- [x] **Verification:** run `./venv/bin/python -m pytest -q
  tests/test_demian_maze_world.py tests/test_demian_maze_runner.py`, then the
  existing closed-loop regression tests. Preserve and explain failures.
- [x] **Review/record:** Astra reviews leakage, action-head contribution,
  controls and conclusions. Register completed outcomes and artifact pins;
  append labbook, update working state and index. No automatic status promotion.

## Later learning experiment: freeze before implementation

Specify the persistent internal variables, their owner, update equation,
feedback signal, eligible routes, budget and credit-assignment horizon. Keep
the action head fixed during this comparison. Separate task state from retained
experience so their resets and transplants have explicit meanings.
For initial eligible additive routes, distinguish a coefficient constraint
`g_r>=0, sum(g_r)<=B_gain` from an effective injection constraint
`sum(||g_r*v_r||²)<=B_injection²`, where `v_r` is the implemented route
contribution. Freeze bounds/reference scaling before results. Neither is a
compute budget; do not infer equal effective influence from equal gain sums.

Compare: learning enabled, identical controller with learning disabled,
experience-variable reset, shuffled feedback, replayed/yoked allocation and
matched random allocation. Evaluation probes begin from matched world/task
conditions; freeze learning during probes to measure acquired competence.
Equalize exposure and action budgets. Paired exogenous tapes remain fixed.
Different closed-loop policies encounter different observations despite equal
budgets. Also compare update rules under a common controlled experience stream,
then probe them interactively. Specify exposure/probe lengths, schedule change,
ordinary task-state reset and retained-variable boundaries before that run.

Learning gate: prior experience improves predeclared held-out behavioral
outcomes across fresh seeds relative to learning-disabled and disrupted-
experience controls. Transplant only retained learning variables between matched
controllers to test loss/rescue; do not restore an entire warmed-up controller.
A decrease
in energy cost must preserve success; fewer operations require actual skipped
computation, not merely smaller route gains.

Only after this baseline compare local processing schedules or changing
connectivity. Preserve named channels as measurement compartments while testing
whether their functional roles and relationships vary with experience.
