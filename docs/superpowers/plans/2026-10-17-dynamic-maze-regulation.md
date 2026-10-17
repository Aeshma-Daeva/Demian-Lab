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
| 3 Internal organization | Channel/route timing around decisions and disturbances | Fresh histories predict outcomes beyond elapsed time, norms and seed identity |
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

Observations: egocentric3×3 occupancy, locally visible door state, declared goal
bearing, energy remaining and previous-action acknowledgement. No full map,
door phase, hidden target, score accumulator or experiment condition flag.
Energy reveals within-episode elapsed cost; record this timing proxy explicitly.
Randomize initial door phase on saved tapes; energy must not uniquely reveal it.
Actions: north/south/east/west/wait. One environment action per recurrent tick;
blocked moves consume the same movement unit as successful moves. Wait also
consumes one unit. Budget96units per episode; goal or exhaustion terminates.
Initial layouts exclude crossings through an occupied door when it closes;
door movement semantics and observation timing are protocol-frozen.

Conditions: static-open reference, static-closed reference, recurring door
schedule, and one changed schedule after exposure. Introduce layout transfer
later; do not change several environmental mechanisms together initially.

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
Oracle uses privileged map information only for solvability and training labels;
its score is a reference, not Demian evidence. No test-layout fitting.

Pilot: width16 smoke checks, then width256; four development seeds84001–84004.
Reserve six fresh confirmation seeds85001–85006, not used to revise the pilot.
Use24episodes per seed,96ticks maximum; freeze exact layouts, episode allocation,
schedule tapes and effect criterion before collecting scientific outcomes.
Benchmark pilot wall time and storage before any larger campaign.

## Implementation tasks

- [ ] **World:** create `development/demian_maze_world.py` and
  `tests/test_demian_maze_world.py`. Interfaces: `MazeConfig`, `MazeObservation`,
  `MazeAction`, `MazeSnapshot`; `reset(config)`, `observe()`, `step(action)`,
  `snapshot()`, `restore(snapshot)`. Test motion, blocked/wait cost, termination,
  door timing, exact replay and invalid configuration/action rejection.
- [ ] **Boundary:** create `development/demian_maze_runner.py` and
  `tests/test_demian_maze_runner.py`. Consume world observations; produce
  `MazeStepRecord` with observation, action, acknowledgement, state and route
  trace. Add stateless encoder and independently fitted/frozen action heads.
  Test no privileged-field influence, fitting isolation, saved-fit prediction
  replay, one-action timing and unchanged Demian parameter hashes.
- [ ] **Frozen controls:** intact recurrence, per-tick recurrence reset,
  observation-only head, random policy and prerecorded-observation replay.
  Reset is a recurrence/state control, not a learning-ablation result.
  Replay validates trajectories; it cannot supply a fair navigation score once
  actions and observations become inconsistent. Match exogenous world tapes;
  closed-loop trajectories may diverge because actions differ.
- [ ] **Protocol/run:** create `docs/DEMIAN_DYNAMIC_MAZE_PROTOCOL.json` and
  `development/run_demian_maze_baseline.py`. Verify seed novelty, fit/eval layout
  disjointness and snapshot completeness before execution. Save states,
  controller fits, world tapes, source/dependencies, predictions and hashes.
- [ ] **Verification:** run `./venv/bin/python -m pytest -q
  tests/test_demian_maze_world.py tests/test_demian_maze_runner.py`, then the
  existing closed-loop regression tests. Preserve and explain failures.
- [ ] **Review/record:** Astra reviews leakage, action-head contribution,
  controls and conclusions. Register completed outcomes and artifact pins;
  append labbook, update working state and index. No automatic status promotion.

## Later learning experiment: freeze before implementation

Specify the persistent internal variables, their owner, update equation,
feedback signal, eligible routes, budget and credit-assignment horizon. Keep
the action head fixed during this comparison. Separate task state from retained
experience so their resets and transplants have explicit meanings.

Compare: learning enabled, identical controller with learning disabled,
experience-variable reset, shuffled feedback, replayed/yoked allocation and
matched random allocation. Evaluation probes begin from matched world/task
conditions; freeze learning during probes to measure acquired competence.
Equalize exposure and action budgets. Paired exogenous tapes remain fixed.

Learning gate: prior experience improves predeclared held-out behavioral
outcomes across fresh seeds relative to learning-disabled and disrupted-
experience controls. Restore the retained variables to test rescue. A decrease
in energy cost must preserve success; fewer operations require actual skipped
computation, not merely smaller route gains.

Only after this baseline compare local processing schedules or changing
connectivity. Preserve named channels as measurement compartments while testing
whether their functional roles and relationships vary with experience.
