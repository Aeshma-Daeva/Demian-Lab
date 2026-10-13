# Uncertainty-Guided Routing Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for native execution.
> Terra Medium is the intended implementer. This is a proposed handoff, not an
> instruction to launch another chat or run a campaign now.

**Goal:** Produce a valid, reproducible discovery harness for internal evidence
retention and later uncertainty-guided routing experiments.

**Architecture:** Repair existing route measurements, then add an isolated
ambiguous-evidence world and evaluator. Reuse Demian's model/runtime utilities
with a stateless encoder and explicit separation of agent inputs and evaluator
metadata. Gate later probes and controllers on measured validity.

**Tech Stack:** Existing Python, PyTorch, pytest, JSON/JSONL; no new dependency
unless an existing repository dependency cannot serve the task.

**Spec:** `docs/superpowers/specs/2026-10-12-uncertainty-routing-design.md`

## Global Constraints

- Keep repository text terse; raw numbers belong in artifacts.
- No Demian weight training, controller, architecture extension, external agent
  memory, dependency installation, publishing, or long run in the initial package.
- Preserve previous results; corrections append provenance and status history.
- All previously inspected seeds are discovery-only. Confirmation seeds and
  episodes stay unexecuted until the predictor and protocol are frozen.
- Use existing branch/worktree if suitable; inspect dirty files before editing.
- Follow `AGENTS.md`, `docs/EXPERIMENT_RULES.md`, and `demian-research`.
- Execute routine implementation and verification without intermediate approval
  requests. Stop for a material scientific design decision, not a routine test.

## Review Focus

- Leakage through target, posterior, future evidence, acknowledgements, or labels.
- Repeated directions/histories counted as independent model seeds.
- Same norm but different carrier loading; scale and observation-window confounds.
- World/encoder/controller state accidentally supplying external memory.
- Probe success or fixture correctness promoted to autonomous capability.

## Task 1 — Repair route measurement contracts

Files: `development/demian_route_stability.py`,
`development/run_demian_route_stability.py`,
`development/demian_channel_visibility.py`,
`tests/test_demian_route_stability.py`.

- [ ] Preserve legacy defaults; add an explicit absolute pulse-norm option.
  Resolve pulse norm once per cell and apply the same realized norm to paired
  carrier/full-state directions. Save requested and realized per-channel norms.
- [ ] Group summaries by stable history ID and full history configuration,
  not cue symbol alone. Include scope, amplitude, direction set, and window.
- [ ] Add a time-aligned contrast: mean over transitions 1..K of
  `(I_carrier(t)-I_control(t))/requested_pulse_norm`. Retain legacy peak
  difference under its accurate name. Reject incomplete windows.
- [ ] Repair prior-state bookkeeping for every arm; independently replay
  consecutive states and assert stored increments equal their one-step norms.
  Check step zero separately from ordinary post-transition measurements.
- [ ] Validate finite positive pulse norms, finite features, unique history IDs,
  cue bounds, and all history interventions strictly preceding the probe.
- [ ] Tests: identical directions produce zero contrast; swapping direction
  labels negates it; multiplier 1 gives zero interaction; histories with the same
  cue remain separate; matched pulses retain their norms across histories;
  invalid/incomplete windows fail. Assert full-state pulses affect multiple
  channels and channel pulses affect only the declared channel.
- [ ] Correct descriptions of prior within-carrier comparisons and separately
  peaked statistics through appended errata. Preserve old numerical artifacts.

Deliverable: measured contrasts with explicit direction, loading, history, and
time semantics. No new selectivity conclusion.

## Task 2 — Add the ambiguous-evidence world and oracle

Create `development/demian_uncertainty_world.py` and
`tests/test_demian_uncertainty_world.py`.

Interfaces:

- Immutable `EvidenceObservation(tick, phase, symbol, reliability)`; missing
  evidence uses null symbol/reliability; phase is evidence/delay/resolve/complete.
- `AmbiguousEvidenceWorld` generates or replays a declared episode, returns only
  public observations, and captures/restores all world/RNG state.
- `reference_belief(public_prefix)` returns probability, log odds, and entropy;
  it never receives the hidden target. Evaluation truth is a separate record.

- [ ] Implement the spec's likelihood model and scripted fixtures.
- [ ] Test: no evidence gives p=.5; one positive .75 cue gives p=.75;
  opposing .75 cues give p=.5; two positive .75 cues give p=.9;
  reliability .5 contributes zero log odds; sequence reversal preserves posterior.
- [ ] Test identical prefixes with different hidden targets produce identical
  observations. Truthful resolution changes observations only at the declared tick.
- [ ] Test RNG/snapshot replay and reject malformed symbols, reliability,
  resolution timing, or continuation beyond completion.

Deliverable: task uncertainty with an explicit oracle, independent of Demian.

## Task 3 — Connect Demian and capture auditable traces

Create `development/demian_uncertainty_probe.py` and
`tests/test_demian_uncertainty_probe.py`. Reuse
`development/demian_v1_gate_state.py` runtime capture/restore and route tracing;
do not alter its transition equations or retrofit hidden task fields into
`development/closed_loop_world.py`.

Interfaces: stateless `encode_evidence(observation, hidden_size)` and
`run_uncertainty_episode(model_seed, config, public_frames)` returning checkpoint
states, timestamped measurements, and configuration provenance. The episode
runner takes public frames only; join evaluator targets after simulation.

- [ ] Freeze an encoder: disjoint slots for evidence symbols, public
  reliability, phase, and missing-evidence indicator; scale .25. Document slot
  assignments, reject insufficient width, and prevent modulo slot collisions.
- [ ] Keep evaluator posterior/entropy, target, condition labels, and episode
  IDs out of model inputs. No read/write register and no feedback of probe output.
- [ ] Save complete model runtime at selected checkpoints, including metadata
  needed for exact continuation. Preserve external observation replay separately.
- [ ] Test equal public input and model initialization yield equal states across
  different evaluator targets; full checkpoint restoration and replay are exact;
  passive instrumentation does not change trajectories; a state reset removes
  model history rather than leaving a hidden encoder cache.

Deliverable: measurable internal state under task-defined ambiguous evidence.

## Task 4 — Freeze a discovery pilot and run integrity checks

Create `docs/DEMIAN_UNCERTAINTY_DISCOVERY_PROTOCOL.json`,
`development/run_demian_uncertainty_discovery.py`, and focused runner tests.

- [ ] Commit/version the protocol before the scientific pilot. Include code
  revision, independent model and episode seed lists, input hashes, dtype,
  device, encoder, observation timing, checkpoint definitions, and exclusions.
- [ ] Smoke run: width 16, 2 model seeds, 4 episodes, 8 delay steps. Require
  deterministic replay, finite outputs, no leakage, and expected oracle values.
- [ ] Discovery pilot: width 256; existing seeds 294–297 and 394–397, all labeled
  discovery; delays 8 and 32; 16 unique public sequences per condition/delay,
  shared across model seeds. Four conditions: balanced informative, consistent
  informative, noise-only, and order permutations of matched evidence multisets.
  Label scripted versus generative sampling explicitly. Two evidence events,
  declared delay, then resolution; save three post-resolution transitions.
- [ ] Use a protocol budget of 15 CPU wall-clock minutes and 500 MB outputs for
  the pilot. Estimate from smoke timing; if exceeded, save a clearly labeled
  partial run with completed cell IDs. Resume without duplicating/overwriting.
- [ ] Save an evidence inventory: completed cells, exclusions/nonfinite failures,
  replay validity, channel/state scales, and input/state contrasts. Separate
  model seeds from shared episodes, histories, and repeated observations.
- [ ] Do not fit/report a confirmed uncertainty predictor in this package.
  Retain checkpoints sufficient for subsequent probe development.

Deliverable: a complete or explicitly partial discovery dataset and harness report.

## Task 5 — Record, verify, and hand off

- [ ] Register the hypothesis as untested with no supporting result unless
  scientific review explicitly changes it. Harness success is instrumentation
  evidence. Record missing fresh-seed/probe/intervention conditions.
- [ ] Index versioned protocol and artifacts in `data/RESEARCH_REGISTRY.json`,
  `data/MANIFEST.json`, `data/INDEX.md`; append labbook chronology and update
  working-state priority. Mark older audits pending registration if incomplete.
- [ ] Pin artifacts to a commit containing those paths, then commit registry
  references separately. Validate schema and immutable artifact provenance.
- [ ] Run focused world, runner, route, runtime, and registry tests, then the
  repository suite once. Report failures by name; prior missing archived-v10
  artifacts are a known possibility, not permission to assume all failures old.
- [ ] Deliver a short report: what is implemented, integrity status, evidence
  available, unresolved issues, exact reproduction commands, next scientific gate.

## Subsequent packages — conditional, not implementation scope yet

1. Freeze probes on discovery data; evaluate episode transfer per model seed.
   Split by entire public sequence/permutation family before fitting scaling or
   models. Compare time/current-cue baselines and shuffled-label controls. A
   probe decoding evidence is not automatically a representation of uncertainty.
2. Test transfer across fresh model seeds using declared invariant features or
   independently validated alignment. Use seed-level uncertainty estimates.
3. Corrected route-response prediction: multiple directions, fixed absolute
   pulse size, state-scale baselines, and finite-horizon tangent response along
   the intact and weakened trajectories. Compare numerical precision before
   attributing residual effects to organization. A Jacobian norm alone is insufficient.
4. Channel intervention/rescue: isolate contributors to retained evidence and
   later functional use. Separate input-reachable histories from directly
   perturbed or transplanted states.
5. Add the costed query task and frozen state-conditioned routing controller;
   validate query value with oracle baselines before claiming functional benefit.
6. Introduce learning/adaptation only through a separately declared mechanism.

Do not choose thresholds or features using confirmation outcomes. Freeze the
primary score, practical effect threshold, coverage/abstention rules, and sample
size before running confirmation; publish failures and unsupported conditions.
