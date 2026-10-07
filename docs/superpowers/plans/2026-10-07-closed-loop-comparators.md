# Closed-Loop Comparators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first matched closed-loop comparator harness for Demian, MLP, vanilla RNN, GRU, and a Demian routing ablation.

**Architecture:** A common model adapter owns complete runtime state, declared action-head access, checkpointing, and state-size accounting. The existing world and connector remain unchanged. A campaign runner executes shared observation and perturbation schedules under separate state-budget and parameter-budget specifications.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing closed-loop world and Demian v1 modules.

**Spec:** `docs/superpowers/specs/2026-10-07-closed-loop-comparators-design.md`

## Global Constraints

- Preserve the existing world, connector, action latency, and bounded action schema.
- Keep untrained and trained comparison regimes separate.
- Report complete persistent state bytes, parameter count, action-head access, and measured compute.
- Use shared episode schedules and perturbation draws, but separate initialization, initial-state, environment, and training seeds.
- Do not claim architectural equivalence or superiority from either normalization track.

## Review Focus

- A memoryless MLP must not retain hidden history through an adapter cache.
- State-size accounting must include all persistent tensors, not exposed readout alone.
- Checkpoint restoration must reject architecture/configuration mismatches.
- Action heads must expose equivalent declared state capacity across models.
- Matched perturbation schedules must be identical for every condition of a parameter seed.

---

### Task 1: Common comparator adapter contract

**Files:**
- Create: `development/closed_loop_comparators.py`
- Create: `tests/test_closed_loop_comparators.py`

**Interfaces:**
- Produces `ClosedLoopModelAdapter`, `ModelRuntimeSnapshot`, and adapter metadata.
- Consumes `WorldObservation`, `ActionAcknowledgement`, and `ActionProposal`.

- [ ] Write failing tests for a memoryless MLP adapter, complete-state byte accounting, and mismatched checkpoint rejection.
- [ ] Implement adapter construction for `mlp`, `rnn`, `gru`, `demian`, and `demian_route_ablation`.
- [ ] Use a shared observation encoder and equal-capacity action head; MLP state must be empty.
- [ ] Verify adapter tests pass.
- [ ] Commit: `feat: add closed-loop comparator adapters`.

### Task 2: Common runner and replay controls

**Files:**
- Modify: `development/closed_loop_comparators.py`
- Modify: `tests/test_closed_loop_comparators.py`

**Interfaces:**
- Produces `ComparatorRunner.step()`, exact checkpoint restore, fixed-observation replay, and autonomous continuation.
- Consumes `ClosedLoopModelAdapter` from Task 1 and the existing world connector.

- [ ] Write failing tests for identical world/action ordering, fresh-runner checkpoint restore, and fixed-observation replay.
- [ ] Implement runner ordering: connector advance, world observe, adapter step, bounded action proposal, connector submit, world tick.
- [ ] Implement post-input autonomous continuation with no new observation injection.
- [ ] Verify focused comparator tests pass.
- [ ] Commit: `feat: add comparator continuation controls`.

### Task 3: Matched interventions and metrics

**Files:**
- Modify: `development/closed_loop_comparators.py`
- Modify: `tests/test_closed_loop_comparators.py`

**Interfaces:**
- Produces relative-norm internal pulse, shared register disturbance, per-tick action/storage metrics, and trajectory summaries.
- Consumes a parameter seed and declared condition.

- [ ] Write failing tests that identical seeds reuse schedules and that pulse norm is relative to complete state norm.
- [ ] Implement baseline, internal, environment, and combined conditions for every architecture.
- [ ] Record correctness, wrong/no-answer, time-to-answer, invalid actions, storage operations, surface/full-state trajectories, state bytes, parameter count, and compute time.
- [ ] Verify focused comparator tests pass.
- [ ] Commit: `feat: measure matched comparator interventions`.

### Task 4: Budget tracks and pilot command

**Files:**
- Create: `development/run_closed_loop_comparator_pilot.py`
- Modify: `tests/test_closed_loop_comparators.py`
- Modify: `README.md`

**Interfaces:**
- Produces parameter-budget and complete-state-budget pilot payloads.
- Consumes comparator adapter specs and campaign conditions from Tasks 1–3.

- [ ] Write failing tests for separate state-budget and parameter-budget configurations.
- [ ] Implement deterministic budget selection and a compact JSON pilot artifact.
- [ ] Add terse reproduction commands and explicit untrained-status wording to README.
- [ ] Run focused comparator, closed-loop, registry, and lint checks.
- [ ] Commit: `feat: add closed-loop comparator pilot`.

### Task 5: Final verification and review

**Files:**
- Modify only files required by review findings.

- [ ] Run the focused closed-loop and comparator suites.
- [ ] Run the full suite; report unrelated archived-artifact failures separately if they recur.
- [ ] Request scientific/control review of the comparator diff.
- [ ] Commit any verified review fix separately.
