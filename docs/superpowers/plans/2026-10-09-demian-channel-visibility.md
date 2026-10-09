# Demian Channel Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a small, channel-local perturbation in Demian becomes immediately or delayed-visible at the exposed surface under identical fixed inputs.

**Architecture:** Start both arms from one replayed trajectory checkpoint. Apply equal-norm plus/minus pulses to one full-state channel, then replay the same encoded inputs. Report signed-pair symmetry, full-state separation, surface separation, and first threshold crossing without equating visibility with memory or adaptation.

**Tech Stack:** Python 3.12, PyTorch, pytest, current closed-loop adapters.

**Spec:** `docs/REGIME_MATCHED_CHARACTERIZATION.md` Phase 1.

## Global Constraints

- Demian only; no architecture-comparison conclusion.
- Fixed input replay after the intervention; no action/environment feedback claim.
- Pulse scale is relative to the selected channel norm; zero-norm channels reject the cell.
- Plus/minus directions are paired and use the same direction seed.
- Visibility is the first surface-separation crossing of a frozen threshold.
- Output labels it as local sensitivity/visibility, not AFP-v2, memory, learning, or adaptation.

## Review Focus

- A zero-amplitude or zero-norm pulse must not fabricate visibility; Task 1 rejects it.
- The two arms must replay identical inputs; Task 2 asserts frame identity.
- Plus/minus asymmetry must be retained, not averaged away; Task 2 records both trajectories.
- A threshold crossed at the intervention tick is immediate, not delayed; Task 1 checks the index.
- A never-visible pulse must emit `null`, not a horizon-sized visibility time; Task 1 checks this.

### Task 1: Pure paired visibility measurement

**Files:**
- Create: `development/demian_channel_visibility.py`
- Create: `tests/test_demian_channel_visibility.py`

**Interfaces:**
- Produces: `channel_pulse`, `first_visibility_tick`, and `run_channel_visibility_cell`.
- Consumes: `ClosedLoopModelAdapter`, recorded `InputFrame` values, and six Demian state channels.

- [ ] Write failing tests for channel-local relative pulse norm, immediate/null visibility, and paired opposite directions.
- [ ] Run the focused test and confirm the missing-interface failure.
- [ ] Implement the pure helpers and paired replay cell.
- [ ] Run the focused test and confirm it passes.
- [ ] Commit the tested measurement core.

### Task 2: Frozen pilot protocol and compact artifacts

**Files:**
- Create: `docs/DEMIAN_CHANNEL_VISIBILITY_PROTOCOL.json`
- Create: `development/run_demian_channel_visibility.py`
- Modify: `README.md`
- Test: `tests/test_demian_channel_visibility.py`

**Interfaces:**
- Produces: a fresh JSON artifact with protocol, per-cell results, and aggregate descriptive summaries.
- Consumes: Task 1 and the frozen protocol.

- [ ] Write failing tests for exact seed/channel/cell coverage and no output-directory reuse.
- [ ] Run the focused test and confirm failure.
- [ ] Implement the runner with fresh output creation and terse README reproduction text.
- [ ] Run focused tests and a small pilot.
- [ ] Commit the runner and protocol.

### Task 3: Verification and scientific review

**Files:**
- Modify only verified review findings.

- [ ] Run the focused visibility, adapter, and tangent suites.
- [ ] Run the full suite; report unrelated archived-artifact failures separately if present.
- [ ] Request a scientific/control review of the complete diff.
- [ ] Run the frozen pilot only after review; record results without upgrading claims.
