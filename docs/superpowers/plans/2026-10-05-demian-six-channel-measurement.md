# Demian Six-Channel Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validated six-channel measurement harness and run a modest AFP-v2 baseline campaign without treating AFP-v2 as Demian's sole research target.

**Architecture:** Add exact six-channel state/surface/runtime adapters, then build trace and continuation probes on those contracts. Reuse the existing AFP-v2 classifier and emit one auditable campaign artifact with robustness slices and restrained evidence labels.

**Tech Stack:** Python 3, PyTorch, dataclasses, JSON, pytest

**Spec:** `docs/superpowers/specs/2026-10-05-demian-six-channel-measurement-design.md`

## Global Constraints

- State channels are exactly `fast`, `slow`, `control`, `message`, `carrier`, `gate`.
- The surface is a projection, not a state channel.
- Parameter provenance and trajectory history are separate fields.
- Update-rule interventions and post-step clamps use different names and result fields.
- Continuation controls cover the same tail interval used for classification.
- Continuation divergence is not reported as task utility.
- Phase one uses unselected random parameters; no training or evolution is added.
- Existing AFP-v2 thresholds remain the primary configuration; robustness uses `0.5x`, `1x`, and `2x` slices.

## Review Focus

- Surface matching with nonzero message/carrier/gate must preserve the requested readout; Task 1 tests exact reconstruction.
- Resumed runs must restore `_step_index` and `_frozen_gate`; Task 1 tests both fields.
- Constructor gate disabling/freezing must remain distinct from post-step clamping; Task 2 tests divergent semantics.
- Null interventions must not create a continuation gap; Task 3 tests sham and exact full restore.
- Empty or incompatible campaign configurations must fail clearly; Task 4 tests invalid seeds, horizons, and modes.

---

### Task 1: Six-channel surface and runtime contracts

**Files:**
- Modify: `development/demian_v1_gate_state.py`
- Modify: `tests/test_demian_v1_gate_state.py`

**Interfaces:**
- Produces: `V1RuntimeSnapshot(state, step_index, frozen_gate)`
- Produces: `match_v1_surface(model, state, target_surface) -> V1State`
- Produces: `capture_v1_runtime(model, state) -> V1RuntimeSnapshot`
- Produces: `restore_v1_runtime(model, snapshot) -> V1State`

- [ ] **Step 1: Write failing surface-matching tests**

Add tests that construct a nonzero six-channel state, request a different
surface, and assert:

```python
assert torch.allclose(model.state_vector(matched), target, atol=1e-6, rtol=1e-6)
assert all(torch.equal(matched[i], state[i]) for i in range(1, 6))
```

- [ ] **Step 2: Run the surface tests and verify failure**

Run: `pytest -q tests/test_demian_v1_gate_state.py -k surface_match`

Expected: FAIL because `match_v1_surface` does not exist.

- [ ] **Step 3: Implement exact surface matching**

Add:

```python
def match_v1_surface(
    model: DemianV1GateState,
    state: V1State,
    target_surface: torch.Tensor,
) -> V1State:
```

Solve for `fast` by subtracting the message, carrier, and gate readout terms
already used by `state_vector`. Preserve the other five tensors.

- [ ] **Step 4: Run the surface tests and verify pass**

Run: `pytest -q tests/test_demian_v1_gate_state.py -k surface_match`

Expected: PASS.

- [ ] **Step 5: Write failing runtime-snapshot tests**

Assert that capture/restore preserves all six tensors, `_step_index`, and a
non-null `_frozen_gate`, while returning cloned tensors rather than aliases.

- [ ] **Step 6: Implement runtime capture and restore**

Add the frozen dataclass and functions named in **Interfaces**. Reject snapshots
with missing or incorrectly shaped channels.

- [ ] **Step 7: Verify Task 1**

Run: `pytest -q tests/test_demian_v1_gate_state.py`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add development/demian_v1_gate_state.py tests/test_demian_v1_gate_state.py
git commit -m "test: define six-channel runtime contracts"
```

### Task 2: Typed six-channel trace runner

**Files:**
- Create: `development/demian_v1_measurement.py`
- Create: `tests/test_demian_v1_measurement.py`
- Modify: `development/demian_v1_gate_state.py`

**Interfaces:**
- Consumes: Task 1 runtime and surface adapters.
- Produces: `V1TraceConfig`
- Produces: `V1TraceResult`
- Produces: `run_v1_measurement(config: V1TraceConfig) -> V1TraceResult`
- Produces: `UpdateMode = Literal["active", "gate_disabled", "gate_frozen"]`
- Produces: `PostStepClamp = Literal["none", *V1_CHANNELS]`

- [ ] **Step 1: Write failing trace-schema tests**

For a four-step trace, assert four surfaces, four full states, four entries for
each of the six channels, four metric rows, and complete configuration metadata.

- [ ] **Step 2: Write failing intervention-semantics tests**

Run constructor `gate_disabled` and post-step `gate` clamp from matched weights
and initial state. Assert their labels differ and at least one route metric or
next-state component differs. Add the equivalent frozen-versus-clamped test.

- [ ] **Step 3: Verify the new tests fail**

Run: `pytest -q tests/test_demian_v1_measurement.py`

Expected: collection or import failure because the module is absent.

- [ ] **Step 4: Implement the trace runner**

`V1TraceConfig` must include seed, hidden size, steps, parameter provenance,
history condition, update mode, post-step clamp, perturbation step/scale, device,
and dtype. Reject unknown modes, `steps < 3`, empty provenance, and perturbation
steps outside the trace.

- [ ] **Step 5: Route surface perturbations through Task 1 matching**

Do not call the inherited `write_surface_state` for six-channel traces. Record
the reconstruction error in the metric row at the intervention step.

- [ ] **Step 6: Verify Task 2**

Run: `pytest -q tests/test_demian_v1_measurement.py tests/test_demian_v1_gate_state.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add development/demian_v1_measurement.py development/demian_v1_gate_state.py tests/test_demian_v1_measurement.py
git commit -m "feat: add six-channel measurement runner"
```

### Task 3: Tail-aligned continuation controls

**Files:**
- Create: `development/probe_demian_v1_continuation.py`
- Create: `tests/test_demian_v1_continuation.py`

**Interfaces:**
- Consumes: Task 1 runtime adapters and Task 2 trace schema.
- Produces: `run_v1_continuation_probe(config, pause_steps, resume_steps) -> dict[str, Any]`
- Produces arms: `full_checkpoint`, `body_surface`, `sham`, `random_norm_matched`, and `<channel>_only` for all six channels.

- [ ] **Step 1: Write failing exact-restore and sham tests**

Assert `full_checkpoint` and `sham` gaps are below `1e-7`, use the same operating
mode and step index, and report the exact continuation interval.

- [ ] **Step 2: Write failing surface/control tests**

Assert `body_surface` reconstructs the paused surface within `1e-6`, random
controls are norm-matched within `1e-5`, and every channel arm is present.

- [ ] **Step 3: Verify the tests fail**

Run: `pytest -q tests/test_demian_v1_continuation.py`

Expected: import failure because the probe is absent.

- [ ] **Step 4: Implement continuation arms**

Clone model weights and runtime metadata for every arm. Use Task 1 surface
matching. Generate random controls with an arm-specific deterministic seed and
match each restored channel's norm without copying its direction.

- [ ] **Step 5: Add interval alignment validation**

Reject `pause_steps < 1` or `resume_steps < 3`. Emit one-based inclusive
`control_steps`, and let the campaign assert equality with `classified_steps`.

- [ ] **Step 6: Verify Task 3**

Run: `pytest -q tests/test_demian_v1_continuation.py tests/test_demian_v1_measurement.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add development/probe_demian_v1_continuation.py tests/test_demian_v1_continuation.py
git commit -m "feat: add six-channel continuation controls"
```

### Task 4: Six-channel AFP-v2 baseline campaign

**Files:**
- Create: `development/run_demian_v1_characterization.py`
- Create: `tests/test_demian_v1_characterization.py`
- Modify: `development/afp_v2_characterization.py`

**Interfaces:**
- Consumes: Task 2 traces, Task 3 controls, and `classify_afp_v2`.
- Produces: `classify_trace_result(trace, continuation_gap, config) -> dict[str, Any]`
- Produces: `run_demian_v1_campaign(...) -> dict[str, Any]`

- [ ] **Step 1: Write failing trace-classification tests**

Use a deterministic toy trace to assert separate surface, full-state, and six
channel classifications. Assert causal continuation and task utility occupy
different fields.

- [ ] **Step 2: Write failing campaign-factor tests**

With two short seeds, assert parameter provenance, history condition, update
mode, and post-step clamp are separate artifact dimensions. Assert the
five-channel predecessor is labeled `historical_control`.

- [ ] **Step 3: Write failing configuration-validation tests**

Reject empty seeds, horizons shorter than the largest tail, unsupported update
modes, and perturbation steps outside the horizon.

- [ ] **Step 4: Verify the tests fail**

Run: `pytest -q tests/test_demian_v1_characterization.py`

Expected: import failure because the campaign module is absent.

- [ ] **Step 5: Extract trace classification without changing thresholds**

Add `classify_trace_result` as a thin adapter around `classify_afp_v2`. Keep the
current AFP-v2 labels and configuration backward compatible.

- [ ] **Step 6: Implement the campaign matrix**

Defaults: seeds `94,95,96`, hidden size `16`, steps `512`, perturb step `256`,
scale `0.05`, update modes active/disabled/frozen, and uninterrupted/perturbed
histories. Run tail-aligned continuation only where its semantics apply.

- [ ] **Step 7: Add robustness slices**

Reclassify saved traces for tails `24,48,96` and threshold multipliers `0.5`,
`1.0`, `2.0`; do not rerun dynamics for classification-only slices. Record
whether the primary label changes.

- [ ] **Step 8: Verify Task 4**

Run: `pytest -q tests/test_demian_v1_characterization.py tests/test_afp_v2_characterization.py`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add development/run_demian_v1_characterization.py development/afp_v2_characterization.py tests/test_demian_v1_characterization.py
git commit -m "feat: characterize six-channel Demian dynamics"
```

### Task 5: Generate and document the phase-one artifact

**Files:**
- Create: `data/diagnostics/demian_v1_characterization_20261005/summary.json`
- Modify: `data/MANIFEST.json`
- Modify: `docs/LABBOOK.md`
- Modify: `docs/CLAIMS.md` only if a replicated claim satisfies `docs/EXPERIMENT_RULES.md`
- Modify: `tests/test_current_artifacts.py` or the repository's current manifest test

**Interfaces:**
- Consumes: Task 4 command-line campaign.
- Produces: one immutable phase-one summary artifact and manifest entry.

- [ ] **Step 1: Add a failing artifact-schema test**

Require campaign factors, classifier configuration, robustness slices,
continuation intervals, per-channel results, null controls, and all five evidence
labels. Assert no `task_utility` field is true or inferred in this task-free run.

- [ ] **Step 2: Run the primary campaign**

Run:

```bash
python -m development.run_demian_v1_characterization \
  --seeds 94,95,96 --hidden-size 16 --steps 512 \
  --out data/diagnostics/demian_v1_characterization_20261005/summary.json
```

Expected: one JSON path printed and exit status 0.

- [ ] **Step 3: Inspect before interpreting**

Check counts by surface regime, internal regime, channel, history condition, and
update mode. Compare robustness slices and continuation nulls before writing any
claim.

- [ ] **Step 4: Register the artifact tersely**

Add the exact command, scope, result counts, interpretation boundary, and
artifact path to the manifest and labbook. Update claims only if replication and
falsification requirements are met.

- [ ] **Step 5: Run complete verification**

Run the focused suite plus the repository's current non-artifact-blocked suite,
`python -m compileall -q development tests`, JSON parsing, and `git diff --check`.

Expected: all runnable tests pass; any missing historical artifact is reported
separately and not converted into a new claim.

- [ ] **Step 6: Request independent scientific review**

Ask Astra to review the immutable artifact, thresholds, controls, and proposed
interpretation. Resolve actionable findings before committing conclusions.

- [ ] **Step 7: Commit Task 5**

```bash
git add data/diagnostics/demian_v1_characterization_20261005 data/MANIFEST.json docs/LABBOOK.md docs/CLAIMS.md tests
git commit -m "research: add six-channel baseline characterization"
```

## Later plans

Create separate plans after phase one for:

1. matched-surface and readout-null microperturbation geometry;
2. delayed-memory capability and delay extrapolation;
3. calibrated uncertainty representation;
4. Jacobian and finite-time Lyapunov characterization;
5. evolved/task-trained provenance with held-out budget-matched controls.

