# Demian Basin-Horizon Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an executable, finite-window basin-horizon pilot that produces validated Demian evidence artifacts without making asymptotic or universal surface-sufficiency claims.

**Architecture:** Pydantic models validate the machine specification and emitted run record. A focused pilot module perturbs a saved six-channel reference state with paired, channel-normalized directions, runs each trajectory once to the maximum horizon, and classifies checkpoint prefixes. Existing continuation controls gain a reachable matched-surface arm; the pilot writes JSON and Parquet but does not modify NullFrame.

**Tech Stack:** Python 3.12, PyTorch, Pydantic 2, PyArrow, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-demian-basin-horizon-pilot.json`

## Global Constraints

- Keep Demian as the canonical evidence authority; do not modify NullFrame in this plan.
- Use finite-window operational language; never infer asymptotic convergence from a checkpoint.
- Separate internal regime, predictive information, continuation effect, and task utility.
- Reuse direction seeds across radii and reuse maximum-horizon trajectories across checkpoints.
- Treat exact replay as verification, not independent replication.
- Label body-surface results as specific reconstruction controls, not universal state-sufficiency proofs.
- Freeze protocol thresholds after pilot seed 94 before interpreting held-out seed 95.

## Review Focus

- A checkpoint shorter than dwell plus persistence must be right-censored, never classified as AFP-v2.
- Radius zero must not be redundantly rerun for every direction seed.
- Zero-norm channels must remain finite under channel normalization.
- Reachable surface matching must expose its residual error and source step rather than imply exact matching.
- Repeated direction/checkpoint observations must retain dependency-group identifiers.

---

### Task 1: Validate Pilot Specifications and Run Records

**Files:**
- Modify: `development/lab_schemas.py`
- Create: `tests/test_basin_horizon_schema.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `BasinHorizonPilotSpec`, `BasinHorizonRunRecord`, and `load_basin_horizon_spec(path: Path) -> BasinHorizonPilotSpec`.
- Consumes: the JSON spec named above.

- [ ] **Step 1: Write failing schema tests** for the checked-in spec, missing radius zero, unsorted checkpoints, invalid escape tolerance, overlapping pilot/held-out seeds, and dependency metadata on run records.
- [ ] **Step 2: Run** `PYTHONPATH=. .venv/bin/pytest tests/test_basin_horizon_schema.py -q` and confirm failure because the models do not exist.
- [ ] **Step 3: Implement the minimal Pydantic models and loader**, and declare the existing research tools under `[project.optional-dependencies].research`.
- [ ] **Step 4: Run the schema tests and** `PYTHONPATH=. .venv/bin/ruff check development/lab_schemas.py tests/test_basin_horizon_schema.py`.
- [ ] **Step 5: Commit** the schema, spec, plan, tests, and dependency metadata.

### Task 2: Add Finite-Window Classification and Basin Perturbations

**Files:**
- Create: `development/basin_horizon_pilot.py`
- Create: `tests/test_basin_horizon_pilot.py`

**Interfaces:**
- Consumes: `BasinHorizonPilotSpec` from Task 1 and six-channel `V1State` values.
- Produces: `perturb_reference_state(state, radius, direction_seed, scale_floor)`, `classify_checkpoint(surface_trace, full_state_trace, checkpoint, protocol)`, and `run_basin_horizon_pilot(spec)`.

- [ ] **Step 1: Write failing tests** proving finite-window censoring, operational settlement, escape detection, radius-zero identity, paired channel-normalized directions, finite zero-channel handling, one maximum-horizon run reused across checkpoints, and dependency-group IDs.
- [ ] **Step 2: Run** `PYTHONPATH=. .venv/bin/pytest tests/test_basin_horizon_pilot.py -q` and confirm failure because the module does not exist.
- [ ] **Step 3: Implement the minimal perturbation, trace, checkpoint, and aggregation behavior.** Use existing internal-dynamics classification only after a complete post-settling persistence window.
- [ ] **Step 4: Run the pilot tests and** `PYTHONPATH=. .venv/bin/ruff check development/basin_horizon_pilot.py tests/test_basin_horizon_pilot.py`.
- [ ] **Step 5: Commit** the executable pilot core.

### Task 3: Strengthen Continuation Controls

**Files:**
- Modify: `development/probe_demian_v1_continuation.py`
- Modify: `tests/test_demian_v1_continuation.py`

**Interfaces:**
- Produces: a `reachable_surface_match` continuation arm containing `source_step`, `surface_match_error`, and the ordinary divergence metrics.
- Consumes: existing exact, sham, body-surface, norm-matched-random, and channel-only controls.

- [ ] **Step 1: Write a failing test** asserting the reachable arm reports provenance and residual match error while exact and sham remain exact.
- [ ] **Step 2: Run** `PYTHONPATH=. .venv/bin/pytest tests/test_demian_v1_continuation.py -q` and confirm the reachable arm is absent.
- [ ] **Step 3: Retain reachable history states, choose the nonterminal state with minimum surface distance, align runtime metadata, and add the arm without calling it exact.**
- [ ] **Step 4: Run continuation and characterization tests plus Ruff.**
- [ ] **Step 5: Commit** the strengthened control.

### Task 4: Write Artifacts and Execute the Small Pilot

**Files:**
- Modify: `development/basin_horizon_pilot.py`
- Modify: `tests/test_basin_horizon_pilot.py`
- Modify: `data/MANIFEST.json`
- Create: `data/diagnostics/demian_basin_horizon_pilot_20261005/summary.json`
- Create: `data/diagnostics/demian_basin_horizon_pilot_20261005/checkpoints.parquet`

**Interfaces:**
- Consumes: `run_basin_horizon_pilot(spec)` and Task 1 run-record validation.
- Produces: `write_pilot_artifacts(payload, output_dir)` and a manifest-indexed pilot artifact bundle.

- [ ] **Step 1: Write a failing artifact test** that validates JSON, Parquet row count/schema, and absence of NaN/Infinity.
- [ ] **Step 2: Run the artifact test and confirm failure** because the writer does not exist.
- [ ] **Step 3: Implement JSON/Parquet writing and CLI entrypoint.**
- [ ] **Step 4: Run seed 94 as the threshold-development pilot, freeze the emitted protocol, then run seed 95 without changing thresholds; write the combined artifact and register it in `data/MANIFEST.json`.**
- [ ] **Step 5: Run targeted tests, full tests, Ruff, artifact validation, and commit. Record the four pre-existing artifact-dependent baseline failures if they remain.**
