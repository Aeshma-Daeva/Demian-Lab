# Demian Research Registry Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a validated Demian research registry for three existing hypotheses and a NullFrame reviewer page that renders the registry without inferring evidence status.

**Architecture:** Demian owns typed hypothesis, experiment, control, result, and artifact records plus immutable evidence references. It exports a validated JSON snapshot. NullFrame validates and renders that snapshot as concise reviewer cards; it does not calculate or promote statuses.

**Tech Stack:** Python 3.12, Pydantic 2, pytest, static HTML/CSS, ES modules, Node test runner.

**Spec:** `docs/superpowers/specs/2026-10-06-demian-research-registry-pilot.json`

## Global Constraints

- Demian is the evidence authority; NullFrame is read-only presentation.
- `null` is a result outcome, never a hypothesis evidence status.
- AFP basin-horizon evidence is `unevaluable`, not negative.
- Evidence status, lifecycle, and replication scope are independent fields.
- Status changes require an explicit manual-review basis and history.
- Every reference resolves to a stable record ID and immutable artifact commit.
- The reviewer page renders provided status and must not derive it from counts.

## Review Focus

- Mixed or unevaluable results must not be collapsed into a hypothesis-wide null.
- Dependent checkpoints must not be presented as independent replication.
- Missing result/control/artifact IDs must fail validation.
- Approximate reachable-history controls must retain surface mismatch, hidden-distance, and history-lag limitations.
- NullFrame must remain usable when JavaScript fails by retaining an explicit static claim boundary.

---

### Task 1: Demian Registry Contracts and Three Records

**Files:**
- Create: `development/research_registry.py`
- Create: `data/RESEARCH_REGISTRY.json`
- Create: `tests/test_research_registry.py`

**Interfaces:**
- Produces: `ResearchRegistry`, `load_registry(path)`, and `validate_registry(path)`.
- Consumes: existing characterization and basin-horizon artifacts at immutable commits.

- [ ] Write failing tests for the checked-in registry, unresolved references, null-as-hypothesis-status, unevaluable/null inconsistency, missing manual status basis, and dependent-checkpoint replication semantics.
- [ ] Run the tests and confirm they fail because the registry module does not exist.
- [ ] Implement minimal Pydantic contracts and populate AFP-v2, gate-associated dynamics, and reconstruction-dependent continuation records.
- [ ] Run tests and Ruff, then commit.

### Task 2: Validated NullFrame Snapshot Export

**Files:**
- Modify: `development/research_registry.py`
- Modify: `tests/test_research_registry.py`
- Create in NullFrame: `site/assets/data/demian-research-registry.json`

**Interfaces:**
- Produces: `export_registry_snapshot(source, destination)` with deterministic JSON.
- Consumes: validated `data/RESEARCH_REGISTRY.json`.

- [ ] Write a failing deterministic-export test that rejects invalid source records.
- [ ] Implement the exporter and generate the NullFrame snapshot from Demian.
- [ ] Verify byte-stable repeated export and commit Demian changes.

### Task 3: NullFrame Reviewer Page

**Files:**
- Create in NullFrame: `site/projects/demian/research/index.html`
- Create in NullFrame: `site/assets/js/research-registry.mjs`
- Modify in NullFrame: `site/assets/css/global.css`
- Modify in NullFrame: `site/projects/demian/index.html`
- Modify in NullFrame: `scripts/build.mjs`
- Create in NullFrame: `tests/research-registry.test.mjs`

**Interfaces:**
- Consumes: `site/assets/data/demian-research-registry.json`.
- Produces: a reviewer page grouped by hypothesis with status, scope, evaluability, controls, falsifiers, next test, and artifact links.

- [ ] Write failing validation and renderer tests, including no status inference and graceful static fallback.
- [ ] Implement snapshot validation, DOM rendering, page structure, styles, build validation, and navigation link.
- [ ] Run `npm test` and `npm run build`, then commit NullFrame changes.

### Task 4: Cross-Repository Verification

**Files:**
- Verify only; modify files only for review findings.

- [ ] Verify Demian targeted tests/Ruff and compare the full suite with its four known artifact failures.
- [ ] Verify NullFrame tests/build and exact snapshot provenance.
- [ ] Request Astra review of both commit ranges; fix Critical/Important findings once.
- [ ] Report both branches and integration status without merging or publishing automatically.
