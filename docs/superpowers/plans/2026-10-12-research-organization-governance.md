# Research Organization Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make validated repository records the durable research interface for Demian work.

**Architecture:** `AGENTS.md` directs repository agents to the typed registry and document ownership rules. A small global Codex skill activates that workflow across sessions. Python validation and GitHub Actions prevent malformed records; NullFrame receives only a validated export and remains read-only.

**Tech Stack:** Markdown, Codex skill metadata, Python 3.11+, Pydantic 2, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-12-research-organization-governance.md`

## Global Constraints

- `data/RESEARCH_REGISTRY.json` is the evidence authority.
- Evidence status changes require explicit manual review; automation must not promote claims.
- Raw artifacts remain immutable evidence and must be linked by commit before registry promotion.
- NullFrame consumes exported status and never derives it.
- Legacy v7.4–v9 instructions remain archival only.

## Review Focus

- A fresh agent must not follow superseded `CLAUDE.md` guidance.
- A result with incomplete evaluation conditions must retain `unevaluable`, not become a null finding.
- Validation must fail before an invalid snapshot is written.
- CI must install only the dependencies required for registry tests.
- The global skill must point to repository authority rather than duplicate changing scientific claims.

---

### Task 1: Current Repository Entry Point

**Files:**
- Create: `AGENTS.md`
- Modify: `CLAUDE.md`
- Test: manual fresh-session read path

**Interfaces:**
- Produces: a concise start surface that directs agents to current records.
- Consumes: `docs/EXPERIMENT_RULES.md` and `data/RESEARCH_REGISTRY.json`.

- [ ] Write `AGENTS.md` with current source ordering, the required research loop, document ownership, and the manual claim-promotion boundary.
- [ ] Keep the existing `CLAUDE.md` legacy banner; do not rewrite archived historical guidance as current science.
- [ ] Verify every path named by `AGENTS.md` exists and no current instruction points agents to the v7.4–v9 workflow.
- [ ] Commit repository-entrypoint changes.

### Task 2: Reusable Codex Research Skill

**Files:**
- Create: `~/.codex/skills/demian-research/SKILL.md`
- Create: `~/.codex/skills/demian-research/agents/openai.yaml`

**Interfaces:**
- Produces: automatic skill discovery for Demian research, claim, protocol, result, and experiment work.
- Consumes: the active repository's `AGENTS.md`, registry, and experiment rules.

- [ ] Create a concise `demian-research` skill with automatic invocation enabled.
- [ ] Require reading the repository authority before interpreting or updating research work; prohibit automatic evidence-status promotion.
- [ ] Validate the skill with `quick_validate.py`.
- [ ] Confirm discovery from the user skill directory.

### Task 3: Registry Command and Verification Contract

**Files:**
- Modify: `development/research_registry.py`
- Modify: `tests/test_research_registry.py`
- Create: `.github/workflows/research-registry.yml`

**Interfaces:**
- Produces: `python -m development.research_registry validate <path>` and `export <source> <destination>`.
- Consumes: a checked-in registry JSON document.

- [ ] Write failing CLI tests for successful validation, invalid input exit status, and deterministic export.
- [ ] Implement a minimal argparse entrypoint that calls the existing validation/export functions and reports only concise paths/status.
- [ ] Run the CLI tests to confirm they pass.
- [ ] Add a dedicated GitHub Actions workflow that installs Python, `pydantic`, and `pytest`, then runs only registry tests and registry validation.
- [ ] Commit validation and CI changes.

### Task 4: Current v1 Evidence Registration Gate

**Files:**
- Modify: `data/RESEARCH_REGISTRY.json`
- Modify: `docs/LABBOOK.md`
- Modify: `docs/CLAIMS.md` only if manual review promotes a claim
- Modify: `data/INDEX.md` and `data/MANIFEST.json`
- Test: `tests/test_research_registry.py`

**Interfaces:**
- Produces: linked route-stability and graded-interaction records only after their artifacts are committed.
- Consumes: immutable experiment commits, protocols, summaries, and manual review.

- [ ] Commit the completed route experiment package before citing it as immutable registry evidence.
- [ ] Add experiment, controls, result, and artifact records that distinguish binary route consequence from graded pulse-route interaction.
- [ ] Record the random-direction matched gain-sweep as the next discriminator; do not claim route specificity before that control.
- [ ] Add concise labbook/index entries and promote `CLAIMS.md` only if manual review supports it.
- [ ] Run registry tests and validate the checked-in registry.

### Task 5: NullFrame Snapshot Handoff

**Files:**
- Verify: `data/RESEARCH_REGISTRY.json`
- Create in NullFrame when its connector is available: exported `demian-research-registry.json` and read-only reviewer page

**Interfaces:**
- Produces: a deterministic registry snapshot for NullFrame.
- Consumes: validated Demian registry only.

- [ ] Export a deterministic snapshot from the validated registry.
- [ ] Copy it into NullFrame only through its connected workspace workflow.
- [ ] Verify the page renders status, scope, controls, uncertainty, falsifiers, next tests, and artifact links without status inference.
- [ ] Record the paired Demian/NullFrame commit identities.

## Self-Review

- Spec coverage: Tasks 1–5 cover entrypoints, reusable workflow, validation,
  current evidence, and read-only presentation.
- Type consistency: CLI names match `validate_registry` and
  `export_registry_snapshot`.
- Proportion: automation validates record integrity, while scientific judgment
  remains explicitly manual.
