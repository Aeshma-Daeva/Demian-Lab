# Demian Research Organization Governance

## Purpose

Make the repository, rather than a chat transcript, the durable research
interface for any future agent or collaborator.

## Authority

`data/RESEARCH_REGISTRY.json` is the structured source of truth for active
hypotheses, experiments, controls, results, and linked artifacts. Claim status
remains a manual-review decision; no script or UI promotes evidence.

## Entry Points

- `AGENTS.md` is the current repository instruction surface for Codex agents.
- `CLAUDE.md` remains an explicitly legacy v7.4–v9 compatibility reference.
- A user-level `demian-research` skill routes relevant work to the repository
  state and evidence discipline without duplicating scientific content.

## Required Research Loop

1. Before a nontrivial run, record or link its hypothesis, protocol, controls,
   falsifier, and intended artifact location.
2. After a run, add the artifact and a result record that separates observation,
   interpretation, uncertainty, and evaluability.
3. On review, manually update hypothesis status and its status basis.
4. Update the labbook, working state, and claims only according to their
   distinct ownership in `docs/EXPERIMENT_RULES.md`.

## Enforcement

Validation must reject malformed registry records, unresolved evidence links,
and invalid snapshots. CI verifies the registry and its exported snapshot.
It does not infer whether a new artifact deserves a scientific claim.

## Presentation

NullFrame consumes an exported, validated snapshot. It is read-only and shows
compact status, scope, controls, falsifiers, next tests, and artifact links.
It must not calculate evidence status or turn metrics into conclusions.

## Non-Goals

- No automatic claim promotion.
- No raw metric dashboard as the primary interface.
- No separate shadow labbook outside the repository.
- No rewrite of historical v7.4–v9 records into current Demian claims.

## Acceptance Criteria

- A fresh Codex instance can find the current sources of truth from `AGENTS.md`.
- Relevant work activates the reusable skill and follows the research loop.
- Invalid registry changes fail locally and in CI.
- Current v1 records can be represented without treating unevaluable results as
  null findings.
- NullFrame renders the exported registry without becoming an authority.
