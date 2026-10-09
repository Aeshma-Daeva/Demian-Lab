# Phase 0 validation implementation plan

> Execution: native, using executing-plans and test-driven-development.

Goal: validate the existing tangent measurement and gate fresh-seed confirmation.
Spec: `REGIME_MATCHED_CHARACTERIZATION.md`, `REGIME_MATCHED_PROTOCOL.json`.
Architecture: unchanged state transitions; sampled operating-point diagnostics,
explicit development audit, immutable provenance, fail-closed confirmation.
Stack: existing Python/PyTorch; no new dependencies.

## Constraints

- Keep readouts, state budgets, drive and model initialization unchanged.
- Development seeds 94–96; confirmation seeds 194–293; no reuse across stages.
- Preserve old runs. Save exact source snapshots for new runs.
- Report directional finite-window growth, not asymptotic chaos or memory.
- Later phases remain gated. No merge.

## Task 1: Complete and validate operation diagnostics

Files: `development/regime_characterization.py`, `tests/test_regime_characterization.py`.
Interface: `StateMap.operating_point(z,x,floor)` returns operation fractions and reconstruction error.

- [ ] RED: test fused-GRU reset/update/candidate diagnostics and transition parity;
  Demian cross-route, final fast activation, gate pressure and readout slopes;
  diagnostics run only at sample ticks; invalid inputs fail explicitly.
- [ ] GREEN: instrument actual module outputs; reconstruct only fused/compound operations.
- [ ] Verify: focused tests pass; sampled diagnostics do not mutate runtime.
- [ ] Commit validated instrumentation.

## Task 2: Gate and freeze confirmation

Files: `development/regime_validation.py`, `development/run_regime_characterization.py`,
`tests/test_regime_validation.py`, protocol and decision JSON.
Interface: `audit_development(run_dirs, protocol)` returns decision with reasons;
runner consumes its hashed decision and enforces complete confirmation settings.

- [ ] RED: missing/duplicate records, failed precision, inconsistent direction/horizon,
  source/protocol mismatch, incomplete coverage, stale decisions and modified settings are rejected.
- [ ] GREEN: audit complete seed×direction×architecture×horizon cells; archive sources;
  enforce schema, state budget and confirmation settings before creating output.
- [ ] Verify: focused and full tests; rerun development at both frozen horizons.
- [ ] Commit instrument/protocol before runs; write decision after development audit.

## Task 3: Execute and report

- [ ] Launch fresh-seed confirmation only after passing decision; retain failures.
- [ ] Record terse development results, limitations, test failures and run status.
- [ ] Fresh branch review; publish approved branch changes without merging.

Review focus: no conflation of operation saturation with an unsaturated state subspace;
paired horizons share delay drive except the final query; incomplete rate cells stay unresolved;
near-null visibility remains undefined; confirmation and gain calibration are distinct gates.
