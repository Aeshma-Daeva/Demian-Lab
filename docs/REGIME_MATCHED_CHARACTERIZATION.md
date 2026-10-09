# Regime-matched characterization

Status: Phase 0 instrumentation validation. Later phases are gated; plasticity is deferred.
Existing 94–193 comparator seeds and 94–96 recovery cells are development evidence.

## Corrected definitions

`F(z,x)` advances complete channel state; `R(z)` is the adapter's continuous exposed vector.
Action logits are a separate readout; argmax action decisions are discontinuous.
Demian's exposed vector is lower-dimensional than its complete state.

- Tangent propagation: `v_next = D_z F(z,x) v`, with identical recorded inputs.
- Directional finite-window rate: sum log tangent growth / measured ticks, retaining negative values.
- Renormalization controls numerical range; multiple initial directions and horizons check convergence.
- A finite-window directional rate is not the largest singular-value FTLE or an asymptotic exponent.
  Full QR spectra require log QR diagonals; summing largest per-block singular values is not equivalent.
- Positive rate: expanding tangent dynamics over the measured window; chaos remains unestablished.
- Readout-relative rate: accumulated state growth plus endpoint log `||J v_unit||` ratio.
  Zero/near-zero initial visibility makes the ratio unresolved, not infinitely informative.
- Delayed sensitivity: `B_k = J_(t+k) Phi(t+k,t)`.
  State-space observability Gramian: `W_o = sum_k B_k^T B_k`, dimension `d x d`.
- Direction time-to-visibility: first `k` where `||B_k v||` exceeds a declared threshold.
  Gramian-column norms do not define visibility time.
- Saturation is an operation-specific derivative measurement, not a large-state-value threshold.
  Gated/additive retention can preserve state even when candidate nonlinearities saturate.
  Instrumentation covers RNN tanh, fused GRU reset/update/candidate, Demian gates,
  cross-routes, compound fast update and exposed-readout operations. Reconstructed operations
  are checked against the actual transition. A projected-unsaturated state experiment remains pending.

## Gates

0. Validate tangent instrumentation against known linear maps and central differences, then
   measure window rates, direction/horizon dependence, state increments and saturation.
   Primary drive is a common cue/delay/query stream with noop acknowledgements; no action feedback.
   Model-specific recorded feedback is a separate conditional-dynamics experiment.
0b. Scale only vanilla RNN `weight_hh` by a declared multiplier. Measure resulting rates;
    multiplier is not spectral radius. Calibrate on development seeds; confirm on fresh seeds.
    Gain matching addresses one confound and does not establish architectural equivalence.
1. Measure immediate and delayed visibility under varied fixed input streams.
   Finite-horizon readout invisibility is not proof of a permanent symmetry or a null space for all inputs.
2. Channel/route interventions with matched subspace and alignment controls.
   Input/output freezing distinguishes interventions, but does not alone identify storage versus transit.
3. Internal plasticity after regime and cue-decoding checks, with reset/frozen/shuffled/linear controls.
   Expanding dynamics do not logically preclude learning; operating-regime choice is an experimental factor.

No confirmation or later phase begins before an explicit previous-phase decision artifact.
Nonmatching default regimes limit interpretation; they do not invalidate the descriptive comparison.

## Claims and statistics

AFP-v2 retains its original requirement: converged exposed trajectory plus structured persistent
internal change, excluding transient settling, drift and trivial accumulation. Delayed visibility is
a functional-state test and additional evidence, not a replacement AFP definition.

Match gain with uncertainty across windows/directions. Paired ablation analyses use paired seed
differences and seed-level bootstrap/permutation; unrelated architecture initializations are not paired
weights merely because seed integers match. Test predefined equivalence margins to support an H0;
failure to reject a difference is inconclusive. Include unresolved outcomes and multiple-test control.
Cliff's delta 0.33, decoder accuracy 80%, and a 10-point advantage are proposed design margins,
not universal scientific thresholds. Later-phase margins must be calibrated on development data
and hashed before confirmation. Channel names remain code identifiers; functional claims require evidence.

Phase 0 settings: [protocol](REGIME_MATCHED_PROTOCOL.json); [development observations](PHASE0_DEVELOPMENT_RESULTS.md).
The campaign verifies known maps, reruns both development horizons, audits complete cells, and
admits confirmation seeds 194–293 only with a valid decision. Direction/horizon tolerances were
development-calibrated and frozen before confirmation. Changed source, protocol, evidence or
confirmation settings invalidate the gate. Later phases remain unauthorized.
Outputs: streamed JSONL, exact source snapshots, frozen protocol, test log, hashed decision and status.
Each worker uses one CPU thread; campaign concurrency is capped at two workers.

Development reproduction (fresh output directory):

```bash
./venv/bin/python -m development.run_regime_characterization --reference-hidden-size 256 --steps 512 --burn-in 128 --out data/diagnostics/regime_phase0_development_512
./venv/bin/python -m pytest tests/test_regime_characterization.py -q
./venv/bin/python -m development.run_regime_campaign --out data/diagnostics/regime_phase0_frozen_campaign
```

Use `--validation-only` for development without automatic confirmation. Never reuse an output directory.

References: [Lyapunov algorithms](https://www.scholarpedia.org/article/Lyapunov_exponent),
[empirical Gramians](https://arxiv.org/abs/1611.00675).
