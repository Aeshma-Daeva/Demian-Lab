# Reachable-anchor local response

Protocol: `DEMIAN_UNCERTAINTY_RADIUS_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_uncertainty_radius_v1_20261013/`.
Astra review: evaluable; registered `R-UNC-RADIUS-001`. No hypothesis promotion.

- Observation: all768 precision-valid cases satisfy the0.1 relative local-linear
  criterion; maximum mathematical error0.0167.
- Control: all48 native baseline continuations replay exactly. Native inputs
  use actual float32 deltas, without renormalization.
- Precision: all192 radius1e-4 cases miss the10x response/arithmetic-floor margin,
  although approximately94% of coordinates change. These are inconclusive,
  not evidence of fragility or unrepresentable inputs.
- Scope: four fresh parameter seeds, four selected histories, two fixed isotropic
  directions, both signs, horizons2/4/8. Largest radius is only0.35% of state norm.
- Interpretation: sampled local smoothness; no global stability, memory use,
  evidence ownership, chaos or adaptation claim.
- Falsifier: precision-valid sampled responses exceed the declared error bound.

Next: all-history one-cue chords versus norm-matched random directions.
History endpoints are reachable; intermediate chord fractions need not be.
