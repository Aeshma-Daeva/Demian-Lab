# Route Feature Audit: Discovery Result

Protocol: `DEMIAN_ROUTE_FEATURE_AUDIT_PROTOCOL.json`
Valid artifact: `demian_route_feature_audit_v2_20261012`

## Observation

- 144 matched cells: 8 parameter seeds × 3 cue histories × 3 directions × 2
  random-pulse scopes.
- The carrier-minus-random interaction ordering remains mixed. With the fixed
  16-transition window, mean gaps were negative in development and positive in
  held-out seeds for both within-carrier and full-state random controls.
- Pre-pulse features vary across parameter seeds. The late cue-history variants
  produce only small changes in the recorded state features.

## Exploratory associations

Pooled seed-history summaries show associations between interaction ordering
and gate/route features. They are not predictive evidence: features were not
frozen, observations share seeds, and pooled associations can encode parameter
identity rather than state dependence.

## Interpretation

This audit does not establish operating regimes, a predictor, selectivity, or
controllability. It identifies a design requirement for confirmation: use
richer reachable histories or an earlier probe, freeze a compact feature rule,
and hold out whole parameter seeds.

## Boundary

No claim about mechanism, regulation, learning, adaptation, memory, task
competence, AFP-v2, or chaos.
