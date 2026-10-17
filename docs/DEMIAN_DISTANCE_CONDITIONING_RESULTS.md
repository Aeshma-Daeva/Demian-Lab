# Fresh distance-conditioning test

Protocol: `DEMIAN_DISTANCE_CONDITIONING_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_distance_conditioning_v1_20261017/`.
Source frozen before collection: `ae8f2a0`.
Six fresh evaluation seeds; original four fitting seeds and disjoint history
allocation. Architecture, weights and encoder unchanged; external transforms only.

## Observation

Median heldout R²; `conditioned` is the sole declared primary variant:

| System | Delay | Distances | Squared contrasts | Conditioned contrasts | Primary passing seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demian | 2 | .297 | .400 | .235 | 5/6 |
| Demian | 8 | .330 | .394 | -.286 | 0/6 |
| Demian | 32 | .375 | .510 | -7.560 | 0/6 |
| Demian | 64 | .407 | .456 | -.570 | 1/6 |
| Reservoir | 2 | .947 | .957 | .966 | 6/6 |
| Reservoir | 8 | .862 | .841 | .860 | 6/6 |
| Reservoir | 32 | .738 | .650 | -4875.682 | 0/6 |
| Reservoir | 64 | .430 | .258 | -21158.860 | 0/6 |

Both systems fail the overall primary gate; no seed passes all four delays.
Ordinary distances reproduce the previous Demian criterion failure and
reservoir6/6 through32,4/6 at64.

Demian's delay2 criterion pass is not improved prediction: median true R² falls
while incorrect-anchor permutation q95 falls to approximately-1522. The
transformation makes incorrect correspondences costly without improving
the primary task score. Alternative variants cannot rescue the frozen gate.

## Conditioning audit

Post-result `support_audit.json` measures squared feature norm after the frozen
whitener relative to pooled fitting-history energy; no refitting.
Median evaluation/training ratios at delays2/8/32/64:

- Demian:4.2/9.1/65.4/103.6.
- Reservoir:2.4/1.5/42281/223702.

At reservoir delay32, whitening retains10 contrast directions with maximum
inverse scale3727, versus9 directions and13.8 for the affine-state surrogate.
That surrogate retains median R² .643 and energy ratio2.38. Amplified residual
directions and feature extrapolation accompany failure; their origin is not
established as numerical noise or a channel mechanism.

## Interpretation and review

Squared contrasts expose signed anchor-span coordinates, including seed-specific
anchor offsets. The fixed whitening policy does not generalize reliably across
fresh seeds. Unwhitened contrasts modestly improve Demian's prediction here;
this remains a prespecified secondary comparison, not the primary confirmation.

Astra independently verified214 manifest hashes,160 frozen-fit hashes,
1,440 prediction arrays and480 metric/gate rows. Review returned; no invalidating
implementation issue found. See `review.json`. Fitted objects and predictions
are versioned; raw states/parameters remain local and hash-listed.

Falsifier: prospective replication fails these declared contrasts. No architecture,
nonlinear-computation, autonomous adaptation, AFP or causal-channel conclusion.
An architectural extension is not justified by this readout result.
