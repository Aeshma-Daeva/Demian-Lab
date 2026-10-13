# Order and time calibration

Protocol: `DEMIAN_UNCERTAINTY_LOCALIZATION_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_uncertainty_localization_v1_20261013/`.
Astra review: evaluable; registered `R-UNC-LOCALIZATION-001`.
`H-UNC-001` unchanged.

| Question | Observation | Boundary |
| --- | --- | --- |
| New-order calibration at cue6 | All12 full-state projected R² values negative; latest-only semantic baseline positive. | No seed/projection-robust advantage. |
| Cue6-frozen temporal calibration | Fails after2 blank ticks even on the original order. | Coordinate/readout drift, not demonstrated erasure. |
| Time-refitted cross-order decoding | Improves partly at delays32/64, below latest-only baseline typically. | Every-seed/projection criterion fails. |
| Within-order unseen-history availability | Not tested here. | All64 source histories fitted; in-sample reconstruction is not independent evidence. |

Controls: train-only scale/coefficients, matched projection-seed conditional
shuffles, current/latest fitted encodings and analytic single-cue log odds.
Schedule-only is constant within the one-order training set: sanity only.
Early cue2/cue4 contain4/16 unique prefixes and remain descriptive.

State-scale observation: median full-state norm rises approximately14→20→57→296→533
from cue6 through delays2/8/32/64. Order-centroid offsets are small by comparison.
This is not a causal decomposition or proof that translation fixes calibration.

Scope: four fresh seeds, one disjoint order pair, three fixed16-dimensional
projections. Six channel probes are descriptive availability measurements,
not causal ownership, selective routes or useful uncertainty regulation.

Falsifier of robust transfer: valid primary projected probes fail to exceed
analytic latest and matched shuffles across every declared seed/projection;
observed in this protocol.

Heldout-history matrix and independent replication completed:
`DEMIAN_UNCERTAINTY_HISTORY_MATRIX_RESULTS.md`.
Within-order availability separates from order/time calibration. No erasure claim.
