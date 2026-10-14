# External decoder tracking

Protocol: `DEMIAN_UNCERTAINTY_DECODER_TRACKING_PROTOCOL.json`.
Artifact: `data/diagnostics/demian_uncertainty_decoder_tracking_v1_20261014/`.
This is an analysis-only external-observer result. Demian architecture and
weights were unchanged; no readout fed back into the system.

## Observation

The frozen 16-D cue6 decoder plus train-only translation/scale/proper-rotation
tracking wins against both analytic latest-cue and matched shuffled-label
controls in all 24 seed-by-projection within-order cells through delay8. At
delay32 it wins 12/24; at delay64, 0/24. Across the heldout cue order it wins
12/24, 16/24, 22/24, 10/24, and 0/24 cells from cue6 through delay64.

The dynamic external observers are feasible but not self-regulation. The
checkpoint-conditioned observer has mean R2 around `.54-.61` within order and
`.39-.58` across order. The small state-regime-conditioned observer is weak at
cue6 through delay8 and reaches about `.60` only at delay32; it adds no
evidence that Demian uses this feature map.

## Interpretation

Low-rank offline alignment can preserve a frozen cue6 coordinate through short
delays within the base cue order. It is insufficient as a robust late or
cross-order coordinate correction. An external time-indexed readout can recover
some matched-history evidence where a single frozen coordinate fails; that is a
decoder capability, not an internal adaptive mechanism.

## Limits / next gate

All fits, transforms and labels use only allocated base-order train histories;
the two source batches were already inspected, so this is exploratory. No
status change. Freeze a single short-delay and late-delay prediction from this
package, then confirm on new parameter seeds and a new order pair before any
claim about systematic coordinate tracking or a future routing controller.
