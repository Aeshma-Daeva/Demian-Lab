# Fresh short-horizon decoder tracking confirmation

Protocol: `DEMIAN_UNCERTAINTY_TRACKING_CONFIRMATION_PROTOCOL.json`.
Artifacts: `data/diagnostics/demian_uncertainty_tracking_confirmation_matrix_v1_20261014/` and `data/diagnostics/demian_uncertainty_tracking_confirmation_v2_20261014/`.
Precheck: `DEMIAN_UNCERTAINTY_DECODER_TRACKING_PRECHECK.md` records an excluded
secondary-observer description/implementation discrepancy.

## Observation

With four fresh parameter seeds, a new reliability-order pair and split97,
the base-trained 16-D orthogonal tracker wins all 12 within-order cells at
delay2. At delay8 it wins 11/12: seed2194/projection17 obtains R2 `.131`,
below its matched shuffled-label R2 `.155` though above latest-cue R2 `-.0002`.

The unchanged base-order tracker wins 0/12 strict heldout-order cells at both
delay2 and delay8. Identity is strongly miscalibrated; scalar scale correction
is also insufficient. The result stores projections, cue6 readout parameters,
trackers and prediction rows for exact replay.

## Interpretation

The declared all-cell short-horizon confirmation criterion is mixed: it holds
at delay2 but not delay8. The fresh result supports a short-lived within-order
external coordinate correction, with no robust strict order portability. It
does not establish that rotation is necessary, that evidence is erased at the
failure cell, or that Demian internally adapts its own readout.

## Limits and next gate

The 12 projection cells are dependent within four parameter seeds. Reliability
order changes also change the public evidence target for a binary history, so
the cross-order result cannot isolate geometry alone. Do not add routing yet.
The next bounded package should use matched-target cue orders to separate target
change from coordinate deformation before testing controllability.

## Independent review

Astra review completed: both source manifests validate, replay reproduces all
504 metric rows, predictions and 732 saved transform arrays exactly, and the
fit boundary is base-order train-only. The review agrees with `mixed` and no
status promotion. It required direct adversarial tests for heldout-state/label
isolation and replay; those tests now pass.
