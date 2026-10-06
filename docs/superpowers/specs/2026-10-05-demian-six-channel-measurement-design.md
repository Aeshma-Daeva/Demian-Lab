# Demian Six-Channel Measurement Design

## Objective

Build a validated measurement harness for `DemianV1GateState` before expanding
into task, uncertainty, evolution, or stability campaigns. AFP-v2 is one output
of the harness, not the definition of Demian.

## Scientific scope

The measured state is:

```text
z = (fast, slow, control, message, carrier, gate)
```

The exposed surface is a projection of `fast`, `message`, `carrier`, and `gate`.
It is not a seventh state channel. Results must separately report:

1. surface regime;
2. full-state and per-channel regime;
3. continuation causality;
4. task utility, when a task exists.

Phase one covers the first three. It does not claim memory, uncertainty
representation, or task utility.

## Required contracts

### Surface matching

`DemianV1GateState` needs an exact surface-matching adapter. Given a target
surface and retained `slow`, `control`, `message`, `carrier`, and `gate`, the
adapter solves for `fast` after subtracting the other readout contributions.
It must verify the reconstructed surface within floating-point tolerance.

### Runtime continuation

A continuation checkpoint includes:

- all six state tensors;
- model weights;
- `_step_index`;
- `_frozen_gate` when present;
- constructor operating mode.

Continuation comparisons must use the same step index and operating mode.

### Intervention semantics

Two intervention families remain distinct:

- `update_mode`: active, constructor gate-disabled, or constructor gate-frozen;
- `post_step_clamp`: clamp a named channel after each update.

Reports must not describe a post-step clamp as equivalent to disabling or
freezing the update rule.

### Trace schema

Each run records:

- surface at every step;
- all six state channels;
- complete concatenated state;
- step metrics;
- seed, hidden size, horizon, parameter provenance, input/history condition;
- update mode and post-step intervention.

Parameter provenance and trajectory history are independent factors.

## Phase-one campaign

Use unselected random parameters only:

- six-channel Demian v1;
- five-channel v9 predecessor as a historical control;
- seeds `94,95,96`;
- hidden size `16`;
- `512` steps;
- uninterrupted and surface-perturbed histories;
- perturbation at step `256`, scale `0.05`;
- active, gate-disabled, and gate-frozen six-channel update modes.

Apply the current AFP-v2 classifier to surface, full state, and each channel.
Run continuation controls over the same steps as the classified tail. Include
full checkpoint, exact body+surface, individual-channel restore, sham restore,
and norm-matched random-state controls.

The campaign may compare classifications but must not claim architectural
superiority because parameter count, state size, and search budget are not yet
matched.

## Robustness requirements

Report sensitivity to:

- tail horizons `24,48,96`;
- float32 numerical scale;
- AFP-v2 thresholds at `0.5x`, `1x`, and `2x`;
- held-out seeds separate from any future selection process.

An AFP-v2 pass remains a finite-horizon operational result. A failure is not an
architectural failure.

## Evidence labels

Artifacts keep separate fields for:

- observation;
- hypothesis;
- control;
- interpretation;
- untested speculation.

Continuation divergence establishes causal influence on continuation only. It
does not establish useful memory or uncertainty representation.

## Deferred phases

After phase one:

1. matched-surface states and readout-null microperturbations;
2. one delayed-memory task with delay extrapolation;
3. calibrated uncertainty task with proper scoring;
4. freeze, disable, restore, and rescue tests on reproducible effects;
5. local Jacobian and finite-time Lyapunov characterization;
6. evolved and task-trained provenance with held-out budgets and controls.

