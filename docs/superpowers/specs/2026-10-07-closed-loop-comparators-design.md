# Closed-Loop Comparator Design

## Goal

Determine whether Demian's six-channel routed state produces reproducible
closed-loop dynamical differences beyond small matched recurrent baselines.

## Scope

First comparator set:

- memoryless MLP sanity control
- vanilla RNN
- GRU
- Demian route/gate disruption control
- Demian reference

Defer LSTM, Mamba, and Transformer until this interface is validated.

## Shared Contract

Every model receives the same bounded world observation frame and produces a
bounded action proposal through an equal-capacity declared action head. The
world, connector, action latency, episode horizon, observation schedules, and
environmental perturbation draws are shared.

Each model declares:

- complete persistent runtime state
- state bytes at each horizon
- parameter count
- action-head accessible state
- checkpoint/restore ownership

The Transformer, if added, must declare a fixed context limit and eviction
rule. The MLP receives no history window or prior action unless that is an
explicit separate control.

## Comparison Tracks

Run separately:

1. Equal complete-state budget.
2. Equal trainable-parameter budget.

Do not call either track complete architectural equivalence.

## Conditions

For each architecture and matched parameter seed:

- baseline
- normalized internal pulse
- register disturbance
- combined perturbation
- fixed-observation replay
- post-input autonomous interval
- exact checkpoint restore

Internal interventions use equal relative state norm. Demian route/gate
disruption is compared with preregistered matched-size random subspace
interventions for non-Demian models.

## Measurements

- correctness, wrong answer, no answer, time to answer, invalid action, and
  storage operations
- exposed and complete-state trajectories
- continuation and replay identity
- immediate and horizon-level perturbation response
- state bytes, parameter count, and measured compute

## Interpretation Rules

- Model initialization, initial state, environment schedule, and training seed
  are separate replication axes.
- Untrained runs measure initialized dynamics and interface behavior only.
- A controllability claim requires a preregistered bounded intervention,
  reliable regime transition, persistence interval, and held-out transfer.
- The existing 100-seed closed-loop runs are development evidence, not a
  comparator benchmark.

## Archived Context

`development/substrate_lab.py` supplies self-loop RNN/GRU/LSTM/SSM methods and
continuation/perturbation precedents. Archived Transformer and Mamba reservoirs
are historical contrast evidence. Neither is pooled with the new closed-loop
results.
