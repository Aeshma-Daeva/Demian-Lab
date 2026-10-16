# Demian internal dynamics: evidence-bounded characterization

Scope: the untrained six-channel `DemianV1GateState`, hidden width 256. This
is a synthesis of reviewed artifacts, not a new hypothesis promotion.

## System

`z_(t+1) = F(z_t, x_t)` has six state owners: `fast`, `slow`, `control`,
`message`, `carrier`, and `gate`. The exposed surface is a readout of the full
state, not the full state itself. Explicit flow is
`fast -> message -> carrier -> slow`; `control/message/carrier/slow -> gate`;
the gate modulates implemented routes. The full state has 1,312 coordinates at
width 256 (five 256-wide channels plus 32 gate coordinates).

## What is measured

| Axis | Observation | Boundary |
| --- | --- | --- |
| Surface versus full state | Current fixed-forcing runs reached no declared surface-convergence window. | No AFP-v2 evidence, positive or negative. |
| Local dynamics | A fixed full-state Jacobian predicts sampled two-step motion across history/order axes; it fails by sampled horizon 4 and all cells fail by 8. Second-order curvature reduces, but does not eliminate, later error. | This is ordinary local nonlinear flow, not chaos or a special memory mechanism. |
| Within-order geometry | Cue-to-delay clouds expand strongly: median scale ratio about 1.18, 3.04, 39.9, 220 at delays 2, 8, 32, 64. Pairwise structure progressively deforms. | Expansion is not information loss by itself. |
| Across-order geometry | Matched clouds are near-isometric after translation, scale, and orthogonal alignment, yet frozen coordinate readouts often fail to transfer. | Relative geometry can persist while absolute coordinates move. Rotation is not shown to be the cause. |
| Evidence availability | In two independently allocated batches, train-only 16-D ridge probes beat latest-cue and conditional-shuffle controls at matched time within a fixed order through 64 blank ticks. | This is externally decoded availability; not use, memory ownership, or adaptation. |
| Route consequence | Gain reduction of `carrier_to_slow` reliably changes future full-state trajectories under fixed forcing. | It does not identify carrier-specific information transmission or mediation. |
| Route selectivity | Carrier-versus-random directional contrast reverses across seed splits. | No stable carrier-specific selectivity. |
| Gate association | Active, disabled, and frozen modes differ in sampled regime frequencies. Gate-to-channel co-change directions are heterogeneous across seeds. | Observational association only; no causal gate mechanism is nominated. |
| Continuation | Full-runtime restore is exact; reconstructions that retain less internal state diverge differently. | Reconstruction-dependent divergence does not prove a hidden manifold or universal surface insufficiency. |

## Recent externally read delayed-cue check

Artifact: `data/diagnostics/demian_frozen_readout_delayed_cue_v1_20261016/`.
The recurrent system and encoder are frozen; only a 16-D external ridge probe
is fit on declared training histories. This implementation reads the exposed
256-D surface, not the full state. It exceeds recency and fixed no-storage
controls in all six seeds at delays 2 and 32, but fails the predeclared
five-of-six margin gate at delay 8. Distractor histories were independently
sampled by delay, so this is not a clean time-course or retention curve. It
does not show intrinsic storage, a task policy, or a full-state readout result.

Corrected matched-prefix full-state artifact:
`data/diagnostics/demian_frozen_full_state_readout_delayed_cue_v2_corrected_20261016/`.
Six fresh seeds receive identical length-32 distractor histories at delays 2,
8, and 32; the probe sees all 1,312 state coordinates through a fixed 16-D
projection. Only delay 32 passes the declared all-seed gate (6/6; .931–1.000).
Delays 2 and 8 fail. A descriptive per-channel audit on the same states finds
that `fast` alone is decodable at delay 2 (mean .914) while the fixed 16-D
full-state projection is not (mean .361). With corrected state slices, means
at delay 32 range .896–.991 across separately fitted channel probes. These
post-hoc, shared-episode probes establish only projection/source-sensitive
external availability—not channel ownership, causal routing, or a channel
ranking.

## Current picture

Demian is a bounded nonlinear recurrent system whose fixed parameter seeds can
retain externally decodable, history-dependent differences while the full state
undergoes large, nonlinear, order-dependent movement. Its measured state code
is contextual rather than presently coordinate-portable. Implemented routes and
gate modes change dynamics, but the evidence does not yet localize retained
information to a channel or establish a regulating mechanism.

## Not established

- asymptotic fixed points, limit cycles, chaos, or Lyapunov spectrum;
- AFP-v2, since no valid surface-settled persistence windows exist;
- channel-level causal evidence ownership or route mediation;
- an internal decision/action policy, learning, adaptation, or smart routing;
- architectural advantage over matched recurrent baselines.

## Evidence references

- `DEMIAN_UNCERTAINTY_HISTORY_MATRIX_RESULTS.md`
- `DEMIAN_UNCERTAINTY_REPRESENTATION_RELATIONS_RESULTS.md`
- `DEMIAN_UNCERTAINTY_JACOBIAN_RESULTS.md`
- `DEMIAN_UNCERTAINTY_LOCAL_MODELS_RESULTS.md`
- `DEMIAN_ROUTE_STABILITY_RESULTS.md`
- `DEMIAN_ROUTE_SELECTIVITY_RESULTS.md`
- `data/RESEARCH_REGISTRY.json`

Falsifier for the central synthesis: a protocol-matched replication fails to
find delayed within-order availability, state expansion/deformation, or
route-consequential perturbation under the declared controls.
