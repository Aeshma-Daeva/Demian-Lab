# Fixed-Point Internal Structure Paper Plan

Working title:

> Fixed Points Are Not Empty: Hidden Internal Structure Behind Apparently
> Static Recurrent Surfaces

Repository:

> <https://github.com/Aeshma-Daeva/Demian-Lab>

## Scientific Hierarchy

**System.** Demian is an experimental discrete-time nonlinear recurrent system:
$z_{t+1}=F(z_t,x_t)$ with exposed readout $y_t=R(z_t)$.

**Observation.** Some runs satisfy surface convergence while measured internal
channels continue to change or separate.

**Core hypothesis.** A convergent readout can coexist with structured,
continuation-relevant internal dynamics.

**Operational AFP.** Require $\Delta y_t \to 0$ and persistent structured
change in $z_t$. Reject numerical drift, ordinary transients, and trivial
accumulation.

**Control.** Full-state versus surface-only continuation establishes whether
$y_t$ is a sufficient continuation state. It is expected that a full restore
contains more information; the control is not the main discovery.

**Interpretation.** A surface fixed point is not necessarily a full-state fixed
point.

**Untested speculation.** Distinct internal regimes may align with Demian's
explicit channels and distinct Jacobian or finite-time Lyapunov signatures.

## Clean Claim

> Projected fixed-point classifications can coexist with measurably different
> internal recurrent regimes. The paper defines a protocol for separating
> surface convergence from internal persistence and for testing channel
> alignment without treating continuation controls as mechanism discovery.

## What This Is Not

Do not frame the paper as:

- Demian beats RNNs, GRUs, LSTMs, Transformers, or Mamba.
- Fixed points are always good.
- Internal richness is consciousness, self-awareness, or intelligence.
- Gate-State Causal Propagation is fully proven.
- Demian v1 is a completed architecture.

Frame it as:

- a measurement paper;
- a warning against over-reading surface attractor labels;
- an artifact-backed methodology for asking whether a static surface hides
  active internal state.

## Experiment Order

1. Surface convergence and classification.
2. Internal-state persistence and change.
3. Same surface class, different internal regimes.
4. Full-state versus surface-only continuation as a sanity/sufficiency control.
5. Channel/component restore and ablation.
6. Perturbation and stability analysis.
7. Jacobian and finite-time Lyapunov analysis as the next rigorous step.

## Evidence Backbone

### 1–3. Surface Convergence, Internal Persistence, And Regime Split

Primary artifact:

- `data/substrate_lab/dual_gru_family_summary.json`

Compact result:

| Substrate | Surface attractor | Interior classes | Key readout |
| --- | --- | ---: | --- |
| `dual_gru_v3b:current` | `FIXED_POINT` in 8/8 | 1 | accumulating fixed point in 8/8; mean message norm `23.1086`; bottleneck entropy `1.7779`. |
| `dual_gru_v3b:tight` | `FIXED_POINT` in 8/8 | 2 | tight and accumulating interiors under the same surface label. |
| `dual_gru_v3b:threshold` | `FIXED_POINT` in 8/8 | 2 | threshold variant splits into tight and accumulating interiors. |
| `dual_gru_v3` | `FIXED_POINT` in 8/8 | 2 | older dual-GRU also separates interior modes. |

Claim supported:

> Same surface attractor class, different internal basin classes.

### 4. Continuation-State Sufficiency Control

Primary artifact:

- `data/substrate_lab/v9_capsule_continuity_20260511/summary.json`
- `docs/CAPSULE_CONTINUITY.md`

Compact result:

| Substrate | Full capsule mean gap | Surface-only mean gap | Interpretation |
| --- | ---: | ---: | --- |
| `demian_native_v9` | `0.0` | at least `0.2262` across sweep | full state resumes; surface alone does not. |
| `v9_five_channel` | `0.0` | at least `0.2746` across sweep | continuity spreads beyond the exposed surface. |

Control interpretation:

> Full restore contains more information than a surface-only zeroed state.
> Divergence establishes that the exposed state is not sufficient for
> continuation; it does not by itself establish AFP dynamics or a novel
> mechanism.

### 5. Channel Restore And Ablation

Use component-only restore and ablation to test whether internal regimes align
with `slow`, `control`, `message`, `carrier`, or `gate`. Treat
channel alignment as a hypothesis until replicated against null controls.

### 6. Perturbation And Stability Baselines

Primary artifact:

- `data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/baseline_comparison_summary.json`

Compact result:

| Substrate | Final gap mean | Recovery-window gap mean | Runs |
| --- | ---: | ---: | ---: |
| `rnn` | `1.19e-7` | `1.11e-4` | 3 |
| `gru` | `5.18e-8` | `1.44e-4` | 3 |
| `lstm` | `4.46e-8` | `6.13e-5` | 3 |
| `diag_ssm` | `0.0650` | `0.00237` | 3 |
| `demian_native_v8` | `0.2206` | `0.00391` | 3 |
| `demian_native_v9` | `0.00322` | `0.000531` | 3 |

Use carefully:

- Plain RNN/GRU/LSTM baselines recover almost exactly under this pseudo-channel
  perturbation protocol.
- Native substrates show larger perturbation consequences.
- This is not a superiority claim; it is a comparison showing why matched
  baselines matter.

### Architecture Context: Transformer And Mamba

Primary artifacts:

- `data/reservoir_batch/batch_summary.json`
- `data/mamba_batch/mamba_batch_summary.json`
- `docs/CLAIMS.md` claims C1 and C2

Use as broader motivation, not the central proof:

- Transformer reservoir runs supported a deterministic period-2 attractor
  observation.
- Mamba self-reference runs landed in a lower-energy fixed-point-like basin
  rather than matching the transformer 2-cycle.
- These results show that surface recurrence primitives differ by architecture,
  so the paper should not treat all recurrence as one thing.

### Negative Control: Gate-State Truth Campaign

Primary artifacts:

- `data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/truth_campaign_summary.json`
- `data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json`

Use as scientific discipline:

- Track B generated interesting internal structure.
- The stricter truth campaign demoted the strong mechanism name when held-out
  strict-profile checks did not pass.
- This protects the fixed-point paper from overclaiming: the paper is about
  measurement and hidden structure, not settled causal mechanism naming.

## Proposed Contribution List

1. A surface/internal distinction for recurrent-system evaluation.
2. An `interior fixed-point` vocabulary separating static surface labels from
   hidden state organization.
3. A compact measurement protocol:
   - surface attractor class;
   - interior class;
   - channel separation;
   - route/channel norms;
   - ablation response;
   - full-state versus surface-only resume.
4. Evidence that several Demian fixed-point surfaces preserve distinct internal
   structure.
5. Matched baselines showing which effects are ordinary recurrent recovery and
   which require further mechanism tests.

## Draft Abstract

Demian is an experimental discrete-time nonlinear recurrent system with full
state $z_t$ and exposed readout $y_t$. We test whether $y_t$ can converge while
$z_t$ retains structured, continuation-relevant dynamics. Dual-GRU artifacts
show one fixed-point surface class with different internal regimes. We define
AFP operationally as surface convergence plus persistent structured hidden-state
change, excluding drift, transients, and trivial accumulation. Channel
ablations and perturbation controls bound the interpretation. Full-state versus
surface-only continuation is a sufficiency control only: full restore contains
more information, and the observed gap shows that the exposed state is not
sufficient for continuation. Jacobian and finite-time Lyapunov analysis remain
future characterization.

## Section Plan

1. Experimental system and scientific hierarchy.
2. Surface dynamics and convergence criteria.
3. Internal-state persistence; AFP operational criteria and exclusions.
4. Same surface class, different internal regimes.
5. Continuation-state sufficiency control.
6. Channel/component restore and ablation.
7. Perturbation and stability analysis.
8. Claim boundaries and negative controls.
9. Next characterization: Jacobians and finite-time Lyapunov estimates.
10. Limitations and reproducibility.

## Figure And Table Plan

- Figure 1: Surface label versus internal-state measurement stack.
- Figure 2: Fixed-point basin interiors in dual-GRU variants.
- Figure 3: Capsule resume schematic: uninterrupted, full capsule,
  surface-only.
- Figure 4: Baseline comparison across RNN/GRU/LSTM/SSM/native substrates.
- Table 1: Definitions: surface fixed point, tight fixed point,
  accumulating fixed point, internal richness, channel separation.
- Table 2: Dual-GRU fixed-point interior summary.
- Table 3: Capsule-continuity resume results.
- Table 4: Baseline perturbation/recovery summary.
- Table 5: Negative checks and claim boundaries.

## Immediate Work Items

1. Generate a compact evidence CSV for the paper tables.
2. Decide whether Transformer/Mamba remain in the main paper or move to an
   appendix.
3. Re-run or validate the most important fixed-point interior artifacts from a
   clean command.
4. Create publication figures from existing SVG/JSON assets.
5. Convert this plan into `papers/fixed_point_internal_structure/main.tex`
   after the evidence tables are frozen.

