# Labbook

Last updated: 2026-10-12

## Purpose

This is the append-only experiment chronology. Fast-moving run details belong
here, not repeated across every evergreen doc. Promote only durable claims into
[docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md), and index artifacts in
[data/INDEX.md](/home/xenith/demian/data/INDEX.md) and
[data/MANIFEST.json](/home/xenith/demian/data/MANIFEST.json).

## Entry Format

Each entry should include:

```text
date
run id
purpose
config
artifacts
result
interpretation
next check
```

## 2026-05-11 - Gate-State Propagation Characterization

Purpose:

- Characterize the top native-emergence Track B genotype before costly
  replications.
- Test whether gate-state propagation is a real internal mechanism through
  route/gain-zero diagnostics, channel-disabled continuation, capsule-style
  resumes, and parameter-signature extraction.

Config:

```text
candidate: data/evolution/demian_v2_track_b_island_1_20260511/candidates/gen012_candidate000.json
seeds: 94-102
perturb scales: 0.2, 0.35, 0.7
motifs: basis:0, gaussian:0
steps: 128
perturb step: 64
conditions: original, routes_disabled, gain_zero, message_disabled, carrier_disabled, control_disabled, slow_disabled
device: cpu
```

Artifacts:

- [data/diagnostics/gate_state_propagation_characterization_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/summary.json)
- [data/diagnostics/gate_state_propagation_characterization_20260511/ablation_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/ablation_summary.csv)
- [data/diagnostics/gate_state_propagation_characterization_20260511/ablation_results.parquet](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/ablation_results.parquet)
- [data/diagnostics/gate_state_propagation_characterization_20260511/capsule_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/capsule_summary.csv)
- [data/diagnostics/gate_state_propagation_characterization_20260511/capsule_results.parquet](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/capsule_results.parquet)
- [data/diagnostics/gate_state_parameter_signatures_summary.json](/home/xenith/demian/data/diagnostics/gate_state_parameter_signatures_summary.json)

Result:

```text
ablation runs: 378
ablation rows: 48384
capsule rows: 432
gain_zero_clean: true
routes_disabled_divergence_positive: true
gain_zero_divergence_positive: true
full_resume_exact: true
surface_resume_gap_positive: true
```

Interpretation:

- This is candidate evidence for gate-state propagation, not a promoted claim.
- Gain-zero and route-disabled divergence are both positive while gain-zero
  release strength/routes remain cleanly zero.
- Channel-disabled probes show the largest causal divergence in the
  `message_disabled` and `carrier_disabled` arms under this grid, with `slow`
  also necessary.
- Full internal-state resume is exact; surface-only and channel-only resumes
  retain substantial final gaps.

Next check:

- Run three native-emergence Track B replications and promote only if the same
  necessary-channel pattern appears in at least two runs with clean gain-zero
  diagnostics and positive held-out divergence.

## 2026-05-11 - Track B Gate-State Replications

Purpose:

- Run the three native-emergence Track B replications required before promoting
  Gate-State Causal Propagation.
- Classify each top candidate with the mechanism labels learned from the first
  characterization pass.

Config:

```text
replications: 3
population: 16
generations: 20
rank mode: native_emergence
initial population: random only
operators: mutation + crossover
paired causal evaluation: enabled
eval seeds: 94, 95
held-out classification seeds: 96, 97, 98
held-out classification scales: 0.35, 0.7
default seed: disabled
elitism: disabled
random injection overlay: disabled
structured operators: disabled
```

Artifacts:

- [data/evolution/demian_v2_track_b_replication_1_20260511/archive.json](/home/xenith/demian/data/evolution/demian_v2_track_b_replication_1_20260511/archive.json)
- [data/evolution/demian_v2_track_b_replication_2_20260511/archive.json](/home/xenith/demian/data/evolution/demian_v2_track_b_replication_2_20260511/archive.json)
- [data/evolution/demian_v2_track_b_replication_3_20260511/archive.json](/home/xenith/demian/data/evolution/demian_v2_track_b_replication_3_20260511/archive.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/replication_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/replication_summary.csv)

Result:

```text
passed replications: 3/3
mean held-out route divergence: 0.2997
mean held-out gain-zero divergence: 0.2997
gain-zero clean: 3/3
full internal-state resume exact: 3/3
surface-only resume gap positive: 3/3
message/carrier top-two necessary channels: 3/3
```

Interpretation:

- Gate-State Causal Propagation is replicated as a Track B native-emergence
  mechanism.
- The replicated path is message/carrier gate-state propagation with slow
  continuation support.
- The top replicated candidates are high-duty phenotypes, so this does not
  solve the sparse delayed release target.

Next check:

- Split future metrics so route-specific causal release and gain-zero
  gate-state propagation are scored separately.

## 2026-05-10 - v10.0 Frozen Evolution Predecessor Evidence

Purpose:

- Replace iterative scoring tweaks with actual selection pressure over time.
- Check whether sparse-ish release phenotypes amplify, stabilize, or get
  outcompeted by flood genotypes.
- Track lineage, event/phase scatter, duty histograms, and regime distribution.
- Preserve this as predecessor evidence for `Demian v1`, not as a new native
  substrate generation.

Config:

```text
experiment: v10.0-frozen-evolution
next_program: demian-v1
islands: 4
population: 8 per island
generations: 20
eval seed: 94
hidden size: 24
steps: 64
perturb step: 32
perturb channels: fast, carrier
perturb modes: clean, external
device: cuda:0
```

Artifacts:

- [data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json)
- [data/evolution/v10_0_frozen_evolution_island_1_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_1_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_2_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_2_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_3_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_3_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_4_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_4_20260510/diagnostics.json)

Result:

```text
candidate count: 640
final generation index: 19
duty first5 -> last5:  0.283 -> 0.214
event first5 -> last5: 0.084 -> 0.266
phase first5 -> last5: 0.152 -> 0.450
```

Final/best candidate:

```text
island_1 gen019_candidate001
rank=5.8039
duty=0.171875
release_geometric_event=0.7896
phase_transition_score=0.9371
regimes: bounded_strange=42, surface_fixed_accumulating=6
ancestor=gen000_candidate004
mutation_count=7
reproduction=elite_copy from gen018_candidate003
```

Interpretation:

- The flood penalty had real selection effect: high duty did not dominate.
- Release did not disappear; the final top candidates sit in sparse-to-borderline
  duty bands with strong event and phase scores.
- Top-10 lineage diversity contracted while mutation depth increased, so late
  winners are inherited lineages rather than only fresh random filters.
- The v9.3 sparse/high-phase phenotype survived as a heritable basin, but this
  predecessor run selected a stronger sparse event/phase hybrid.

Next check:

- Cross-validate the predecessor winner and near-winners on held-out eval seeds
  before treating genotype-level rank ordering as stable.
- Inspect final and runner-up lineages to isolate how sparse release produces
  event/phase gains.

## 2026-05-09 - Full v9 Five-Channel Evolution

Purpose:

- Establish whether message/carrier/release evolution can preserve multiple
  bounded regimes rather than optimizing a single surface class.

Artifacts:

- [data/evolution/v9_5ch_release_20260509_full/archive.json](/home/xenith/demian/data/evolution/v9_5ch_release_20260509_full/archive.json)
- [data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json](/home/xenith/demian/data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json)

Result:

- The archive preserved `surface_fixed_accumulating`, `bounded_strange`,
  `edge_of_chaos`, and small limit-cycle pockets.
- Release evolved toward rarer openings, but release-local causal effect
  remained weak.

Interpretation:

- Fixed-point surface behavior can coexist with rich internals.
- The next pressure should improve release-local consequence without turning
  release into always-on coupling.

Next check:

- Use the v10.0 predecessor evidence and held-out eval seeds to determine which
  sparse release lineages remain stable.

## 2026-05-13 - Identity Continuity Characterization

Purpose:

- Test whether the current Track B v9 five-channel/Demian v1 scaffold shows
  early evidence for identity continuity, boundary discrimination, or primitive
  self-maintenance beyond control-channel norm recovery.

Artifacts:

- [development/identity_continuity_characterization.py](/home/xenith/demian/development/identity_continuity_characterization.py)
- [tests/test_identity_continuity_characterization.py](/home/xenith/demian/tests/test_identity_continuity_characterization.py)
- [data/diagnostics/identity_continuity_20260513/README.md](/home/xenith/demian/data/diagnostics/identity_continuity_20260513/README.md)
- [data/diagnostics/identity_continuity_20260513/identity_summary.json](/home/xenith/demian/data/diagnostics/identity_continuity_20260513/identity_summary.json)
- [data/diagnostics/identity_continuity_20260513/identity_summary.csv](/home/xenith/demian/data/diagnostics/identity_continuity_20260513/identity_summary.csv)
- [data/diagnostics/identity_continuity_20260513/identity_rows.jsonl](/home/xenith/demian/data/diagnostics/identity_continuity_20260513/identity_rows.jsonl)
- [data/diagnostics/identity_continuity_20260513/identity_signatures.json](/home/xenith/demian/data/diagnostics/identity_continuity_20260513/identity_signatures.json)

Run shape:

```text
candidate: gen012_candidate000
seeds: 94-102
scales: 0.2, 0.35, 0.7
motifs: basis:0, gaussian:0
steps: 128
burn-in: 32
perturb/lesion step: 64
tail window: 16
trace rows: 67,392
summary rows: 702
identity signatures: 702
```

Method:

- Derive an endogenous identity signature from the burn-in means of `slow`,
  `control`, and `carrier`.
- Use `lifetime` probes to compare preserved, reset, disabled, and scrambled
  identity-channel conditions.
- Use `boundary` probes to compare internally sourced versus externally injected
  perturbations of matched schedule and scale.
- Use `lesion` probes to zero or scramble identity-bearing channels and measure
  tail return toward the burn-in signature.

First read:

- `identity_preserved` mean tail cosine: `0.4311`.
- `identity_reset` mean tail cosine: `0.2011`.
- `identity_scrambled` mean tail cosine: `-0.0004`.
- Boundary source means are nearly identical: external tail cosine `0.4311`,
  internal tail cosine `0.4329`.
- Lesion results do not show a clear repair result; `identity_scramble` collapses
  cosine near zero, while `slow_zero` lowers distance partly by shrinking the
  identity vector magnitude.

Interpretation:

- Observation: the full instrumented sweep completed and produced usable trace,
  summary, and signature artifacts.
- Inference: the candidate shows weak/partial identity-continuity signal by
  cosine, but not strong self-maintenance.
- Negative result: the current boundary probe does not support source
  discrimination, and the lesion probe does not support a repair claim.

Discipline note:

- GitHub issue/project-board tracking remains blocked in the local environment:
  `GITHUB_PROJECT_NUM` and `GH_PROJECT_OWNER` are unset, and `gh auth status`
  reports an invalid token.

Next check:

- Repeat the identity-continuity characterization on the Track B replicated top
  candidates before promoting any mechanism claim.

## 2026-05-13 - Identity Continuity Replication Check

Purpose:

- Test whether the initial preserved-over-reset identity-cosine signal survives
  across the three Track B replicated top candidates.

Artifacts:

- [data/diagnostics/identity_continuity_replication_1_20260513/identity_summary.json](/home/xenith/demian/data/diagnostics/identity_continuity_replication_1_20260513/identity_summary.json)
- [data/diagnostics/identity_continuity_replication_2_20260513/identity_summary.json](/home/xenith/demian/data/diagnostics/identity_continuity_replication_2_20260513/identity_summary.json)
- [data/diagnostics/identity_continuity_replication_3_20260513/identity_summary.json](/home/xenith/demian/data/diagnostics/identity_continuity_replication_3_20260513/identity_summary.json)
- [data/diagnostics/identity_continuity_replication_summary_20260513.json](/home/xenith/demian/data/diagnostics/identity_continuity_replication_summary_20260513.json)
- [data/diagnostics/identity_continuity_replication_summary_20260513.csv](/home/xenith/demian/data/diagnostics/identity_continuity_replication_summary_20260513.csv)

Run shape:

```text
replication 1: gen016_candidate012, 67,392 trace rows
replication 2: gen019_candidate008, 67,392 trace rows
replication 3: gen017_candidate000, 67,392 trace rows
summary rows per run: 702
protocol: same as identity_continuity_20260513
```

Result:

```text
preserved-minus-reset tail cosine:
original gen012_candidate000: +0.2300
rep1     gen016_candidate012: -0.1010
rep2     gen019_candidate008: +0.0986
rep3     gen017_candidate000: -0.0452

boundary internal-minus-external tail cosine:
original: +0.0018
rep1:     -0.0003
rep2:     -0.0002
rep3:     -0.0001
```

Interpretation:

- The identity-cosine signal does not replicate cleanly: preserved beats reset in
  2/4 candidates and loses in 2/4.
- Boundary discrimination is consistently absent under this probe.
- Do not promote identity continuity, boundary formation, or self-maintenance as
  replicated Track B findings.

## 2026-05-13 - Sparse Low-Duty Identity Probe

Purpose:

- Test whether a sparse morphology-low-duty candidate has different
  identity-continuity behavior than the dense gate-state propagator candidates.

Candidate:

- `data/evolution/track_b_discovery_20260513/track_b_seed_motif_timing_sweep_09/candidates/gen011_candidate014.json`
- Archived metrics: `release_duty_cycle=0.07877604166666667`,
  `internal_richness=1.0`

Artifacts:

- [data/diagnostics/identity_continuity_sparse_low_duty_20260513/README.md](/home/xenith/demian/data/diagnostics/identity_continuity_sparse_low_duty_20260513/README.md)
- [data/diagnostics/identity_continuity_sparse_low_duty_20260513/identity_summary.json](/home/xenith/demian/data/diagnostics/identity_continuity_sparse_low_duty_20260513/identity_summary.json)
- [data/diagnostics/identity_continuity_sparse_low_duty_20260513/identity_summary.csv](/home/xenith/demian/data/diagnostics/identity_continuity_sparse_low_duty_20260513/identity_summary.csv)
- [data/diagnostics/identity_continuity_sparse_low_duty_comparison_20260513.json](/home/xenith/demian/data/diagnostics/identity_continuity_sparse_low_duty_comparison_20260513.json)
- [data/diagnostics/identity_continuity_sparse_low_duty_comparison_20260513.csv](/home/xenith/demian/data/diagnostics/identity_continuity_sparse_low_duty_comparison_20260513.csv)

Run shape:

```text
candidate: gen011_candidate014
trace rows: 67,392
summary rows: 702
protocol: same as identity_continuity_20260513
```

Result:

```text
sparse low-duty lifetime:
identity_preserved tail cosine: 0.3133
identity_reset tail cosine:     0.3062
identity_scrambled tail cosine: 0.2506
preserved-minus-reset:          +0.0072

boundary:
external tail cosine: 0.3133
internal tail cosine: 0.3141
internal-minus-external: +0.0007
```

Interpretation:

- The sparse candidate has higher absolute preserved tail cosine than the
  gate-candidate mean, and substantially lower identity distance under this
  metric scale.
- Reset identity is nearly as good as preserved identity, so this is not strong
  evidence that sparse dynamics preserve an ongoing identity better.
- Boundary discrimination remains absent.
- Treat this as a useful morphology contrast and a reason to test a sparse
  candidate set, not as a promoted sparse-gating identity claim.

## 2026-05-14 - Control Capsule Maintenance, Setpoint, and Narrative Update Probes

Purpose:

- Test whether the replicated Track B top candidates support three increasingly
  strong continuity claims: capsule maintenance, directional setpoint-like
  regulation, and structured narrative-like history integration.

Artifacts:

- [development/control_capsule_maintenance.py](/home/xenith/demian/development/control_capsule_maintenance.py)
- [development/control_setpoint_probe.py](/home/xenith/demian/development/control_setpoint_probe.py)
- [development/control_narrative_update_probe.py](/home/xenith/demian/development/control_narrative_update_probe.py)
- [tests/test_control_capsule_maintenance.py](/home/xenith/demian/tests/test_control_capsule_maintenance.py)
- [tests/test_control_setpoint_probe.py](/home/xenith/demian/tests/test_control_setpoint_probe.py)
- [tests/test_control_narrative_update_probe.py](/home/xenith/demian/tests/test_control_narrative_update_probe.py)
- [data/diagnostics/control_capsule_maintenance_20260514/control_capsule_summary.json](/home/xenith/demian/data/diagnostics/control_capsule_maintenance_20260514/control_capsule_summary.json)
- [data/diagnostics/control_setpoint_probe_20260514/control_setpoint_summary.json](/home/xenith/demian/data/diagnostics/control_setpoint_probe_20260514/control_setpoint_summary.json)
- [data/diagnostics/control_narrative_update_probe_20260514/control_narrative_summary.json](/home/xenith/demian/data/diagnostics/control_narrative_update_probe_20260514/control_narrative_summary.json)

Run shape:

```text
candidates: gen016_candidate012, gen019_candidate008, gen017_candidate000
held-out seeds: 96, 97, 98
scales: 0.35, 0.7
motifs: basis:0, gaussian:0
capsule channels: slow + message + carrier
maintenance/setpoint steps: 128
narrative steps: 160
narrative event steps: 64, 96
narrative histories: none, A, B, A_then_B, B_then_A, A_then_A, B_then_B
```

Result:

```text
control_capsule rows: 252
control_setpoint rows: 648
control_narrative rows: 504

maintenance aggregate:
  control_zero_index_mean: 0.1270
  control_clamped_zero_index_mean: 0.4178

setpoint aggregate:
  control_specific_correction_mean: 8.9769
  control_preserved_directional_asymmetry_mean: 0.00699
  control_preserved_abs_setpoint_bias_mean: 0.00549

narrative aggregate:
  control_preserved_classification_accuracy_mean: 0.0
  control_preserved_separation_margin_mean: -0.2029
  control_preserved_order_sensitivity_mean: 0.0137
```

Interpretation:

- The capsule-maintenance probe supports control as an active maintenance
  contributor: continuous control clamping degrades continuation more strongly
  than one-shot control zeroing.
- The directional setpoint probe supports control-specific correction, but its
  low asymmetry and low setpoint bias read as symmetric stabilization rather
  than directional homeostasis.
- The narrative-update probe does not show structured history integration in
  the explicit `control` channel: same-history control updates are not more
  separable than cross-history updates, and `A_then_B` vs `B_then_A` order
  sensitivity is small.
- Evidence stack after this run: capsule memory yes; active continuity
  maintenance yes; setpoint/homeostasis no; narrative-like integration no.

Next check:

- If pursuing the narrative hypothesis further, test an explicit added
  narrative state variable or a plastic-state-centered protocol rather than
  promoting the existing control channel as a narrative carrier.

## 2026-05-15 - Gate-State Truth Campaign, Original Three Candidates

Purpose:

- Test whether "Gate-State Causal Propagation" is distinct from ordinary
  recurrent history or broad five-channel recurrence.
- Freeze calibration thresholds before held-out testing.
- Compare Track B positives against random/default controls and recurrence
  baselines.

Artifacts:

- [development/gate_state_truth_campaign.py](/home/xenith/demian/development/gate_state_truth_campaign.py)
- [tests/test_gate_state_truth_campaign.py](/home/xenith/demian/tests/test_gate_state_truth_campaign.py)
- [data/diagnostics/gate_state_truth_campaign_20260515/truth_campaign_summary.json](/home/xenith/demian/data/diagnostics/gate_state_truth_campaign_20260515/truth_campaign_summary.json)
- [data/diagnostics/gate_state_truth_campaign_20260515/locked_thresholds.json](/home/xenith/demian/data/diagnostics/gate_state_truth_campaign_20260515/locked_thresholds.json)

Run shape:

```text
Track B candidates available locally: 3
calibration Track B: 1
held-out Track B: 2
random unselected controls: 1000
default/default-jitter controls: 100
total candidates characterized: 1103
```

Result:

```text
held-out Track B broad mechanism pass: 2/2
held-out random controls broad mechanism pass: 440/700
held-out default-jitter broad mechanism pass: 59/70

locked strict-profile thresholds:
  min_route_divergence: 0.2417293204225006
  min_gain_zero_divergence: 0.2417293204225006
  min_message_carrier_dominance: 0.44469043517608386

held-out Track B strict-profile pass: 0/2
held-out random controls strict-profile pass: 20/700
held-out all controls strict-profile pass: 20/770
```

Interpretation:

- The broad gate was too permissive and cannot support statistical rarity
  language or a strong named mechanism claim.
- The result is underpowered for Track B because the campaign had only 1
  calibration and 2 held-out Track B candidates.
- Treat the strong "Gate-State Causal Propagation" name as unestablished.
- Keep the weaker "structured recurrent history propagation" interpretation
  unless a larger held-out campaign reverses it.

Next check:

- Generate the missing Track B candidates and rerun with enough held-out
  positives to test strict-profile pass/fail structure.

## 2026-05-16 - Multi-Timepoint Surgery On Original Three Track B Candidates

Purpose:

- Re-test state surgery at multiple pause points instead of one final-gap view.
- Check the anatomy reframe: message/carrier may steer early dynamics while
  slow/control may stabilize later dynamics.

Artifacts:

- [data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/truth_campaign_summary.json](/home/xenith/demian/data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/truth_campaign_summary.json)

Run shape:

```text
candidates: gen016_candidate012, gen019_candidate008, gen017_candidate000
surgery pause steps: 32, 64, 96
windows: full, early, mid, late
surgery splits: calibration,test
```

Result:

```text
gen016_candidate012: message/carrier dominated 12/12 pause-window comparisons
gen017_candidate000: message/carrier dominated 11/12 pause-window comparisons
gen019_candidate008: mixed; slow/control dominated pause 64 full/mid/late
```

Interpretation:

- The first one-point surgery was not a strong enough test.
- The calibration strict-profile candidate, `gen016_candidate012`, is exactly
  the candidate that should have been tested first; it strongly favors
  message/carrier under the improved surgery.
- This revives the structured channel-anatomy hypothesis, but it does not
  restore the full mechanism claim because the sample is still only three
  candidates.

Next check:

- Run the same multi-timepoint surgery across the 30-candidate set and split
  results by strict-profile pass/fail status.

## 2026-05-16 - 27 Additional Track B Evolution Replications

Purpose:

- Generate the missing 27 independent Track B candidates needed for the intended
  30-candidate truth campaign.
- Preserve the sharper hypothesis: strict-profile passers should show stronger
  message/carrier dominance under multi-timepoint surgery than failures.

Config:

```text
replications: 27, numbered 4-30
population: 16
generations: 20
seeds: 2026051304-2026051330
eval seeds: 94,95
steps: 128
perturb step: 64
perturb scales: 0.35,0.7
perturb channels: fast
rank mode: native_emergence
reproduction mode: native
default seed: disabled
elitism: disabled
random injection overlay: disabled
device: cpu
parallelism: 3 concurrent replications, 4 workers each
```

Artifacts:

- `/media/xenith/Games/demian_track_b_27_20260516_parallel/`
- `/media/xenith/Games/demian_track_b_27_20260516_parallel/manifest_summary.json`
- `/media/xenith/Games/demian_track_b_27_20260516_parallel/manifest_summary.csv`
- [data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260516_30/replication_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260516_30/replication_summary.csv)

Validation:

```text
exit files: 27
nonzero exit files: 0
archive files: 27
generation files: 27
missing candidate paths in combined summary: 0
artifact size: about 3.1G
```

Result:

```text
new generated candidates: 27/27
combined candidate summary rows: 30
original held-out-confirmed rows: 3
new training-selected archive-best rows: 27
```

Interpretation:

- The generation phase succeeded and removes the prior "only 3 candidates"
  blocker.
- The new 27 are not yet held-out-confirmed positives; they are archive-best
  candidates selected by saved training `rank_score`.
- Technical correction: the first manifest reported 27/27 message/carrier
  top-two training channels, but that was a zero-tie/order artifact. The saved
  per-channel causal-divergence values in the selected archive rows are zero,
  so channel dominance for the new 27 is unresolved until held-out diagnostics
  and state surgery.
- The next truth campaign can now test whether strict-profile passers really
  carry stronger message/carrier anatomy than failures and controls.

Technical details:

```text
selection rule: max saved rank_score row from each archive.json
rank_score min/mean/max: 3.1112 / 3.2907 / 3.6556
release_duty_cycle min/mean/max: 0.0625 / 0.2249 / 0.6875
internal_richness min/mean/max: 0.9091 / 0.9851 / 1.0000
channel_separation min/mean/max: 0.6140 / 0.6734 / 0.6942
release_geometric_event min/mean/max: 0.0083 / 0.4602 / 1.8968
phase_transition_score min/mean/max: 0.0209 / 0.3771 / 1.0403
archive length min/mean/max: 22 / 27.67 / 37
generation log lines per replication: 20 / 20 / 20
```

Next check:

- Run the full truth campaign on
  `data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json`
  with locked thresholds, large controls, recurrence baselines, probes, and
  multi-timepoint surgery.

## 2026-05-16 - Full 30-Candidate Truth Campaign, Gen016-Locked

Purpose:

- Run the full 30-candidate truth campaign while preserving the original
  gen016 strict-profile lock.
- Test whether the gen016 strict profile generalizes to the 29 held-out Track B
  candidates and remains rare against random/default controls.

Guard added before run:

- `development/gate_state_truth_campaign.py` now raises if no calibration Track B
  candidate passes the broad mechanism gate before threshold locking.
- This prevents silent infinite/garbage thresholds.
- `tests/test_gate_state_truth_campaign.py` covers the guard.

Verification before full run:

```text
pytest tests/test_gate_state_truth_campaign.py -q: 6 passed
ruff check development/gate_state_truth_campaign.py tests/test_gate_state_truth_campaign.py: pass
gen016-only preflight: calibration broad gate passed
```

Locked thresholds:

```text
min_route_divergence: 0.2417293204225006
min_gain_zero_divergence: 0.2417293204225006
min_message_carrier_dominance: 0.44469043517608386
```

Config:

```text
replication summary: data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json
output: /media/xenith/Games/gate_state_truth_campaign_20260516_30_gen016locked
track_b_count: 30
calibration_track_b_count: 1
random controls: 1000
calibration random controls: 300
default/default-jitter controls: 100
calibration default/default-jitter controls: 30
seeds: 96,97,98
perturb scale: 0.35
motifs: basis:0, gaussian:0
steps: 128
perturb step: 64
capsule pause step: 64
surgery pause steps: 32,64,96
surgery splits: calibration,test
baselines: rnn, gru, lstm, diag_ssm, demian_native_v8, demian_native_v9
```

Artifacts:

- `/media/xenith/Games/gate_state_truth_campaign_20260516_30_gen016locked/truth_campaign_summary.json`
- `/media/xenith/Games/gate_state_truth_campaign_20260516_30_gen016locked/held_out_test_summary.csv`
- `/media/xenith/Games/gate_state_truth_campaign_20260516_30_gen016locked/state_surgery_summary.json`

Result:

```text
decision: demote_to_ordinary_or_unselected_recurrence
held-out Track B strict pass rate: 0.0
held-out control strict pass rate: 0.025974025974025976
max plain baseline recovery gap: 0.0039061478739458254

held-out Track B:
  mechanism gate: 13/29 = 0.4482758620689655
  strict profile: 0/29 = 0.0
  mean message_carrier_dominance: 0.17313303172163025

held-out random controls:
  mechanism gate: 440/700 = 0.6285714285714286
  strict profile: 20/700 = 0.02857142857142857
  mean message_carrier_dominance: 0.24731316107216858

held-out default-jitter controls:
  mechanism gate: 59/70 = 0.8428571428571429
  strict profile: 0/70 = 0.0
  mean message_carrier_dominance: 0.3695976521845639
```

Interpretation:

- The gen016 strict profile did not generalize to the 29 held-out Track B
  candidates.
- Random controls had a nonzero strict pass rate while held-out Track B had
  zero, so the strong Gate-State Causal Propagation claim remains demoted.
- Track B did retain some broad-gate signal, 13/29, but the broad gate is less
  selective than controls under this run.
- Next analysis should inspect the multi-timepoint surgery artifacts by candidate
  and by broad-gate pass/fail status to see whether a weaker channel-anatomy
  subset exists despite strict-profile failure.

## 2026-10-05 - Six-Channel Phase-One Characterization

Command:

```text
python -m development.run_demian_v1_characterization --seeds 94,95,96 --hidden-size 16 --steps 512 --out data/diagnostics/demian_v1_characterization_20261005/summary.json
```

Scope:

- unselected `DemianV1GateState` parameters
- active, gate-disabled, and gate-frozen update modes
- uninterrupted and surface-perturbed histories
- five-channel v9 historical control
- AFP-v2 tail and threshold robustness slices

Observation:

- six-channel runs: 18
- `surface_nonconvergent`: 12
- `surface_small_step_drift`: 6
- operational surface convergence: 0/18
- independent full-state regimes: 10 transient, 6 structured-persistent, 2
  trivial-accumulation
- gate regimes: 12 fixed, 6 transient
- primary classifications changed under at least one robustness slice: 5/18
- fixed-body surface continuation gaps were nonzero in all 18 six-channel
  history/mode conditions

Interpretation:

- This campaign does not test task utility, trained parameters, uncertainty, or
  selected/evolved regimes.
- No AFP-v2 conclusion is available because no primary six-channel trajectory
  reached the surface-convergence gate.
- Independent internal labels show dynamics that the joint AFP gate would
  otherwise mask; these labels do not imply task utility.
- Continuation divergence establishes exposed-state insufficiency only.

Artifact:

- `data/diagnostics/demian_v1_characterization_20261005/summary.json`

## 2026-10-11 - Fixed-Forcing Route Interventions

Purpose:

- Test whether `carrier_to_slow` is dynamically consequential and whether its
  graded strength changes the response to a carrier pulse.

Artifacts:

- `data/diagnostics/demian_route_stability_v1_20261011/summary.json`
- `data/diagnostics/demian_route_interaction_v1_20261011/summary.json`
- `docs/DEMIAN_ROUTE_STABILITY_RESULTS.md`
- `docs/DEMIAN_ROUTE_INTERACTION_RESULTS.md`

Result:

- Gain-zero route lesions strongly alter the trajectory alone; this does not
  isolate pulse mediation.
- Reduced route multipliers produce nonzero carrier-pulse
  difference-of-differences norms in development and held-out seeds.

Interpretation:

- Preliminary fixed-forcing route-sensitive propagation evidence only.

Next check:

- Matched random-direction gain sweep for direction specificity.

## 2026-10-12 - Matched Route Selectivity

Purpose:

- Test whether `carrier_to_slow` distinguishes carrier pulses from matched
  random-direction pulses across the same route-multiplier sweep.

Observation:

- Carrier-minus-random interaction is negative for every reduced multiplier in
  development seeds and positive for every reduced multiplier in held-out seeds.

Interpretation:

- The fixed-forcing response is seed-sensitive and direction-dependent, but
  not reproducibly carrier-selective.

Next:

- Characterize seed-level route/state features and preregister a confirmation
  or revised selectivity criterion before temporal mediation.

## 2026-10-12 - Route Feature Audit

Observation:

- Pre-pulse features were recorded across 8 seeds, 3 cue histories, 3
  directions, and within-carrier/full-state random controls.
- Features vary across seeds; late cue histories weakly alter them.

Interpretation:

- Discovery only. Pooled feature associations do not distinguish parameter
  identity from state-conditioned route response.

Next:

- Freeze a compact rule after a richer-history discovery design; confirm on
  fresh parameter seeds.

## 2026-10-12 - Uncertainty Discovery Harness

Observation:

- 256 scripted public-evidence traces across 8 fixed model seeds, 4 episode
  seeds, 2 delays, and 4 evidence conditions were finite and replay-exact.

Interpretation:

- Instrumentation only. No internal uncertainty representation, routing, or
  functional-use conclusion.

## 2026-10-12 - Held-Out Uncertainty Representation Probe

Observation:

- Per-seed full-state ridge probes produced positive held-out log-odds scores.
- Astra review found that schedule index was paired with target parity.

Interpretation:

- Unevaluable for retained evidence: schedule-associated evaluator labels can
  explain the apparent decoding.

Next:

- Run the balanced corrective corpus: each reliability permutation receives all
  64 symbol histories, independent balanced targets, schedule-only controls,
  and stratified complete-episode splits.

## 2026-10-13 - Low-Capacity Uncertainty Transfer

Purpose:

- Test frozen low-capacity transfer to a new six-event reliability order and
  to an eight-event history family.

Artifacts:

- `docs/DEMIAN_UNCERTAINTY_TRANSFER_PROTOCOL.json`
- `data/diagnostics/demian_uncertainty_transfer_v1_20261013/summary.json`

Observation:

- Six-event order transfer is positive but not seed-robust under the
  preregistered criterion.
- Frozen six-to-eight transfer fails; eight-event state scale lies far outside
  six-event training normalization.

Interpretation:

- This is a calibrated-transfer failure, not evidence that all target-related
  rank information vanished.

Next:

- Test two appended reliability-0.5 cues, which leave public log odds unchanged,
  with predeclared length-aware controls before causal channel interventions.

## 2026-10-13 - Paired calibration and geometry gates

- Observation: extra delay alone fails frozen calibration; global L2 attenuates
  but does not repair it. Between-history distances remain highly correlated.
- Interpretation: shared translation/rotation remains possible; raw cosine and
  distance preservation do not establish a stable evidence axis.
- Record: `docs/DEMIAN_UNCERTAINTY_GEOMETRY_RESULTS.md`; formal registry entries
  pending. The prior hypothesis status is unchanged.
- Next: fixed channelwise radial normalization on fresh model seeds and a new
  heldout reliability order, with matched transformed shuffle control.

## 2026-10-13 - Fresh channelwise radial confirmation

- Artifact: `data/diagnostics/demian_uncertainty_channel_invariance_v1_20261013/`.
- Observation: B median R2 -36.56, C -33.77; matched-control wins 5/72 and
  14/288 dependent settings. Every model seed fails at least one sensitivity.
- Boundary: A is also imperfect (51/72 wins). Positive B/C correlations do not
  imply calibrated frozen transfer. Global/channelwise radial search ends.
- Review: sent to Astra; formal registration pending review. No causal work.
- Record correction: actual width 1,312 and corrective train split 176;
  executed scores unchanged.
- Review complete: Astra confirms evaluability and per-seed gate failure;
  registered `R-UNC-CHANNEL-001` without promoting `H-UNC-001`.

## 2026-10-13 - Heldout similarity geometry

- Artifact: `data/diagnostics/demian_uncertainty_similarity_v1_20261013/`.
- Observation: full similarity residual 0.111–0.271 relative to A-cloud RMS;
  centered residual 0.038–0.064; heldout excluded variance <0.006.
- Boundary: strict full-map criterion fails. Remaining error is mostly a shared
  centroid offset; this does not establish history-dependent deformation.
- Record: `docs/DEMIAN_UNCERTAINTY_SIMILARITY_RESULTS.md`.
- Review/registry pending. Exact states/transforms and hashes preserved.
- Review complete: Astra confirms evaluability; registered
  `R-UNC-SIMILARITY-001` as geometry diagnostics, with no hypothesis promotion.

## 2026-10-13 - Two-axis Jacobian baseline

- Artifact: `data/diagnostics/demian_uncertainty_jacobian_v1_20261013/`.
- Observation: fixed full-state Jacobian meets the two-step <=0.1 criterion in
  all seed/axis cells. Similarity fails strict unseen-order placement.
- Interpretation: generic local recurrent flow suffices for this finite window.
  Cloud variance coverage does not imply cross-order centroid coverage.
- Astra review complete; registration pending; no hypothesis promotion.
- Record: `docs/DEMIAN_UNCERTAINTY_JACOBIAN_RESULTS.md`.
- Continue: cached-case horizons 2/4/8/16/32 with one fixed source centroid.

## 2026-10-13 - Local horizon and curvature scope

- Horizon artifact: `data/diagnostics/demian_uncertainty_horizon_v1_20261013/`.
- Curvature artifact: `data/diagnostics/demian_uncertainty_curvature_v1_20261013/`.
- Observation: one fixed-centroid linear model meets all-seed threshold only
  at 2 steps; quadratic correction helps most cases but fails all-seed criterion.
- Interpretation: partial ordinary-curvature account, with seed/horizon limits.
- Astra reviews complete; registry entries pending. No evidence-status promotion.
- Record: `docs/DEMIAN_UNCERTAINTY_LOCAL_MODELS_RESULTS.md`.
- Continue: fresh reachable-anchor radius check; no higher Taylor-order search.

## 2026-10-13 - Reachable-anchor radius and precision

- Artifact: `data/diagnostics/demian_uncertainty_radius_v1_20261013/`.
- Observation:768/768 valid responses satisfy local-linear threshold;
 192 smallest responses are precision-limited, despite representable inputs.
- Interpretation: local smoothness in sampled directions, not stable memory.
- Astra review: evaluable. Registered Jacobian, horizon, curvature and radius
  diagnostics; `H-UNC-001` unchanged.
- Verification: full suite488 passed,4 failed in historical artifact/manifest
  tests: `test_current_artifact_loaders`, `test_manifest_validation_runs`,
  `test_v10_summary_contract_digest`, `test_rebuild_v10_summary_core_fields`.
- Ruling: exhaustive32 pairs per cue flip, both orientations, rather than
  selected backgrounds; additional compute buys complete finite-history coverage.
- Continue: fresh natural-history chord gate; all64 histories, two cue positions,
  norm-matched controls and exact native endpoint replay.

## 2026-10-13 - Exhaustive natural-history chords

- Artifact: `data/diagnostics/demian_uncertainty_chord_v1_20261013/`.
- Observation: all13,824 responses precision-valid; all baseline/endpoint replay
  checks exact. Quarter chords pass through h8; some larger chords fail.
- Control: natural-minus-random error contrast reverses in seed1395.
- Interpretation: seed/direction/amplitude-dependent finite curvature; no
  seed-general special memory geometry or causal localization.
- Review pending; record: `DEMIAN_UNCERTAINTY_CHORD_RESULTS.md`.
- Packaging: original393MiB prediction NPZ retained locally;14 row parts
  verified lossless and each below50MB. No measurement or original hash changed.
- Final suite:490 passed; same4 historical artifact/manifest failures listed above.
- Astra review complete: registered `R-UNC-CHORD-001`; narrow evidence status
  unchanged. Natural-only failures concentrate in seed1295.
- Decision: stop local-geometry gates; move upstream to time-resolved calibration
  across cue boundaries/delay. Descriptive channel decomposition is non-causal.

## 2026-10-13 - Time-resolved calibration protocol

- Protocol: `DEMIAN_UNCERTAINTY_LOCALIZATION_PROTOCOL.json`.
- Astra pre-run audit complete: unique prefixes deduplicated; cue2/4 descriptive,
  cue6/delays primary; same prefix reliability histograms across orders.
- Ruling: add analytic single-current/latest-cue controls before full run;
  learned latest trained on a neutral final cue lacks informative-cue support.
- Plan: fresh seeds, time-refitted versus cue6-frozen projections, channel probes,
  conditional-shuffle and public controls. No erasure or causal-channel claim.

## 2026-10-13 - Calibration localization result

- Artifact: `data/diagnostics/demian_uncertainty_localization_v1_20261013/`.
- Observation: cue6 cross-order full-state projections all negative R²;
  cue6-frozen calibration fails after2 blanks even within the original order.
- Interpretation: order and temporal calibration failures are separate;
  neither establishes information erasure.
- Astra review: evaluable; entry pending. Record:
  `DEMIAN_UNCERTAINTY_CALIBRATION_RESULTS.md`.
- Continue: fresh heldout-history × order × time matrix, disjoint32/32 allocation.
  In-sample source reconstruction is not retention evidence.
- Verification:493 passed; same4 historical artifact/manifest failures.
- Registered `R-UNC-LOCALIZATION-001` after Astra review; status unchanged.
- History-matrix pre-run audit passed: split fixed on base-order labels only;
  cross-order heldout target distribution need not match base-order distribution.

## 2026-10-13 - Heldout availability and independent replication

- Artifacts: history-matrix and history-matrix-replication diagnostics directories.
- Observation: both separately allocated batches satisfy all within-order
  matched-time projected-state comparisons through64 blank ticks.
- Controls: disjoint32/32 histories; analytic latest and matched conditional
  shuffle. New seeds, split83 and second schedule pair in replication.
- Portability: cross-order improvements remain partial; cue6-frozen decoding
  fails across delays in both batches.
- Astra: matrix evaluable/supportive, registered `R-UNC-MATRIX-001`;
  replication review pending. No stronger evidence-status promotion.
- Record: `DEMIAN_UNCERTAINTY_HISTORY_MATRIX_RESULTS.md`.
- Verification:495 passed; same4 historical artifact/manifest failures.
- Remaining work is a research-objective fork, not another geometry sweep:
  dissect existing code versus pursue portability under a declared objective.
- Replication Astra review complete: registered `R-UNC-MATRIX-REP-001`.
  Eight parameter seeds across two independent batches; status remains preliminary.
  No routine measurement control remains for the bounded availability claim.
- Independent package code review: no findings;41 focused tests and all new
  artifact pins verified. Reconstructed matrix fits/predictions and publication
  arrays match records. Full-suite historical failures remain explicitly excluded.

## 2026-10-13 - Representation geometry and channel relations

- Artifact: `data/diagnostics/demian_uncertainty_representation_relations_v3_20261013/`.
- Observation: within-order blank delay expands the state cloud while history
  geometry partly deforms; matched-time order clouds remain near-isometric only
  up to translation, scaling and orthogonal alignment.
- Relation audit: gate-to-channel change-norm correlations vary in sign across
  seed, interval and order; no reproducible directional relation.
- Astra review: source manifest, states and metadata hashes must be pinned;
  implemented. No hypothesis-status change or causal interpretation.
- Next: if descriptive work continues, preregister a single identity-versus-
  alignment decoder-transfer test rather than another broad geometry sweep.

## 2026-10-14 - Fresh short-horizon decoder tracking confirmation

- Artifacts: `demian_uncertainty_tracking_confirmation_matrix_v1_20261014` and
  `demian_uncertainty_tracking_confirmation_v2_20261014`.
- Protocol: four fresh parameter seeds, new reliability-order pair and split97;
  cue6/delay2/delay8 only; frozen 16-D base-order tracker and readout.
- Observation: within-order orthogonal tracking clears 12/12 delay2 cells and
  11/12 delay8 cells. Strict cross-order portability clears 0/12 at both delays.
- Interpretation: short within-order correction is present but the declared
  all-cell two-delay confirmation is mixed. No internal adaptive-readout or
  routing conclusion.
- Next: matched-target order design to separate changed evidence targets from
  coordinate deformation before controllability work.
- Astra review: complete. Both artifacts hash-validate; recomputation exactly
  reproduced 504 metrics, prediction rows and 732 saved arrays. Review agrees
  with mixed and no status promotion; adversarial isolation and replay tests
  added afterward.

## 2026-10-14 - Fresh matched-target transport descriptor gate

- Artifacts: `demian_uncertainty_matched_target_descriptor_{matrix,tracking,v1}_20261014`.
- Protocol: new seeds 2494/2495/2594/2595, reliability-preserving cue
  permutation, split131, and base-order-only fits.
- Correction: the first descriptor artifact fit channel maps with heldout-order
  train states, so its association result is unevaluable. Corrected fits use
  base-order train32 only and save fitted objects.
- Observation: strict heldout transport remains 0/12 at delay2 and 2/12 at
  delay8. No descriptor is nomination-eligible because neither delay has a
  sufficient pass/fail seed contrast.
- Interpretation: no residual or channel candidate is nominated. This does
  not imply information erasure or a causal channel conclusion.

## 2026-10-14 - Matched-target cue-order factorial

- Same four fresh seeds and split, two reliability-preserving cue permutations.
- Strict heldout passes: A/B = `3/12`/`8/12` at delay2 and `4/12`/`10/12` at
  delay8; B exceeds A within every seed at both delays.
- Boundary: a condition-dependence observation, not a channel or routing mechanism.

## 2026-10-14 - Cue-pair permutation decomposition

- Discovery: identity plus five adjacent pair-preserving swaps; continuous
  transport margin selected `swap_56 - identity` at delay8.
- Fresh confirmation: positive in 4/4 parameter seeds; mean margin difference
  `+0.2363`.
- Boundary: `swap_56` places reliability `.5` at the final slot. Its large
  margin difference is mainly the analytic latest-cue baseline changing
  (`0.4062 -> -0.0069`), while true R² changes `0.2369 -> 0.2792`.
- Geometry: small centroid displacement and nearly unchanged spread; no
  channel relation was analyzed or intervened on.
- Next: final-slot-matched control before mechanism work.

## 2026-10-17 - Prospective cross-seed anchor transfer

- Protocol frozen at `8ec1ecf`; artifact `demian_cross_seed_anchor_transfer_v1_recovered_20261017`.
- Four fitting / six fresh evaluation seeds; Demian and weights unchanged.
- Observation: Demian has positive R² but0/6 seeds pass at2/8/32/64, specifically
  due permutation-q95+.10. Reservoir passes6/6 through32 and4/6 at64.
- Shared geometry replicates; low-rank affine response explains most early variation.
- JSON export failed after predictions were saved. Report-only correction `5ad71b6`
  exactly reproduces1,084 prediction arrays; original failure preserved.
- Astra review returned: hashes, splits, independent predictions and gates verified;
  no remaining invalidating issue. `R-CROSS-TRANSFER-001`; no status promotion.
- Verification:38 focused pass; full560 pass/4 historical-artifact failures.
- Continue with a frozen feature-conditioning diagnosis before channel hypotheses.
