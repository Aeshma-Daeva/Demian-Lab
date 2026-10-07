# Centec Submission Checklist — AFP / Demian

Updated: 2026-10-01

## Track A — Centec Labs / CVT Fortaleza

Purpose: obtain development/prototyping workspace and a local technical
environment for the project.

Current public eligibility explicitly includes young creators, developers,
studios and micro/small businesses with analog or digital projects, including
software, platforms, machines and robots.

Submission package:

- [ ] Project title
- [ ] Short project description
- [ ] Responsible person
- [ ] Development team, if any
- [ ] Current project stage
- [ ] What will be developed/tested in the space
- [ ] Required equipment/infrastructure
- [ ] Expected usage schedule
- [ ] Links to public code/evidence
- [ ] Demian demo / figure

Recommended project name:

**Demian — Plataforma experimental para dinâmica recorrente, estado latente e
acoplamento temporal**

Recommended one-paragraph description:

> Demian is an experimental recurrent-state platform for studying how
> observable trajectories relate to complete internal dynamical state. The
> current project measures surface/latent separation, operational accumulating
> fixed-point regimes, causal state surgery and temporal coupling to external
> systems. The first reproducible result set compares a lineage of recurrent
> architectures and includes held-out native-v9 AFP candidates whose hidden
> state remains causally relevant while the exposed surface is operationally
> quiescent.

## Track B — CEP&T P&D / C&T project submission

The public CEP&T page currently states that interested proponents submit a
project directly for committee review.

The proponent must:

- [ ] indicate a project coordinator; the proponent may be the coordinator;
- [ ] fill the submission form;
- [ ] complete the official work-plan model;
- [ ] electronically sign the work plan via Gov.BR;
- [ ] complete the project presentation letter;
- [ ] electronically sign the presentation letter via Gov.BR;
- [ ] attach the work plan, presentation letter and supporting evidence to the
      submission form.

If approved, the proponent/coordinator is contacted by email and implementation
begins with an advisor indicated to accompany the project.

## Project title for CEP&T

**Dinâmica latente sob convergência observável em sistemas recorrentes:
accumulating fixed points, dependência de trajetória e acoplamento temporal**

Shorter fallback:

**Dinâmica latente e observabilidade em sistemas recorrentes**

## Proposed duration

**6 months**

This is long enough for a bounded research cycle while keeping the first
submission realistic.

## Work packages

### WP1 — Reproducibility and measurement audit — Month 1

Deliverables:

- freeze repository/environment versions;
- consolidate the AFP operational definition;
- archive exploratory and held-out confirmation outputs;
- produce lineage figure and causal-surgery figure;
- document historical-classifier limitations.

### WP2 — Replication across architecture and seed blocks — Months 1–2

Deliverables:

- additional held-out seed block for native v9;
- hidden-size sensitivity;
- selected parameter sensitivity;
- negative controls;
- confidence intervals / descriptive distributions.

### WP3 — Causal state interventions — Months 2–3

Deliverables:

- slow/control ablations;
- stale-hidden-state replacements;
- capsule/full-state and surface-only controls where applicable;
- matched-future perturbation analysis.

### WP4 — Generalized temporal coupling — Months 3–4

Deliverables:

- controlled synthetic dynamical-system adapter;
- one second ordered-data domain using the same recurrent observer protocol;
- comparison of trajectory metrics across adapters.

Initial domain should remain non-clinical. EEG can be introduced using
synthetic/public datasets under clearly non-clinical claims.

### WP5 — External application / demonstration — Months 4–5

Deliverables:

- compact real-time or replay demo;
- visualization of surface versus latent state;
- reproducible example that can be demonstrated at CVT/Centec.

### WP6 — Technical report and dissemination — Month 6

Deliverables:

- final technical report;
- public reproducibility package;
- short scientific manuscript/preprint draft if results justify it;
- presentation/demo for Centec;
- continuation proposal if warranted.

## Central research question

> When an observable recurrent trajectory becomes approximately stationary,
> does the complete recurrent state also converge, or can path-dependent latent
> dynamics persist and remain causally relevant to future continuation?

## Main hypotheses

H1. Some recurrent architectures exhibit measurable separation between exposed
surface motion and complete recurrent-state motion.

H2. A subset can satisfy a preregistered operational AFP criterion: quiescent
surface, active latent state, increasing latent norm and continuing
surface convergence/stability.

H3. Hidden-state interventions at an AFP checkpoint can alter future exposed
trajectories while preserving the current exposed surface exactly.

H4. The incidence and form of this regime depend on state exposure and
architecture.

H5. The measurement framework can be generalized to ordered observations from
external dynamical systems through domain-specific adapters.

## Preliminary evidence to attach

### Held-out AFP confirmation

Fresh seeds 102–129, criterion frozen before inspection:

- native v9: **4/28 clean AFP candidates**;
- three of those four also passed the secondary perturbed arm;
- native v3/v8: one shared clean candidate;
- v2, public v9-five-channel scaffolds and Demian v1: zero under the frozen
  criterion.

Confirmed native-v9 candidate latent/surface motion ratios:

- seed 104: 24.46x
- seed 112: 55.85x
- seed 123: 102.00x
- seed 126: 107.12x

### Candidate-specific causal intervention

At exactly the same current `fast` surface:

- full-state clone mean future gap: **0.000**
- reset slow+control: **~0.102**
- reset slow only: **~0.097**
- reset control only: **~0.0034**
- stale 16–96-step hidden context: reproducible nonzero future divergence

The same shared future impulse produced nearly the same effect.

Interpretation boundary:

- latent state is causally relevant to continuation;
- slow dominates this effect in the tested native-v9 configuration;
- causal latent state is not unique to AFP;
- AFP is the subset where this causal hidden state coexists with a
  preregistered surface-quiescent/latent-accumulating regime.

## Evaluation-criteria mapping

### Relevance and originality

Do not claim that hidden dynamics under a fixed output are generically new.
Position the contribution as:

- an operational measurement framework;
- architecture-lineage comparison;
- state-preserving causal interventions;
- reproducible recurrent-state capsules;
- cross-domain temporal coupling.

### Methodology

Strengths already available:

- explicit falsifiable thresholds;
- discovery/held-out split;
- deterministic seeds;
- negative results retained;
- exact-surface causal controls;
- reproducible CI artifacts;
- historical classifier audit.

### Viability

First phase is CPU/GPU software research and already has working code/results.
No expensive laboratory infrastructure is required.

Available development hardware is sufficient for the proposed recurrent
experiments; Centec infrastructure is primarily useful for institutional
development, demonstration and collaboration.

### Potential impact

Scientific/technical:

- tools for analyzing hidden recurrent state instead of relying only on model
  outputs;
- reproducible methods for state continuity and intervention;
- possible applications in temporal sensing, robotics, signal analysis and
  dynamical-system observers.

### Team qualification

Use evidence rather than credentials-only language:

- existing public repositories;
- implemented recurrent runtime;
- deterministic checkpoint/restore;
- ablations;
- multi-domain adapters;
- documented experiment lineage;
- held-out and causal results.

## Information still requiring the proponent

Do not place these in a public repository unless desired:

- legal full name;
- CPF;
- personal email/phone;
- Gov.BR signature;
- address, if requested;
- bank/payment data, if a later bolsa form requires it.

These should be inserted only into the official Centec submission documents and
forms.
