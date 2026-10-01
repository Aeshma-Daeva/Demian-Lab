# AFP Related-Work Positioning

Status: working literature map for proposal framing. This is not a novelty claim.

## Core caution

The broad statement

> an observed output can remain fixed while internal state continues to evolve

is **not new by itself**.

At least three established research traditions contain closely related ideas.

## 1. Zero dynamics and output-nulling behavior

Nonlinear control theory studies **zero dynamics**: internal system dynamics that
remain when an output is constrained to zero (or to a regulated value).

This is the closest mathematical warning against claiming that hidden motion
under a stationary output is itself novel.

Useful language for Demian:

- external/output convergence does not imply internal-state convergence;
- observability depends on the chosen output map;
- internal dynamics can matter for future stability even when the regulated
  output is unchanged.

Difference from the current Demian question:

- AFP is being detected empirically inside recurrent architectures rather than
  imposed as an output-zeroing control constraint;
- the experiment compares architecture lineages and state exposure;
- continuation/state-surgery tests ask whether the latent difference has future
  consequence.

## 2. Output-null and output-potent subspaces

Systems neuroscience distinguishes activity that projects onto a downstream
readout (**output-potent**) from activity in dimensions that cancel at that
readout (**output-null**).

Strong internal population dynamics can therefore occur without changing the
chosen downstream output.

This is conceptually close to:

```text
G(z_A) ~= G(z_B)
while
z_A != z_B
```

and provides useful vocabulary for explaining why a projection can hide
dynamical structure.

Difference from the Demian experiment:

- Demian exposes an engineered recurrent state whose channels can be directly
  serialized, ablated and swapped;
- the project asks how such separation changes across known architecture
  revisions;
- capsules make continuation from the complete latent state directly testable.

## 3. Continuous-attractor and recurrent-dynamics literature

RNN literature already contains continuous attractors, line/ring attractors,
fixed-point analyses and memory manifolds. Therefore the word "attractor" alone
does not constitute a novel contribution.

The relevant question is narrower:

> can an architecture enter an output/surface-stationary regime while its
> explicitly measured internal recurrent state retains nontrivial trajectory
> motion, history dependence and future consequence?

## Defensible novelty target

The project should avoid claiming discovery of the generic mathematical
possibility of hidden dynamics beneath a fixed output.

A stronger and more defensible contribution would be a combination of:

1. **operational AFP criterion** separating surface velocity from complete
   recurrent-state velocity;
2. **longitudinal architecture study** showing how the regime changes as state
   exposure and routing change from Demian ancestors to v1;
3. **causal continuation controls** using capsules, state surgery and matched
   future perturbations;
4. **cross-domain coupling protocol** applying the same recurrent observer to
   differently ordered dynamical measurements;
5. a reproducible measurement package that distinguishes historical heuristic
   labels from direct state-space measurements.

## Suggested proposal language

Instead of:

> "We discovered that a system can move internally while its output is fixed."

Use:

> "We developed an experimentally testable framework for measuring and
> intervening on latent recurrent dynamics under observable convergence, and
> are studying how this regime changes with architecture, state exposure and
> temporal coupling."

That statement leaves room for a genuine contribution without conflicting with
older control-theory, neural-dynamics or recurrent-attractor work.
