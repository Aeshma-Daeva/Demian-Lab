# Demian v1 Route Dynamics Program

Status: untrained recurrent substrate.

Progression:

`route consequence → selective propagation → temporal mediation → endogenous regulation → competitive organization → functional adaptation`

## E1 — Route Selectivity

- Hypothesis: `carrier_to_slow` changes carrier perturbations differently from
  amplitude-matched random-direction perturbations.
- Method: repeat the existing multiplier sweep `1.0, .9, .75, .5, .25, 0.0`
  with matched random-direction pulses; preserve states, forcing, timings,
  amplitudes, and seeds.
- Outcome: compare normalized pulse × gain interaction curves by channel/time.
- Alternative: curves are equivalent; current evidence is generic gain
  sensitivity.
- Registry link: `H-ROUTE-005`; completed with mixed evidence. The interaction
  difference reverses sign across development and held-out seeds, so E2 is
  gated on seed-level characterization and a preregistered confirmation or
  revised criterion.

## E2 — Temporal Causal Mediation

- Hypothesis: `carrier_to_slow` produces an early, localized transfer into
  `slow` before global trajectory divergence.
- Method: preregister transition-local windows; compare intact, weakened,
  disabled, sham, and alternative-channel controls.
- Alternative: effects are later amplification or distributed sensitivity.

## E3 — Endogenous Gate Regulation

- Hypothesis: dynamic gate state changes propagation beyond gain-matched static
  controls.
- Method: compare dynamic, frozen, static gain-matched, and yoked/shuffled gate
  conditions with matched weights, inputs, and initial state.
- Alternative: static gain accounts for the measured effects.

## E4 — Competitive Routing

- First architecture extension. Introduce a minimal finite routing budget and
  compare it with matched baseline capacity.
- Measure allocation diversity, interference, boundedness, and perturbation
  recovery.
- Alternative: competition collapses, saturates, or adds no regulation benefit.

## E5 — Functional Adaptation

- Introduce controlled cue/delay/query tasks, then closed-loop worlds, with
  measurable objectives and held-out sequences.
- Compare fixed, dynamic, and budget-constrained routing on retention,
  recovery, performance, and resource use.
- Learning requires a separately declared and independently tested learning
  mechanism; recurrent state dependence alone is not adaptation.

## Discipline

- Keep development and held-out seeds separate.
- Predefine windows and success criteria.
- Separate early causal effects from long-horizon sensitivity.
- Do not equate divergence with information, memory, adaptation, or utility.
