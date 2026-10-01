# Native-v9 AFP Causal Follow-up Preregistration — 2026-10-01

This intervention plan is recorded before viewing candidate-specific causal
surgery outputs.

## Source candidates

Held-out native-v9 seeds that passed the frozen AFP rule:

```text
104, 112, 123, 126
```

The follow-up will also run all other held-out native-v9 seeds 102–129 as
descriptive controls. Candidate status will not be redefined.

## Checkpoint

Each model is run autonomously for 384 steps using the same construction as the
lineage confirmation.

At step 384, native v9 has:

```text
state = (fast, slow, control)
surface = fast
```

This architecture is especially useful for causal surgery because `slow` and
`control` can be changed while keeping the exposed surface **exactly
identical**.

## Primary causal question

For an operational AFP candidate:

> If the current exposed `fast` surface is kept exactly fixed, does changing
> only the hidden recurrent context change the future exposed trajectory?

A positive result establishes causal relevance of latent state at that
surface-quiescent checkpoint. It does not establish cognition or semantic
memory.

## Interventions

All arms start with the same step-384 `fast` tensor.

1. `full_clone`
   - exact copy of fast/slow/control;
   - deterministic sanity control;
   - expected future gap: numerical zero.

2. `surface_only`
   - preserve fast;
   - zero slow and control.

3. `slow_reset`
   - preserve fast and control;
   - zero slow.

4. `control_reset`
   - preserve fast and slow;
   - zero control.

5. `stale_16`
   - preserve current fast;
   - replace slow/control with their own values from 16 steps earlier.

6. `stale_32`
   - same, using hidden state from 32 steps earlier.

7. `stale_64`
   - same, using hidden state from 64 steps earlier.

8. `stale_96`
   - same, using hidden state from 96 steps earlier.

The stale-state arms are the most directly related to accumulation: they alter
the hidden history while leaving the current exposed surface untouched.

## Future conditions

Each intervention is tested under two identical-future conditions:

### autonomous

Continue both reference and intervention arms for 64 steps with no external
coupling.

### shared impulse

Apply the same deterministic bounded vector to the fast surface of both arms
immediately before continuation, then run the same 64 autonomous steps.

Because the intervention arms share the same fast state before the future
condition, and the impulse is identical, any deterministic post-step divergence
must originate from the altered latent state.

## Measurements

For every arm:

- initial surface RMS gap;
- first-step surface RMS gap;
- mean 64-step surface RMS gap;
- maximum 64-step surface RMS gap;
- final surface RMS gap;
- area/sum of the surface-gap trajectory;
- initial latent RMS distance from the reference.

The initial surface gap must be exactly zero within float tolerance before the
continuation is accepted.

## Interpretation

The primary result is continuous effect size, not a tuned pass/fail threshold.

Required sanity check:

```text
full_clone future gap ~= 0
```

Candidate-specific causal evidence is present when a latent intervention with
zero initial surface gap produces a reproducible future surface gap above the
full-clone numerical floor.

The same analysis on noncandidate held-out seeds is reported as a control and
must not be hidden if latent surgery affects them too.

## Stronger conclusion boundary

This experiment can support:

> hidden state is causally relevant to future continuation at an operational
> AFP checkpoint.

It cannot by itself support:

- semantic memory;
- consciousness;
- selfhood;
- general intelligence;
- clinical or biological interpretation.
