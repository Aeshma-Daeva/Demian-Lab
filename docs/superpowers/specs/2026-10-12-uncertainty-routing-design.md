# Uncertainty-Guided Routing — Proposed Research Design

Status: proposed; no new result or capability claim. Implementer: Terra Medium.

## Question

Can internal Demian state preserve evidence supporting unresolved alternatives,
update with later observations, and eventually guide beneficial routing or
information acquisition?

Demian remains the experimental system. State magnitude, persistence, gate
variation, and response sensitivity are not uncertainty measures by themselves.

## Immediate scope

Repair route measurement contracts and build a small ambiguous-evidence world,
evaluator, stateless encoder, replay harness, and preregistered discovery pilot.
Keep the six-channel transition rule and weights unchanged. Do not implement a
controller, train Demian, or start a large campaign in this package.

## Evidence boundary

- Existing route/history runs are discovery data; none are fresh confirmation.
- Existing primary and channel-control pulses both use random carrier directions.
  Their labels do not identify meaningful versus meaningless information.
- Earlier peak contrasts are differences of separate peaks, even with equal
  observation windows. Preserve these artifacts and correct their descriptions.
- Relative pulses change absolute magnitude when history changes carrier norm.
- Historical summary grouping by cue symbol merges distinct pulse histories.
  Use raw cells to recover per-history comparisons.
- Direct channel pulses create intervention-conditioned states. Do not describe
  those states as naturally reachable through the observation interface.
- Feature/history audits require explicit registration or pending-registration
  markers; do not infer reviewed status from prose alone.

## Environment and evaluator

Use a separate world module; preserve the existing cue-delay-query API.

Hidden target `h` is binary with prior P(h=1)=0.5. Public evidence events are
`(symbol, reliability)`, where symbol is 0 or 1 and reliability is one of
0.5, 0.75, 0.95. For generative trials P(symbol=h)=reliability, independently
conditional on h. Reliability is public; target and evaluator beliefs are private.

The evaluator computes the posterior from public evidence only:

`log_odds += (2*symbol-1) * log(reliability/(1-reliability))`.

Record posterior probability, binary entropy, and log odds outside the agent.
These are reference quantities under the declared task model, not measurements
of Demian's subjective uncertainty.

Discovery conditions: balanced/conflicting evidence; consistent informative
evidence; reliability-0.5 noise; order permutations with identical posterior;
then delay and a truthful resolving token. Resolving evidence is visible only at
its scheduled tick. Missing observations are distinct from reliability-0.5 cues.
Do not expose condition labels, target, posterior, entropy, score, or future cues
through observations, acknowledgements, filenames consumed by the encoder, or
action feedback before the episode ends.

Scripted evidence sequences may be used for deterministic controls. Label them
as scripted and keep them separate from frequency/calibration claims based on
generative trials. Identical public prefixes must yield identical encoded inputs
and states for fixed model initialization, regardless of hidden target.

## Internal state and measurement

Encoder and action decoder are stateless. No external storage register or
reference bank may carry evidence for Demian. World RNG and queues are external
environment state and must be checkpointed, but never fed back as memory.

Save features before each current observation and after its transition, with
explicit timestamps. Record full-state/channel norms, past-only increments,
gate summaries, and selected route contributions. Store full states at evidence,
pre-resolution, and post-resolution checkpoints; aggregate remaining traces.

External probes may predict evidence statistics or evaluator posterior from
internal state. They are fitted measurement instruments, never inputs or routing
policies. Report their capacity and training split. Compare against time-only,
current-observation-only, and simple evidence-count/reference baselines. A probe
can recover retained evidence without establishing an internal uncertainty code.

## Sequential research gates

1. Harness validity: no leakage, exact continuation, fixed-input replay,
   consistent pulse loading, distinct histories, explicit observation windows.
2. Representation: frozen probes generalize across episodes and report results
   per model seed; distinguish retained evidence from elapsed time, magnitude,
   and generic growth. Generalization across model seeds requires an explicit
   coordinate-invariant feature strategy or declared alignment procedure.
3. Causal contribution: matched within-model channel resets, reciprocal swaps,
   and restore controls change the relevant readout. Match residual state
   distances where practicable; mark transplanted states as potentially off
   distribution. Establish functional use separately from probe decodability.
4. Regulation: a separately declared frozen controller uses only accessible
   state. Compare fixed, random, and yoked schedules under matched sensing,
   parameters, persistent state, and computational/action budgets.
5. Utility: improve delayed decisions, calibrated predictions, or information
   acquisition under a fixed evaluator. Larger perturbation response is not utility.
6. Adaptation: introduce a declared update mechanism and changed cue reliability;
   test improvement on fresh sequences with fixed external evaluation criteria.

## Later query task

Reserve `request_evidence` for a subsequent controller experiment. Use a deadline
with no guaranteed free resolving cue before it; otherwise a costly early query
has no decision benefit unless latency is explicitly scored. Charge each accepted
query, specify response reliability/latency, and evaluate answer reward minus
query cost. Compare oracle query/no-query policies to validate the task tradeoff.
Do not call the original untrained decoder's arbitrary actions competence.

## Success and falsification

Initial success is a validated harness and an auditable discovery record.
Evidence for usable internal uncertainty additionally requires predictive
representation, appropriate revision, causal influence, and functional benefit.
Null findings, leakage, scale-only explanations, failed transfer, and added
controller-capacity explanations remain explicit outcomes.
