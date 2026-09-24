# GATGRILS v1 Design — Learned Temporal Control Surfaces

**Date:** 2026-09-24

## Purpose

GATGRILS v0 is frozen as a method result: exact gauge invariance and ordered temporal structure passed, while the preregistered predictive-forecast gate failed. v1 does not retune that result.

The old v1 proposal, **Same Behavior, Different Mechanism**, is deferred to v2. It asks whether GATGRILS can distinguish two hand-chosen recurrent architectures. That is still useful, but the newer question comes first:

> **When a learned recurrent system is given three physically different places where control can act — on the local operator, on temporal admission, and on publication — does task learning use them as functionally separable control surfaces without being told which cue belongs to which surface?**

This design is inspired by the computational distinction exposed in `KolmeOvea`:

- compartment/state-dependent modulation can change the effective local transformation;
- rhythmic perisomatic control can make identical input effective at one phase and ineffective at another;
- a post-integration publication gate can suppress output while leaving the internal computation intact.

Those are architectural inspirations, not claims that the artificial mechanisms are biological apical tufts, basket cells, or chandelier cells.

The v1 claim is deliberately narrower than “recover a grammar of computation.” First establish that a learned system develops separable temporal control surfaces. Only then ask whether GATGRILS can recover their grammar from local responses.

## Why the old v1 is deferred

Three candidate directions were considered.

### A. Learned three-surface system — selected

Train one recurrent system with three distinct intervention points but a shared control state and only task loss. Then use matched counterfactual transplants to ask what each surface actually controls.

This directly tests the question created by `KolmeOvea` without baking in three supervised labels.

### B. Supervise the three gates directly — rejected

Give the model explicit losses saying “this gate is operator,” “this gate is phase,” and “this gate is route.” This would demonstrate that the architecture can implement the desired decomposition, but `KolmeOvea` already establishes the known-answer version. It would not test learned functional separation.

### C. A1/A2/B mechanism separation — deferred to v2

Keep the existing `sol-gatgrils-v1-design` experiment comparing one-stage and two-stage recurrent cells under matched behavior. This remains a clean test of the GATGRILS instrument, but it presupposes the mechanism classes instead of asking whether a useful temporal decomposition appears inside one learned system.

## Experimental task

### Episode factors

Each episode has three independent binary latent factors known to the experimenter but never supplied as separate labeled channels to the control heads:

1. **operator factor** `r ∈ {0,1}` — which computation is applied to the relevant content stream;
2. **phase factor** `p ∈ {0,1}` — which of two temporally interleaved streams is relevant;
3. **route factor** `q ∈ {0,1}` — which output route may publish the result.

The triple `(r,p,q)` is presented at the start of the episode as **one 8-way cue token**. The model is not given three one-hot factor inputs.

### Timing

Use a fixed four-step microcycle. Two content streams use the same scalar content input and the same learned content weights. Their only distinguishing property is arrival phase:

- stream A arrives at microphase 1;
- stream B arrives at microphase 3.

The system also receives a fixed two-dimensional clock basis `(sin φ_t, cos φ_t)` for the admission mechanism. The clock is architecture, not learned evidence about which stream matters.

An episode is:

```text
cue -> silent hold -> five content cycles -> GO -> release
```

The hold duration is sampled from a frozen small set so the cue must persist as resident state rather than being tied to one exact delay.

### Content computation

Each stream supplies five values in `{-1,+1}`. Five values avoid ties.

For the phase-selected stream, the operator factor chooses one of two order-sensitive/insensitive computations:

```text
r = 0: target = sign(x1 + x2 + x3 + x4 + x5)
r = 1: target = sign(x1 - x2 + x3 - x4 + x5)
```

The irrelevant phase stream is independently sampled distractor content.

During cue, hold, and content, both emitted routes should remain near zero. At GO, route `q` should emit the target sign and the other route should remain near zero.

This gives three independent demands in one task:

- the same admitted values can require a different transformation;
- the same synapse/content channel can carry relevant or irrelevant information depending only on arrival phase;
- the completed result can exist internally before it is permitted to leave on one route.

## Learned system

Use one compact recurrent state `h_t`. The cue is written into this ordinary state; there is no supervised “apical state,” “basket state,” or “chandelier state.” All three control heads read the same resident state.

Training uses float64 and deterministic CPU execution. Use PyTorch autograd for training only, then export frozen parameters to NumPy arrays for deterministic analysis and GATGRILS instrumentation. This avoids implementing a second large hand-written BPTT engine while keeping analysis independent of autograd.

### 1. Operator control surface

The recurrent update contains a low-rank, state-dependent correction:

```text
g_op(t)   = tanh(W_op h_t + b_op)
Delta_op  = B_op [ g_op(t) * (A_op h_t) ]
```

This gives learning a place to change the effective local operator without directly adding task content to the output.

### 2. Admission control surface

Only the scalar content channel is multiplied by an admission gate:

```text
a_t = sigmoid(b_a + u_a^T h_t + phi_t^T(v_a + M_a h_t))
x_eff(t) = a_t * x_content(t)
```

where `phi_t = (sin φ_t, cos φ_t)`.

The bilinear term lets resident state move the open phase. Cue and GO channels are not multiplied by this gate.

### Recurrent update

The three ordinary input components are the 8-way cue vector, gated scalar content, and GO flag. The hidden update is explicitly:

```text
h_(t+1) = tanh(
    W_h h_t
    + Delta_op
    + W_cue cue_t
    + w_x x_eff(t)
    + w_go GO_t
    + b_h
)
```

No phase identity, operator bit, or route bit is injected separately into this update. They are available only through the learned effect of the single cue token on resident state.

### 3. Publication control surface

The hidden state produces two ungated latent route values:

```text
v_t = tanh(W_y h_t + b_y)      # two routes
```

and a separate publication head produces

```text
p_t = sigmoid(W_pub h_t + u_go * GO_t + b_pub)
y_t = p_t * v_t
```

The task loss is applied to emitted `y_t`; there is no direct target on `p_t` or `v_t` separately.

This is important: learning is free to keep `p_t` open and suppress `v_t`, ignore the operator correction, or ignore rhythmic admission if those strategies solve the task. Functional use of a surface must be earned by intervention, not assumed from architecture.

## Training objective

Use one common task loss only:

- mean squared error on the two emitted routes;
- zero targets during cue/hold/content;
- selected-route target `±1` at GO/release;
- nonselected route remains zero.

Use fixed loss weights chosen on development-only seeds before the canonical run. No auxiliary disentanglement loss, gate-label loss, factor-classification loss, or intervention loss is allowed.

A small L2 penalty may be used consistently across all trainable parameters. Do not add sparsity or gate-specific penalties after inspecting canonical results.

## Development/canonical split

Use two development seeds outside the canonical panel, e.g. `1000` and `1001`, for implementation debugging and choosing frozen training hyperparameters.

The canonical panel is seeds `0..11` and is not inspected until:

1. unit/integration tests pass;
2. development seeds learn the task without numerical instability;
3. all thresholds below are frozen in configuration.

The first complete 12-seed run after that freeze is canonical. Negative seeds remain in the receipt.

## Primary measurement: matched counterfactual transplants

The key experiment does not inspect which cue bit a control head “represents.” It asks what changing that control surface **causes**.

Construct matched episode pairs that have identical content and identical two factors, but differ in exactly one factor. Run both normally and record the control traces.

Then rerun the recipient episode from its own initial state while overriding exactly one control surface with the donor trace. **Do not transplant or clamp the donor hidden state.** The recipient hidden trajectory is allowed to evolve normally under the altered control, so any downstream change is a causal consequence of the transplanted surface.

### Operator transplant

Pair episodes that differ only in `r`. During the content interval, transplant the donor `g_op(t)` trace into the recipient while recipient admission, publication, inputs, clock, and initial resident state remain its own.

A successful operator transplant makes the recipient answer the donor operator's counterfactual target on the same admitted content.

### Admission transplant

Pair episodes that differ only in `p`. During content, transplant the donor admission trace `a_t` while recipient operator control, publication, inputs, and initial resident state remain its own.

A successful admission transplant makes the recipient compute from the donor-relevant temporal stream, even though both streams still use the same content input weights.

### Publication transplant

Pair episodes that differ only in `q`. At GO/release, transplant the donor publication trace `p_t` while recipient hidden dynamics, latent route values, operator control, admission, and content remain its own.

A successful publication transplant moves the already-computed result to the donor route without requiring the hidden computation to be rerun.

## Specificity matrix

For every factor change, also transplant each of the two nonmatching surfaces. This creates a 3 × 3 causal specificity matrix:

```text
                    counterfactual factor
surface transplanted    operator   phase   route
operator
admission
publication
```

For each cell, report the fraction of matched pairs whose emitted result moves to the target implied by the donor factor.

The architecture names the intervention points, but **no training target says which cue factor should use which point**. The matrix is the test of learned specialization.

## Publication state-preservation control

The publication distinction needs a stronger test than “route output changed,” and it must test **continued updating while silent**, not merely storage of an answer computed before the mute.

Use a diagnostic continuation that begins from a normal held-out episode after the first part of its content sequence. Then:

1. present a GO/permission signal that would normally make the selected route publish;
2. externally clamp only the selected publication gate closed;
3. continue delivering new relevant content for a frozen number of cycles, including at least one update that flips the correct running answer;
4. release the publication clamp **without adding another content update**.

Measure:

- emitted selected-route output during the clamp;
- ungated latent value `v_t` after each hidden content update;
- latent sign accuracy immediately before release against the updated running target;
- first-step emitted accuracy immediately after release.

Run a matched control in which publication is left available but **content admission** is clamped closed for the same continuing-content interval.

The intended distinction is:

```text
publication clamp: output silent, hidden/latent computation tracks new evidence
admission clamp:   new evidence never enters, latent answer misses the update
```

This directly tests “admission gating != emission gating” and mirrors the key `KolmeOvea` E3 distinction without assuming a biological AIS mechanism.

## Rhythmic admission control

For held-out episodes, replace the learned time-varying admission trace within each cycle by its cycle mean while preserving the same mean admission.

This is the learned analogue of replacing rhythmic basket inhibition with tonic inhibition of equal mean.

Report phase-factor counterfactual accuracy under:

- learned rhythmic admission;
- cycle-mean admission;
- half-cycle retiming of both streams.

Retiming is an intervention on arrival time, not on weights or stream identity.

## Frozen v1 pass gates

A seed is **task-valid** if held-out selected-route sign accuracy is at least `0.90` and mean absolute emitted activity on both routes during silent periods is at most `0.10`.

The canonical v1 learned-separation claim passes only if all of the following hold:

1. at least **9/12** seeds are task-valid;
2. on task-valid seeds, median own-surface counterfactual transplant success is at least **0.80** for each of operator, phase, and route;
3. for each factor, median own-surface transplant success exceeds the best nonmatching-surface transplant by at least **0.30**;
4. under publication clamp, median latent selected-route sign accuracy immediately before release is at least **0.85**, and first-step emitted sign accuracy after release is at least **0.80**;
5. the matched admission-clamp control lowers latent sign accuracy by at least **0.20** relative to publication clamp;
6. replacing rhythmic admission by its cycle mean lowers phase-factor counterfactual accuracy by at least **0.15** on the median task-valid seed.

If the model solves the task but the specificity matrix is not diagonal/selective, report **successful task learning without separable control surfaces**. Do not change thresholds or add regularizers after seeing the canonical panel.

## Secondary measurements: the GATGRILS instrument comes back in

The primary v1 result is causal functional separation. Run the following as secondary measurements after the primary gate has been evaluated; they cannot rescue it.

### Silent-ping tomography

At the end of the silent hold, clone the state and apply a fixed neutral content ping at each microphase. Measure the local response (`Delta h`, ungated `Delta v`, and emitted `Delta y` under a diagnostic publication-open condition).

Fit validation-only linear decoders for the three cue factors from the ping response and evaluate on held-out episodes. Repeat after resetting the resident state before the same ping.

This asks whether a silent cue state can be read from **how the system responds to a fixed perturbation**, and whether reset destroys that information.

### Context-invariance / temporal window

Repeat matched central content snippets after different preceding cues and vary the cue-to-snippet lag. Measure how long the preceding context continues to alter the local response.

Report an operational context window for the hidden response and for each control surface. Also report how the window changes when episode timing is uniformly stretched/compressed.

This borrows the black-box logic of temporal-context-invariance measurements: how far back can a context perturbation still change the present response?

### Gauge-aware local-response grammar

Treat the complete continuous recurrent state as the object of the v0 local-Jacobian instrument. Construct exact black-box gauge twins by wrapping the frozen transition:

```text
F_G(s', u) = G F(G^-1 s', u)
```

with the observable readout transformed accordingly.

Reuse the v0 invariant response descriptor and ordered grammar without changing its formulas. Report whether recovered response coordinates/grammar are stable under gauge change and whether they align with the causally identified operator/admission/publication regimes.

This is secondary because v1 first asks whether the regimes exist. The old A1/A2/B “same behavior, different mechanism” design becomes v2 after this.

## Controls and baselines

Report, but do not make primary pass gates:

- the same recurrent hidden size with all three explicit control surfaces removed (plain RNN baseline);
- a collapsed-controller baseline in which one scalar control signal is broadcast to all three intervention points;
- parameter counts and training loss for all systems;
- control-head variance and saturation statistics;
- task accuracy by all eight cue combinations and by each hold duration;
- single-stream episodes in which the distractor phase is empty, to distinguish timing failure from operator failure;
- route-open diagnostic runs to separate latent-value failure from publication failure.

A plain RNN may solve the task. v1 does **not** require the three-surface architecture to outperform every generic recurrent model. The claim is about whether the learned three-surface system uses its available control points in causally separable ways.

## Receipt

Write one machine-readable canonical receipt containing:

- frozen configuration and dependency versions;
- all 12 seeds, including failures;
- task accuracy and silence metrics;
- 3 × 3 transplant specificity matrix per seed and aggregate;
- publication/admission clamp metrics;
- rhythmic-versus-cycle-mean and retiming metrics;
- secondary silent-ping, temporal-window, and gauge metrics when computed;
- explicit gate outcomes and failure reasons;
- a canonical summary block suitable for rendering into the README.

## Claim boundary

A positive v1 would support only this statement:

> In this synthetic recurrent task, a system trained only on behavior made functionally selective use of three architecturally distinct control surfaces: state-dependent operator modulation, phase-dependent input admission, and post-computation publication.

It would not establish that cortex uses these exact algorithms, that apical/basket/chandelier cells map one-to-one onto the three artificial controls, that the decomposition is unique, or that a plain RNN cannot implement the same input/output function.

The biological work motivates **where to look for distinct control surfaces**. The learned intervention experiment determines what this artificial system actually uses.

## Deferred v2

After v1 is frozen:

1. return to the existing **Same Behavior, Different Mechanism** design;
2. compare independently learned systems with matched observable behavior;
3. ask whether the gauge-invariant local-response grammar can distinguish mechanism class from same-architecture training variation;
4. test whether the v1 causal control-surface labels provide a useful external validation target for recovered grammar coordinates.
