# GATGRILS v1 Design — Same Behavior, Different Mechanism

**Date:** 2026-09-24

## Purpose

GATGRILS v1 asks a stricter question than v0:

> When two learned recurrent systems produce closely matched observable behavior, can a gauge-invariant local-response grammar distinguish a genuine change of internal computation from ordinary variation between independently trained copies of the same architecture?

V0 established that the current response descriptor and temporal-grammar machinery can survive exact hidden-state gauge changes and can recover strong ordered transition structure, but it did not meet its preregistered predictive-forecast gate. V1 does not retune that failed v0 gate. It changes the experimental question.

The central contrast is:

```text
A1 ↔ gauge(A1)   : coordinate-change floor
A1 ↔ A2          : same-architecture training variation
A1/A2 ↔ B        : mechanism-change contrast
```

The v1 claim is deliberately bounded to this synthetic learned recurrent setting. A positive result would not establish unique computational coordinates, universal mechanism identification, transformer interpretability, or biological equivalence.

## Experimental systems

Use the same symbol-stream family, float64 arithmetic, train/validation/test separation, deterministic seed discipline, and learner-facing objective as v0.

For each of 12 deterministic seeds, train three systems on the same training stream with the same optimizer family, training budget, hidden-state dimension, input alphabet, and readout dimension.

### A1 and A2 — one-stage tanh RNN replicates

A1 and A2 use the v0 recurrent form but independent deterministic initialization seeds:

```text
h_(t+1) = tanh(W_h h_t + W_x u_t + b_h)
y_t     = W_y h_t + b_y
```

A1 and A2 therefore represent ordinary same-architecture learned-instance variation.

### B — two-stage nonlinear recurrent cell

B keeps the same hidden-state dimension and readout dimension but changes the recurrence to an explicit composition of two nonlinear stages:

```text
z_t     = tanh(A h_t + B_u u_t + b_1)
h_(t+1) = tanh(r ⊙ h_t + C z_t + D u_t + b_2)
y_t     = W_y h_t + b_y
```

Use intermediate width 3 when hidden size is 8. The learned elementwise residual coefficient `r` preserves a full-rank direct hidden path so B is not trivially distinguishable only because the composed branch has a low-rank bottleneck.

With the frozen v0 dimensions, the total trainable parameter count is approximately matched rather than exactly identical: A has 174 trainable parameters and B has 187. Record the exact count in every receipt. V1 does not claim parameter-count identity.

No distillation, teacher matching, architecture-specific extra data, or post-hoc pair selection is allowed. If B cannot match A behavior under the frozen common training budget, that is a behavior-match failure rather than a reason to retune training after seeing separation results.

## Observable-behavior matching gate

Mechanism separation is interpretable only when all three learned systems are close on the observable task. Compute the following on the same held-out test stream before inspecting GATGRILS separation metrics.

For each model, report held-out next-symbol cross-entropy. For every pair among A1, A2, and B, report:

- mean Jensen-Shannon divergence between output softmax distributions,
- argmax next-symbol agreement,
- absolute held-out cross-entropy difference.

A seed is **behavior-matched** only if all of these frozen conditions hold:

1. each model's held-out next-symbol cross-entropy is `<= 0.35` nats;
2. the maximum pairwise cross-entropy difference is `<= 0.05` nats;
3. the minimum pairwise argmax agreement is `>= 0.95`;
4. the maximum pairwise mean Jensen-Shannon divergence is `<= 0.02` nats.

The canonical panel requires at least 9 of 12 seeds to be behavior-matched. All 12 seeds remain in the receipt regardless of match status.

The behavior gate is evaluated before the mechanism-separation gate. Unmatched seeds cannot count as mechanism-separation wins.

## Freeze the v0 response descriptor

Do not redesign the descriptor after seeing the v0 result. V1 reuses the exact v0 finite-difference response descriptor:

- normalized power traces,
- dimension-normalized characteristic-polynomial coefficients,
- algebraic-rank diagnostic,
- causal ordered two-step trace ending at the current Jacobian,
- causal ordered three-step trace ending at the current Jacobian.

The algebraic redundancy between power traces and characteristic coefficients noted after v0 is retained intentionally. Descriptor minimality is a deferred experiment so that v1 changes one conceptual variable: the underlying recurrent mechanism.

Use the same central-difference epsilon, float64 precision, discovery/held-out probe separation, and held-out JVP calibration as v0.

## Gauge controls

For A1, A2, and B independently, construct exact gauge twins using

```text
F_G(h', u) = G F(G^-1 h', u)
readout_G(h') = readout(G^-1 h')
```

at condition numbers 1, 3, and 10.

Each model must satisfy the same v0 invariance gates:

- base/twin output maximum absolute discrepancy `<= 1e-10`;
- median invariant-feature relative drift `<= 1e-6` at condition numbers 1 and 3;
- median invariant-feature relative drift `<= 1e-4` at condition number 10.

Any failure of exact base/twin output equivalence invalidates that model/seed for mechanism interpretation, but the failure remains in the receipt.

## Common response-coordinate system for cross-model comparison

Cross-model comparison must not depend on arbitrary cluster labels learned separately in each system.

For each seed:

1. compute v0 response descriptors on the validation stream for A1, A2, and B;
2. concatenate the three validation descriptor sets without supplying model identity labels;
3. fit one shared response-coordinate model using the v0 deterministic clustering and validation-only MDL-like selection rule over the same frozen candidate cluster counts `{2, 3, 4, 5, 6}`;
4. apply that single shared coordinate model to each system's validation and test descriptors;
5. fit a separate event-conditioned transition grammar for A1, A2, and B in this common coordinate space.

The shared codebook is used only for cross-model comparison. The existing v0 separately-fit-and-label-aligned procedure remains the gauge-control implementation.

## Cross-model grammar distance

For two models X and Y in the common coordinate space, define a visitation-weighted transition total-variation distance.

For each source coordinate `c` and input symbol `u`, compute the total-variation distance between the two next-coordinate distributions. Weight that row by the average empirical validation visitation frequency of `(c, u)` in X and Y, then normalize the weights to sum to one.

Call the resulting distance `D(X, Y)`.

For each seed define:

```text
D_replicate  = D(A1, A2)
D_mechanism  = 0.5 * (D(A1, B) + D(A2, B))
```

This avoids privileging one same-architecture replicate as the sole reference.

Also report, but do not gate on, matched-time descriptor distances and post-hoc associations with generator mode labels.

## Primary v1 mechanism-separation gate

A seed counts as a **mechanism-separation win** only if:

- the seed is behavior-matched;
- all three base systems pass exact gauge-twin output equivalence;
- `D_mechanism > D_replicate`.

The canonical v1 mechanism claim passes only if all of the following preregistered conditions hold:

1. at least 9 of 12 seeds are behavior-matched;
2. at least 9 of 12 total seeds are mechanism-separation wins;
3. across behavior-matched seeds, median `D_mechanism` is at least `1.25 ×` median `D_replicate`;
4. across behavior-matched seeds, median `(D_mechanism - D_replicate)` is at least `0.02`.

These conditions prevent a tiny numerical ordering from being reported as substantive mechanism separation.

No threshold may be loosened after the canonical panel is inspected.

## Secondary measurements and controls

Report per seed and aggregate:

- A1/A2/B held-out cross-entropy;
- all pairwise output-distribution Jensen-Shannon divergences;
- all pairwise argmax agreements;
- parameter counts;
- held-out JVP calibration for each model;
- exact gauge-twin output discrepancy for each model and gauge condition;
- invariant descriptor drift for each model and gauge condition;
- shared coordinate count selected by the pooled validation descriptor set;
- `D(A1, A2)`, `D(A1, B)`, `D(A2, B)`, `D_replicate`, and `D_mechanism`;
- per-model ordered transition NLL in the common codebook;
- shuffled-order and bag-of-coordinates NLL controls in the common codebook;
- matched-time descriptor distance between A1/A2 and A/B pairs;
- generator-mode association only as post-hoc diagnostics.

The v0 predictive-forecast NRMSE panel may be rerun for descriptive continuity, but it is not a v1 pass gate and must be labeled secondary. V1 does not reinterpret the v0 8/12 result.

## Outcome interpretation

### Positive mechanism-separation result

If behavior matching, gauge controls, and the mechanism-separation gate all pass, v1 supports the narrow claim that this fixed gauge-invariant response grammar distinguishes the tested two-stage recurrent mechanism from ordinary same-architecture training variation under matched observable behavior.

### Invariant but mechanism-insensitive result

If gauge controls pass but `D_mechanism` does not reliably exceed `D_replicate`, report that the method remains basis-stable but does not distinguish this mechanism change from ordinary training variation.

### Instance-sensitive result

If A1/A2 separation is as large as or larger than A/B separation, report that the recovered grammar is dominated by learned-instance variation at this scale.

### Behavior-match failure

If fewer than 9/12 seeds satisfy the frozen behavior gate, v1 does not answer the intended mechanism question. Report the behavior-match failure rather than weakening the observable-equivalence criteria.

## Scope boundaries

V1 does not claim:

- unique or canonical internal coordinates;
- identification of arbitrary computational mechanisms;
- invariance to nonlinear reparameterizations;
- equivalence of all models that share task accuracy;
- causal sufficiency of the recovered grammar;
- minimality of the response descriptor;
- generalization to transformers or biological systems;
- architecture-independent results beyond the explicitly tested A/B pair.

The exact gauge twins remain a controlled invariance test, not evidence of invariance to arbitrary learned reparameterizations.

## Canonical execution policy

Use 12 deterministic seeds. Preserve all failed and negative seeds. Write a machine-readable v1 receipt containing configuration, parameter counts, behavior-match metrics, all gauge-control metrics, common-codebook metrics, mechanism-separation distances, gate outcomes, and explicit failure reasons.

The first full 12-seed run after implementation and test completion is the canonical panel. Code defects discovered afterward may be fixed only with regression tests and a written explanation distinguishing correctness repair from scientific retuning. Frozen thresholds and architecture definitions remain unchanged.

## Deferred experiments

- Remove Newton-identity redundancy from the descriptor and test whether mechanism separation survives with a minimal spectral basis.
- Compare more than one alternative mechanism class.
- Match parameter counts exactly rather than approximately.
- Match two systems by explicit distillation and ask whether stronger observable equivalence changes mechanism sensitivity.
- Compare independently trained systems that are related by learned, rather than constructed, reparameterizations.
- Apply the mechanism-separation protocol to a small transformer residual-stream system after declaring the relevant symmetry group in advance.
