# GATGRILS

## GATGRILS v0 — frozen result

**Gauge-Aware Temporal Grammar Recovery in a Learned System** asks a deliberately small question: can local response behavior in a learned recurrent system be compressed into reusable computation coordinates whose **temporal grammar predicts future local responses**, and does that description survive an exact change of hidden-state basis?

This v0 is a deterministic, CPU-only **synthetic recurrent model** experiment. It is a method test. It does not claim a universal algorithm, biological equivalence, transformer interpretability, unique/minimal coordinates, or superiority to recurrent networks as predictors.

## Run it

Requires Python 3.11+ and NumPy. Pytest is only a development dependency.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m gatgrils.experiment --config default --output results/gatgrils_v0.json
```

The default command trains and evaluates all 12 preregistered deterministic seeds and writes the complete per-seed and aggregate receipt to `results/gatgrils_v0.json`. Negative runs and failing gates are part of the result and are not discarded.

## Experimental object

A persistent latent mode selects one of several fixed symbol-transition maps. The learner sees only the current symbol and a next-symbol objective. **Mode labels are diagnostic metadata only**: they never enter RNN training, response-feature construction, cluster-count selection, grammar fitting, or gate calculation.

A small tanh RNN is trained with explicit truncated backpropagation through time in float64 and then frozen. For a hidden state `h` and one-hot input `u`, its recurrent update is `F(h, u)`. At held-out validation and test states, GATGRILS estimates the local Jacobian

```text
J_t = dF(h_t, u_t) / dh
```

with central finite differences. A discovery probe bank spans hidden space; a separate **held-out probe** bank checks unseen Jacobian-vector products. That JVP score is calibration only and is not used as the cross-gauge predictive score.

## Exact gauge twins

For an invertible, bounded-condition matrix `G`, the conjugate system is

```text
F_G(h', u) = G F(G^-1 h', u)
readout_G(h') = readout(G^-1 h')
```

with `h' = G h`. The base model and gauge twin therefore implement the same input/output computation, while their raw hidden coordinates and raw Jacobian entries generally differ. v0 tests condition numbers 1, 3, and 10 and rejects draws outside their assigned condition bound.

## Response descriptor

The discovered descriptor never uses the generator's hidden mode. For each finite-difference Jacobian it contains deterministic similarity-invariant quantities:

- normalized power traces `tr(J^k) / d` for short powers,
- dimension-normalized characteristic-polynomial coefficients,
- an algebraic-rank diagnostic,
- causal ordered short-window traces ending at the current step: `tr(J_(t-1) J_t)/d` and `tr(J_(t-2) J_(t-1) J_t)/d`.

The last two terms deliberately retain operator order that per-step spectra alone cannot. A separate per-step-only control removes those ordered terms.

Candidate coordinate counts are clustered with deterministic NumPy k-means and selected using one fixed validation-only MDL-like score. The fitted temporal grammar is a first-order event-conditioned distribution

```text
p(c_(t+1) | c_t, u_t)
```

where `c_t` is the recovered response coordinate. Gauge twins are aligned only by coordinate-label assignment at matching validation times; hidden mode labels are not used for alignment.

## Controls

v0 reports all preregistered controls:

- **activation clustering** — the same coordinate-selection machinery on raw hidden activations;
- **per-step invariant only** — removes ordered two- and three-step trace features;
- **shuffled response sequence** — preserves response-feature distribution while destroying temporal order;
- **bag of coordinates** — preserves coordinate counts while discarding transition order;
- **held-out probe bank** — checks finite-difference Jacobian generalization to unseen directions;
- **gauge twins** — exact same learned computation in another hidden basis.

The primary held-out predictive score is the NRMSE of the **next gauge-invariant response descriptor**, predicted from the current coordinate and current input. The JVP calibration is reported separately so coordinate-invariance is not confused with a coordinate-dependent probe-response target.

## Frozen v0 gates

The pass thresholds come from the approved design spec and are not changed after observing results:

1. Base/twin output maximum absolute discrepancy **≤ 1e-10** in float64.
2. Median invariant-feature relative drift **≤ 1e-6** at condition numbers 1 and 3, and **≤ 1e-4** at condition number 10.
3. After coordinate-label alignment, gauge-twin transition-distribution total variation **≤ 0.02**.
4. Grammar-conditioned held-out forecast NRMSE must beat both activation clustering and shuffled responses on at least **9/12** seeds, and its median NRMSE must be at least **10%** lower than activation clustering.
5. Ordered grammar held-out transition NLL must beat both shuffled-order and bag-of-coordinates controls on at least **9/12** seeds.

Any failure of exact gauge-twin output equivalence invalidates that seed. Other unmet gates remain ordinary negative results. No metric is dropped because it is inconvenient.

## Claim boundary

A positive v0 would show only that, in this synthetic learned recurrent system under this traffic, a gauge-invariant local-response description can be recovered and can carry predictive temporal structure. It **does not claim** that the recovered coordinates are unique, globally minimal, causally sufficient, robust to arbitrarily ill-conditioned bases, identical to the generator modes, biological computation primitives, or directly transferable to a transformer.

A negative predictive gate is also informative: it would mean that the experiment recovered basis-invariant response coordinates without earning the stronger claim that this particular temporal grammar forecasts better than the preregistered controls.

## Canonical result

The canonical result block below is generated from the committed receipt after the frozen 12-seed run.

<!-- canonical-summary:start -->
### Canonical 12-seed receipt summary

- **Overall:** FAIL
- Exact gauge-twin output equivalence (≤ 1e-10): **PASS**
- Median invariant drift: cond 1 = 8.352e-12, cond 3 = 1.400e-11, cond 10 = 5.098e-11 → **PASS**
- Maximum aligned grammar transition TV = 0 (≤ 0.02) → **PASS**
- Forecast wins vs both activation and shuffled controls = 8/12; median grammar NRMSE = 0.75229, activation = 0.797405 → **FAIL**
- Ordered-transition NLL wins vs both shuffled-order and bag controls = 12/12 → **PASS**
- Median held-out JVP relative error (calibration only) = 4.570e-11

This block is rendered from `results/gatgrils_v0.json`; failing gates are retained rather than retuned.
<!-- canonical-summary:end -->

## Reproducibility notes

- All experiment arithmetic is float64.
- Each stream, model, probe bank, shuffle, and gauge draw uses an explicit deterministic NumPy seed.
- Train, validation, and test symbol streams use disjoint derived seeds.
- The finite-difference step is fixed in configuration.
- Discovery and held-out probe banks are distinct.
- The canonical receipt includes every seed, every control, every gauge condition, gate values, and explicit failure reasons.

See `docs/superpowers/specs/2026-09-23-gatgrils-design.md` for the approved design and `docs/superpowers/plans/2026-09-23-gatgrils-v0.md` for the implementation plan.

## GATGRILS v1 — learned temporal control surfaces

V1 keeps the v0 result frozen and asks a different question: when one learned recurrent system is given three architecturally distinct intervention points—state-dependent operator modulation, phase-dependent input admission, and post-computation publication—does task-only learning use them as causally separable control surfaces?

The cue is a single 8-way token, both temporal streams share one scalar content channel, and no auxiliary gate-label, factor-classification, disentanglement, transplant, or intervention loss is used. The primary measurement is a 3×3 matched counterfactual transplant matrix, with publication-vs-admission clamps and rhythmic-vs-cycle-mean admission controls. Silent-ping, temporal-context, and gauge-aware grammar measurements are secondary and cannot rescue a failed primary gate.

Run the frozen v1 panel with:

```bash
python -m gatgrils.v1_experiment --output results/gatgrils_v1.json
```

The canonical v1 result block will be rendered here from the first complete frozen 12-seed receipt; negative seeds and failed gates are retained.
