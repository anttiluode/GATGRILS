# GATGRILS

Gauge-Aware Temporal Grammar Recovery in a Learned System.

This repository now contains two frozen experiments. **v0** asks whether a local, basis-independent grammar of recurrent computation can be recovered. **v1** asks whether task-only learning turns three architecturally different control surfaces into three causally different jobs: operator modulation, phase-dependent admission, and publication.

## GATGRILS v0 — frozen result

V0 remains unchanged as a historical method result. It trains a small recurrent system, estimates local Jacobians, describes them with similarity-invariant features, and checks the same learned computation after exact invertible changes of hidden-state basis.

Canonical 12-seed result:

- **Overall:** FAIL
- Exact gauge-twin output equivalence: **PASS**
- Maximum aligned grammar transition TV: **0**
- Forecast wins vs both activation and shuffled controls: **8/12**
- Median grammar NRMSE: **0.75229**
- Median activation-clustering NRMSE: **0.797405**
- Ordered-transition NLL control: **12/12 PASS**

The important negative is preserved: v0 recovered basis-invariant temporal structure but did not earn the stronger preregistered predictive claim.

## What “gauge-aware” means here

The hidden coordinates of an RNN are not unique. For any invertible matrix `G`, write

```text
h' = G h
F_G(h', u) = G F(G^-1 h', u)
```

and transform the readout back with `G^-1`. The base network and its gauge twin compute the same input/output function even though the raw hidden activations and Jacobian entries look different.

GATGRILS therefore describes local computation with quantities unchanged by the similarity transform `J' = G J G^-1`: power traces, characteristic-polynomial information, and ordered multi-step traces. “Gauge-aware” means the analysis tries to follow the computation rather than the arbitrary orientation of hidden-state axes.

## GATGRILS v1 — learned temporal control surfaces

V1 uses one trained recurrent system with three genuinely different intervention points:

- **operator surface** — changes the local recurrent transformation;
- **admission surface** — changes whether content arriving at a particular within-cycle phase enters effectively;
- **publication surface** — changes whether the already-computed latent result is emitted.

Training gets only task loss. The cue is one 8-way token; the operator/phase/route factors are experimenter metadata, not three labeled training channels. The primary causal test is a 3×3 matched transplant matrix plus publication-vs-admission clamps and rhythmic-vs-cycle-mean admission controls.

The configuration was frozen on development seeds `1000` and `1001` before canonical evaluation. The canonical seeds are exactly `0..11`. A final correctness repair added masked intervention windows so recipient dynamics recompute normally outside the transplanted interval; the pre-fix receipt was invalidated and the same frozen models/config/seeds were rerun.

### Corrected canonical 12-seed result

- **Overall:** FAIL
- Task-valid seeds: **4/12** (requires 9)
- Own-surface transplant medians — operator **0.328**, admission/phase **0.339**, publication/route **0.516** (requires 0.80 each)
- Specificity margins — operator **0.286**, phase **0.182**, route **0.000** (requires 0.30 each)
- Publication-clamp latent / first-release medians: **0.266 / 0.552** (requires 0.85 / 0.80)
- Admission-clamp damage median: **-0.0625** (requires +0.20)
- Cycle-mean admission damage median: **0.151** (requires 0.15) → **PASS**

So task-only learning did **not** demonstrate the preregistered three-way operator/admission/publication specialization. Only the timing control survived the primary gates: replacing the learned within-cycle admission profile by its cycle mean reduced phase-counterfactual performance just beyond the frozen threshold.

The strongest positive is secondary but clean: a neutral **silent ping** at the end of the hold decoded resident operator, phase, and route factors with median held-out accuracy **1.0 / 1.0 / 1.0**. Resetting the resident state dropped all three to **0.5 / 0.5 / 0.5**. The learned system therefore carried a silent resident state that changed what a later ping encountered, even though that state did not decompose cleanly across the three control surfaces.

Gauge controls remained stable after the v1 repair: maximum base/twin output discrepancy `1.06e-14`, aligned grammar transition TV `0`, and median invariant drift approximately `5.55e-12`, `9.87e-12`, and `3.61e-11` at condition numbers 1, 3, and 10.

The corrected canonical run reproduced byte-for-byte on a second complete run. The full canonical JSON SHA-256 is:

```text
74230dff1d85c605eec43a0489c8bcd240dba2811d19727753db83ccf890ffb4
```

`results/gatgrils_v1.json` is the readable index. The exact full corrected receipt is stored losslessly as `results/gatgrils_v1.json.gz`; the receipt checker follows the index to that gzip artifact automatically.

## Relation to KolmeOvea

[`anttiluode/KolmeOvea`](https://github.com/anttiluode/KolmeOvea) is the independent companion line for the same apical–basket–chandelier question. Its results and GATGRILS v1 both caution against assuming that task learning automatically produces a clean three-door decomposition, while both leave timing as the most robust distinction. The two repos should be read as complementary experiments, not as claims that these toy control surfaces are established biological functions.

`FrequencyAndNeurons` answers a different, upstream question—what sets the local oscillation period—so it is intentionally not folded into this experiment.

## Run / verify

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m gatgrils.v1_experiment --output /tmp/gatgrils_v1_full.json
python -m gatgrils.v1_experiment --check-receipt results/gatgrils_v1.json
```

See `docs/superpowers/specs/2026-09-24-gatgrils-v1-temporal-control-design.md` and `docs/superpowers/plans/2026-09-24-gatgrils-v1-temporal-control.md` for the frozen v1 design and implementation plan. V0’s original design and plan remain under the corresponding `2026-09-23` files.