# GATGRILS v1 Learned Temporal Control Surfaces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved GATGRILS v1 experiment in which one learned recurrent system receives three architecturally distinct control surfaces—operator modulation, phase-dependent admission, and route-specific publication—and test whether task-only learning makes those surfaces causally separable.

**Architecture:** Keep v0 frozen. Add a parallel `v1_*` implementation: deterministic episode generation, a compact PyTorch float64 recurrent cell with three control surfaces, causal transplant/clamp measurements, black-box ping/window/gauge instrumentation, and one receipt-producing experiment runner. Reuse the v0 invariant descriptor and temporal grammar without changing their formulas.

**Tech Stack:** Python 3.11+, NumPy, PyTorch CPU float64, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-gatgrils-v1-temporal-control-design.md`

## Global Constraints

- Preserve all v0 code, thresholds, receipt, and the documented negative predictive result unchanged.
- Canonical seeds are exactly `0..11`; development seeds are exactly `1000` and `1001`.
- Canonical results may not be inspected until unit/integration tests pass, development seeds learn stably, and all v1 thresholds/hyperparameters are frozen in configuration.
- Training uses task loss only: no factor-classification, gate-label, disentanglement, transplant, or intervention loss.
- The cue is one 8-way token; operator/phase/route factors remain experimenter metadata and are never supplied as three separate model inputs.
- Both temporal streams share the same scalar content channel and the same content weight; their only model-visible distinction is arrival phase.
- Gauge-aware secondary analysis reuses the v0 invariant descriptor and grammar unchanged.
- Negative seeds remain in the machine-readable receipt.

## Review Focus

- **Padded episode timesteps:** steps after GO must not contribute task loss, silence metrics, transplant metrics, or hidden-state targets.
- **Uninformative counterfactual pairs:** operator/phase transplant scoring must exclude matched pairs whose donor and recipient expected emitted vectors are identical, or success can be inflated without a causal flip.
- **Override semantics:** a transplanted control trace replaces only that control value; recipient hidden state must continue evolving normally under the override rather than being clamped or replaced.
- **Publication clamp semantics:** content continues arriving and hidden state continues updating while publication is externally closed; release is scored before any new content update.
- **Gauge wrapper:** the conjugated transition must transform hidden state only and leave exogenous cue/content/GO/clock inputs fixed; base/twin observable outputs must agree numerically.

---

### Task 1: V1 configuration and deterministic episode generator

**Files:**
- Create: `gatgrils/v1_config.py`
- Create: `gatgrils/v1_data.py`
- Create: `tests/test_v1_data.py`

**Interfaces:**
- Produces: `V1Config`, `default_v1_config()`, `Episode`, `generate_episode(seed, factors=None, hold_steps=None, streams=None)`, `generate_dataset(seed, count)`.
- `Episode` exposes fixed-length float64 arrays `cue[T,8]`, `content[T,1]`, `go[T,1]`, `clock[T,2]`, `target[T,2]`, `loss_mask[T,1]`, plus integer metadata `r,p,q,hold_steps,go_index` and source arrays `stream_a[5]`, `stream_b[5]`.

- [ ] **Step 1: Write failing generator tests**

```python
from gatgrils.v1_config import default_v1_config
from gatgrils.v1_data import generate_episode


def test_v1_episode_uses_one_cue_and_shared_content_channel():
    cfg = default_v1_config()
    ep = generate_episode(7, cfg=cfg, factors=(1, 0, 1), hold_steps=8,
                          streams=([1, -1, 1, 1, -1], [-1, -1, 1, -1, 1]))
    assert ep.cue.shape == (cfg.max_steps, 8)
    assert ep.content.shape == (cfg.max_steps, 1)
    assert ep.cue.sum() == 1.0
    assert ep.cue[0, 5] == 1.0  # binary 101 -> r=1,p=0,q=1
    assert set(ep.content[:, 0]).issubset({-1.0, 0.0, 1.0})


def test_v1_stream_identity_is_phase_only():
    cfg = default_v1_config()
    ep = generate_episode(8, cfg=cfg, factors=(0, 1, 0), hold_steps=4,
                          streams=([1, 1, -1, 1, -1], [-1, 1, 1, -1, -1]))
    start = 1 + ep.hold_steps
    assert [ep.content[start + 4*k, 0] for k in range(5)] == ep.stream_a.tolist()
    assert [ep.content[start + 4*k + 2, 0] for k in range(5)] == ep.stream_b.tolist()
    assert ep.content.shape[1] == 1


def test_v1_target_matches_factor_triplet_and_padding_is_masked():
    cfg = default_v1_config()
    ep = generate_episode(9, cfg=cfg, factors=(1, 0, 1), hold_steps=12,
                          streams=([1, -1, 1, -1, 1], [-1, -1, -1, 1, 1]))
    # r=1 on phase A gives sign(1-(-1)+1-(-1)+1)=+1, route q=1
    assert ep.target[ep.go_index].tolist() == [0.0, 1.0]
    assert ep.loss_mask[:ep.go_index + 1].all()
    assert not ep.loss_mask[ep.go_index + 1:].any()
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v1_data.py -q`

Expected: import failure because `gatgrils.v1_config` / `gatgrils.v1_data` do not exist.

- [ ] **Step 3: Implement configuration and generator**

Use the exact configuration shape:

```python
@dataclass(frozen=True)
class V1Config:
    canonical_seeds: tuple[int, ...] = tuple(range(12))
    development_seeds: tuple[int, ...] = (1000, 1001)
    hidden_size: int = 12
    operator_rank: int = 3
    hold_steps: tuple[int, ...] = (4, 8, 12)
    cycles: int = 5
    microcycle: int = 4
    train_episodes: int = 768
    validation_episodes: int = 192
    test_episodes: int = 384
    batch_size: int = 64
    training_epochs: int = 80
    learning_rate: float = 0.008
    weight_decay: float = 1e-5
    task_accuracy_threshold: float = 0.90
    silence_mae_threshold: float = 0.10
    transplant_threshold: float = 0.80
    specificity_margin: float = 0.30
    publication_latent_threshold: float = 0.85
    publication_release_threshold: float = 0.80
    admission_damage_margin: float = 0.20
    rhythmic_damage_margin: float = 0.15
    max_steps: int = 34
```

Factor cue index is `(r << 2) | (p << 1) | q`. The clock is the fixed four-step basis `sin(2πt/4), cos(2πt/4)`. Stream A arrives at global microphase 1 and stream B at global microphase 3 because content begins at `1 + hold_steps` and every allowed hold is a multiple of four. Only the GO timestep carries a nonzero target. `loss_mask` includes cue through GO and excludes padded steps.

- [ ] **Step 4: Run GREEN and full v0 suite**

Run: `python -m pytest tests/test_v1_data.py -q && python -m pytest -q`

Expected: new tests pass; existing v0 suite remains green.

- [ ] **Step 5: Commit**

Commit message: `feat: add v1 temporal-control episodes`

---

### Task 2: Three-surface recurrent cell and task-only trainer

**Files:**
- Create: `gatgrils/v1_model.py`
- Create: `gatgrils/v1_train.py`
- Modify: `pyproject.toml`
- Create: `tests/test_v1_model.py`

**Interfaces:**
- Consumes: `V1Config`, `Episode`, `generate_dataset`.
- Produces: `ThreeSurfaceCell`, `ControlOverride`, `ForwardTrace`, `train_v1(seed, cfg)`, `evaluate_task(model, episodes, cfg)`.
- `ForwardTrace` exposes `hidden[B,T,H]`, `g_op[B,T,R]`, `admission[B,T,1]`, `latent[B,T,2]`, `publication[B,T,2]`, and `emitted[B,T,2]`.

- [ ] **Step 1: Write failing model/override tests**

```python
import numpy as np
import torch
from gatgrils.v1_config import default_v1_config
from gatgrils.v1_model import ThreeSurfaceCell, ControlOverride


def test_three_surface_cell_shapes_and_float64():
    cfg = default_v1_config()
    model = ThreeSurfaceCell(cfg).double()
    B, T = 3, cfg.max_steps
    z8 = torch.zeros(B, T, 8, dtype=torch.float64)
    z1 = torch.zeros(B, T, 1, dtype=torch.float64)
    z2 = torch.zeros(B, T, 2, dtype=torch.float64)
    trace = model(z8, z1, z1, z2)
    assert trace.hidden.shape == (B, T, cfg.hidden_size)
    assert trace.g_op.shape == (B, T, cfg.operator_rank)
    assert trace.admission.shape == (B, T, 1)
    assert trace.publication.shape == (B, T, 2)
    assert trace.emitted.dtype == torch.float64


def test_control_override_replaces_only_selected_surface():
    cfg = default_v1_config()
    torch.manual_seed(0)
    model = ThreeSurfaceCell(cfg).double()
    B, T = 1, cfg.max_steps
    z8 = torch.zeros(B, T, 8, dtype=torch.float64)
    content = torch.zeros(B, T, 1, dtype=torch.float64)
    go = torch.zeros(B, T, 1, dtype=torch.float64)
    clock = torch.zeros(B, T, 2, dtype=torch.float64)
    base = model(z8, content, go, clock)
    donor = torch.full_like(base.admission, 0.25)
    changed = model(z8, content, go, clock, override=ControlOverride(admission=donor))
    assert torch.allclose(changed.admission, donor)
    assert torch.allclose(changed.g_op, base.g_op)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v1_model.py -q`

Expected: import failure because v1 model does not exist.

- [ ] **Step 3: Implement the cell exactly at the three intervention points**

Use these update equations:

```python
g_op = torch.tanh(W_op(h) + b_op)
operator_delta = B_op(g_op * A_op(h))
admission = torch.sigmoid(b_a + u_a(h) + (clock * (v_a + M_a(h))).sum(-1, keepdim=True))
x_eff = admission * content
pre = W_hh(h) + cue_proj(cue) + content_weight * x_eff + go_proj(go) + operator_delta
h_next = torch.tanh(pre)
latent = torch.tanh(W_y(h_next))
publication = torch.sigmoid(W_pub(h_next) + go * u_go + b_pub)
emitted = publication * latent
```

`ControlOverride` has optional tensors `g_op`, `admission`, and `publication`; each override replaces that computed trace at the current step only. Hidden state always evolves from the recipient state using the overridden control.

Add `torch` to project dependencies without changing NumPy/pytest versions.

- [ ] **Step 4: Add failing trainer behavior test**

```python
from gatgrils.v1_train import train_v1, evaluate_task


def test_development_training_reduces_loss_and_is_deterministic():
    cfg = default_v1_config()
    a, ra = train_v1(1000, cfg, epochs=2, episode_count=96)
    b, rb = train_v1(1000, cfg, epochs=2, episode_count=96)
    assert ra["final_loss"] < ra["initial_loss"]
    assert ra == rb
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
```

- [ ] **Step 5: Run RED, implement deterministic Adam training, then GREEN**

Run before implementation: `python -m pytest tests/test_v1_model.py::test_development_training_reduces_loss_and_is_deterministic -q`

Expected: import/name failure for `train_v1`.

Implement seeded PyTorch CPU float64 Adam over deterministic generated episodes. Loss is masked MSE on emitted routes only; there are no auxiliary losses. `evaluate_task` reports selected-route sign accuracy at GO and silent-period mean absolute emitted value using `loss_mask` while excluding GO itself from silence.

Run: `python -m pytest tests/test_v1_model.py -q && python -m pytest -q`

Expected: all pass.

- [ ] **Step 6: Commit**

Commit message: `feat: train v1 three-surface recurrent cell`

---

### Task 3: Causal transplant, clamp, and rhythm interventions

**Files:**
- Create: `gatgrils/v1_interventions.py`
- Create: `tests/test_v1_interventions.py`

**Interfaces:**
- Consumes: trained `ThreeSurfaceCell`, `Episode`, `ForwardTrace`.
- Produces: `make_matched_pair`, `score_specificity_matrix`, `publication_vs_admission_clamp`, `rhythmic_admission_controls`.

- [ ] **Step 1: Write failing matched-pair and override-semantics tests**

```python
from gatgrils.v1_interventions import make_matched_pair, expected_vector


def test_matched_pair_changes_one_factor_only():
    cfg = default_v1_config()
    a, b = make_matched_pair(20, cfg, factor="operator")
    assert (a.p, a.q, a.hold_steps) == (b.p, b.q, b.hold_steps)
    assert a.r != b.r
    assert np.array_equal(a.stream_a, b.stream_a)
    assert np.array_equal(a.stream_b, b.stream_b)


def test_counterfactual_filter_rejects_identical_expected_vectors():
    # identical emitted targets carry no evidence that a transplant flipped computation
    assert expected_vector(sign=1, route=0) == (1.0, 0.0)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v1_interventions.py -q`

Expected: import failure.

- [ ] **Step 3: Implement transplant scoring**

For each factor, build matched pairs with identical streams/hold and one flipped factor. For operator and phase factors, retain only **informative** pairs where donor and recipient expected emitted vectors differ. This is a required Review Focus behavior; report attempted and informative pair counts.

For each recipient, run the donor trace override on each of `g_op`, `admission`, and `publication`. Score success only if the emitted GO vector matches the donor counterfactual route and sign. Return a `3x3` matrix keyed by transplanted surface × changed factor.

- [ ] **Step 4: Write failing clamp tests with a deterministic scripted cell fixture**

Create a tiny test-only scripted model whose latent state sums admitted content and whose publication gate can be externally closed. Assert:

```python
pub = publication_vs_admission_clamp(scripted_model, scripted_episode)
assert pub["publication"]["latent_correct_before_release"] is True
assert pub["publication"]["first_release_correct"] is True
assert pub["admission"]["latent_correct_before_release"] is False
```

- [ ] **Step 5: Run RED, implement clamp semantics, then GREEN**

Publication clamp: set publication override to zero for the frozen late-content interval, continue content and recurrent updates, then release and score emitted output **on the release step before another content event**. Admission clamp: set admission override to zero over the same interval. Return latent/emitted accuracies and raw values.

Rhythmic controls: replace admission within each four-step cycle by that cycle's mean; separately retime both stream arrivals by two microphases while preserving values and weights. Return held-out accuracies.

Run: `python -m pytest tests/test_v1_interventions.py -q && python -m pytest -q`

Expected: all pass.

- [ ] **Step 6: Commit**

Commit message: `feat: add v1 causal control-surface interventions`

---

### Task 4: Silent-ping and temporal-window instruments

**Files:**
- Create: `gatgrils/v1_instrument.py`
- Create: `tests/test_v1_instrument.py`

**Interfaces:**
- Consumes: frozen trained cell and held-out episodes.
- Produces: `silent_ping_tomography`, `temporal_context_curve`.

- [ ] **Step 1: Write failing ping/reset test**

Use a deterministic test cell with a resident binary state that is silent at rest but changes a fixed ping response. Assert that `silent_ping_tomography` decodes the resident factor from ping response and that reset-state decoding falls to chance on the balanced fixture.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v1_instrument.py::test_silent_ping_reads_resident_state_but_reset_does_not -q`

Expected: missing instrument.

- [ ] **Step 3: Implement silent-ping tomography**

At the final hold state, clone hidden state. For each of four microphases, compare one neutral `content=+1` step against an otherwise identical `content=0` step. Concatenate `Δh`, `Δlatent`, and diagnostic-publication-open `Δemitted`. Fit validation-only least-squares linear decoders for `r`, `p`, and `q` encoded as `{-1,+1}`, then score sign on test responses. Repeat after replacing the hold state by zero before both ping and no-ping steps.

- [ ] **Step 4: Write failing temporal-context curve test**

On a scripted exponentially decaying resident-state cell, assert the returned context effect decreases monotonically with lag and the reported half-decay lag is finite.

- [ ] **Step 5: Run RED, implement context curve, then GREEN**

For frozen lags `[4, 8, 12, 16, 24, 32]`, compare the response to the same neutral central ping after two different cue tokens while keeping the later neutral traffic identical. Return normalized response-distance curves for hidden state and each control surface plus the first lag where effect falls to at most half the shortest-lag effect. No v1 pass gate uses these values.

Run: `python -m pytest tests/test_v1_instrument.py -q && python -m pytest -q`

Expected: all pass.

- [ ] **Step 6: Commit**

Commit message: `feat: add v1 ping and temporal-window instruments`

---

### Task 5: Frozen NumPy adapter and gauge-aware v0 grammar reuse

**Files:**
- Create: `gatgrils/v1_gauge.py`
- Create: `tests/test_v1_gauge.py`

**Interfaces:**
- Consumes: trained PyTorch model.
- Produces: `FrozenV1Cell.from_torch`, `step(hidden, cue, content, go, clock)`, `readout(hidden, go)`, `GaugeWrappedV1`, `run_v1_gauge_instrument`.

- [ ] **Step 1: Write failing torch/NumPy parity test**

```python
def test_frozen_numpy_step_matches_torch_step():
    cfg = default_v1_config()
    torch.manual_seed(2)
    model = ThreeSurfaceCell(cfg).double()
    frozen = FrozenV1Cell.from_torch(model)
    h = np.linspace(-0.2, 0.2, cfg.hidden_size)
    cue = np.eye(8)[3]
    content = np.array([1.0])
    go = np.array([0.0])
    clock = np.array([1.0, 0.0])
    th = model.step_from_hidden(torch.tensor(h), torch.tensor(cue), torch.tensor(content),
                                torch.tensor(go), torch.tensor(clock)).hidden.detach().numpy()
    nh = frozen.step(h, cue, content, go, clock)
    assert np.allclose(th, nh, atol=1e-12, rtol=1e-12)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v1_gauge.py::test_frozen_numpy_step_matches_torch_step -q`

Expected: missing frozen adapter.

- [ ] **Step 3: Implement adapter and exact gauge wrapper**

`GaugeWrappedV1.step(h_prime, cue, content, go, clock)` computes `G @ base.step(Ginv @ h_prime, cue, content, go, clock)`. `readout` maps hidden back with `Ginv` before applying base latent/publication readout. Exogenous inputs are not transformed.

- [ ] **Step 4: Write failing exact-equivalence test**

Generate one held-out episode and one bounded-condition gauge. Starting from `h'=G h`, step base and twin across the same episode and assert maximum emitted discrepancy `<= 1e-10`.

- [ ] **Step 5: Run RED, implement v0 descriptor/grammar adapter, then GREEN**

Use the existing v0 finite-difference Jacobian/invariant descriptor functions without modifying their formulas. Build response descriptors along held-out traffic for base and gauge twin and feed them to the existing coordinate/grammar machinery. Return invariant drift and aligned grammar-transition TV as secondary measurements.

Run: `python -m pytest tests/test_v1_gauge.py -q && python -m pytest -q`

Expected: all pass.

- [ ] **Step 6: Commit**

Commit message: `feat: reuse gauge grammar for v1 learned cell`

---

### Task 6: Receipt runner, frozen gates, documentation, development freeze, canonical panel

**Files:**
- Create: `gatgrils/v1_experiment.py`
- Create: `gatgrils/v1_report.py`
- Create: `tests/test_v1_experiment.py`
- Modify: `README.md`
- Create after canonical run: `results/gatgrils_v1.json`

**Interfaces:**
- Consumes: all Tasks 1–5.
- Produces CLI `python -m gatgrils.v1_experiment --output results/gatgrils_v1.json`, per-seed receipts, aggregate gate verdict, README canonical summary.

- [ ] **Step 1: Write failing receipt-schema test**

```python
def test_v1_seed_receipt_contains_primary_and_secondary_sections():
    receipt = run_seed(1000, tiny_v1_config())
    assert set(receipt) >= {"seed", "training", "task", "specificity", "clamps", "rhythm", "ping", "window", "gauge"}
    assert np.asarray(receipt["specificity"]["matrix"]).shape == (3, 3)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_v1_experiment.py -q`

Expected: missing experiment runner.

- [ ] **Step 3: Implement seed runner and aggregate frozen gates**

Aggregate exactly the spec thresholds stored in `V1Config`:

1. at least 9/12 task-valid;
2. median own-surface transplant success ≥ 0.80 for operator, phase, route;
3. own-surface median exceeds best nonmatching surface by ≥ 0.30 for each factor;
4. publication-clamp latent accuracy ≥ 0.85 and first-release accuracy ≥ 0.80;
5. admission clamp damages latent accuracy by ≥ 0.20 relative to publication clamp;
6. cycle-mean admission damages phase counterfactual accuracy by ≥ 0.15.

No secondary ping/window/gauge result can rescue a failed primary gate.

- [ ] **Step 4: Add README contract test before editing README**

Assert README contains separate sections `GATGRILS v0 — frozen result` and `GATGRILS v1 — learned temporal control surfaces`, and that the v0 canonical numbers remain present verbatim (`8/12`, `0.75229`, `0.797405`). Run it RED before the README edit.

- [ ] **Step 5: Run development seeds and freeze**

Run: `python -m gatgrils.v1_experiment --development --output /tmp/gatgrils_v1_dev.json`

Expected before canonical execution: both development seeds finish without NaN/Inf; median development selected-route accuracy ≥ 0.90. If this fails because of a code defect, debug under `superpowers:systematic-debugging`. If it fails because the frozen architecture cannot learn under the planned hyperparameters, adjust only development hyperparameters, update `V1Config`, rerun tests, and record the final values in the receipt **before any canonical seed is executed**.

- [ ] **Step 6: Freeze configuration and run full suite**

Run: `python -m pytest -q`

Expected: all tests pass. After this point, canonical thresholds and training hyperparameters are immutable except correctness fixes with regression tests and written explanation.

- [ ] **Step 7: Run the first complete 12-seed canonical panel**

Run: `python -m gatgrils.v1_experiment --output results/gatgrils_v1.json`

Expected: command completes and writes all 12 seeds, including failures, with an explicit aggregate PASS/FAIL. Do not rerun selectively or discard seeds.

- [ ] **Step 8: Render README summary from the committed receipt and verify reproducibility**

Run the report renderer, then run the canonical command a second time to a temporary file and compare the JSON byte-for-byte or with canonical sorted JSON serialization. Any mismatch is a correctness bug, not scientific variance.

- [ ] **Step 9: Final full verification**

Run: `python -m pytest -q && python -m gatgrils.v1_experiment --check-receipt results/gatgrils_v1.json`

Expected: full suite green; receipt recomputation/check passes.

- [ ] **Step 10: Commit**

Commit message: `feat: complete GATGRILS v1 temporal-control experiment`

---

## Plan self-review

- **Spec coverage:** task generation, three learned surfaces, task-only training, transplant matrix, publication/admission clamp, rhythm-vs-tonic control, silent ping, temporal context, gauge grammar, canonical seed policy, receipt, and claim boundary all have owning tasks.
- **Placeholder scan:** no TODO/TBD implementation placeholders remain.
- **Type consistency:** episode shapes and control-trace shapes are consistent across model, intervention, and instrument tasks.
- **Review Focus coverage:** padded masks are tested in Task 1; informative counterfactual filtering and override semantics in Task 3; publication release timing in Task 3; gauge exogenous-input semantics in Task 5.
