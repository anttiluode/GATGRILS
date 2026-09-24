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


def test_development_training_reduces_loss_and_is_deterministic():
    from gatgrils.v1_train import train_v1
    cfg = default_v1_config()
    a, ra = train_v1(1000, cfg, epochs=2, episode_count=96)
    b, rb = train_v1(1000, cfg, epochs=2, episode_count=96)
    assert ra["final_loss"] < ra["initial_loss"]
    assert ra == rb
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_training_traffic_is_deterministically_resampled_each_epoch():
    from gatgrils.v1_train import _training_dataset_seed
    assert _training_dataset_seed(1000, 0) == _training_dataset_seed(1000, 0)
    assert _training_dataset_seed(1000, 0) != _training_dataset_seed(1000, 1)
    assert _training_dataset_seed(1000, 1) != _training_dataset_seed(1001, 1)


def test_training_pins_tiny_cpu_workload_to_one_torch_thread():
    from gatgrils.v1_train import train_v1
    cfg = default_v1_config()
    torch.set_num_threads(max(2, min(4, torch.get_num_threads())))
    train_v1(1000, cfg, epochs=1, episode_count=32)
    assert torch.get_num_threads() == 1


def test_control_override_mask_recomputes_unmasked_steps_from_evolving_state():
    from gatgrils.v1_model import ControlOverride
    cfg = default_v1_config()
    torch.manual_seed(17)
    model = ThreeSurfaceCell(cfg).double()
    B, T = 1, 6
    cue = torch.zeros(B, T, 8, dtype=torch.float64)
    content = torch.ones(B, T, 1, dtype=torch.float64)
    go = torch.zeros(B, T, 1, dtype=torch.float64)
    clock = torch.zeros(B, T, 2, dtype=torch.float64)
    base = model(cue, content, go, clock)
    donor = torch.full_like(base.g_op, 0.9)
    mask = torch.zeros(B, T, 1, dtype=torch.bool)
    mask[:, 1:4] = True
    changed = model(cue, content, go, clock,
                    override=ControlOverride(g_op=donor, g_op_mask=mask))
    assert torch.allclose(changed.g_op[:, 1:4], donor[:, 1:4])
    expected_last = torch.tanh(model.op_gate(changed.hidden[:, 4]))
    assert torch.allclose(changed.g_op[:, 5], expected_last)
