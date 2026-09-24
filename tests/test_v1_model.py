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
