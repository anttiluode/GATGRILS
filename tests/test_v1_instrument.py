import numpy as np
import torch
from gatgrils.v1_config import default_v1_config
from gatgrils.v1_data import generate_episode
from gatgrils.v1_model import ThreeSurfaceCell
from gatgrils.v1_instrument import ping_response, temporal_context_curve


def test_ping_response_depends_on_resident_cue_but_reset_does_not():
    cfg = default_v1_config()
    torch.manual_seed(5)
    model = ThreeSurfaceCell(cfg).double()
    streams = ([1,-1,1,-1,1], [-1,1,-1,1,-1])
    a = generate_episode(1, cfg, factors=(0,0,0), hold_steps=4, streams=streams)
    b = generate_episode(1, cfg, factors=(1,1,1), hold_steps=4, streams=streams)
    fa = ping_response(model, a, reset=False)
    fb = ping_response(model, b, reset=False)
    ra = ping_response(model, a, reset=True)
    rb = ping_response(model, b, reset=True)
    assert not np.allclose(fa, fb)
    assert np.allclose(ra, rb)


def test_temporal_context_curve_detects_decay_in_scripted_recurrence():
    cfg = default_v1_config()
    model = ThreeSurfaceCell(cfg).double()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.recurrent.weight.copy_(0.5 * torch.eye(cfg.hidden_size, dtype=torch.float64))
        model.cue_proj.weight[0, 0] = 0.8
        model.cue_proj.weight[0, 7] = -0.8
        model.content_weight[0] = 0.1
    result = temporal_context_curve(model, cfg, lags=(4,8,12,16,24,32))
    curve = np.asarray(result['hidden']['normalized'])
    assert curve[0] == 1.0
    assert np.all(np.diff(curve) <= 1e-8)
    assert result['hidden']['half_decay_lag'] is not None
