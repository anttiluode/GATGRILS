import numpy as np
import torch
from gatgrils.v1_config import default_v1_config, tiny_v1_config
from gatgrils.v1_data import generate_episode
from gatgrils.v1_model import ThreeSurfaceCell
from gatgrils.v1_gauge import FrozenV1Cell, GaugeWrappedV1, run_v1_gauge_instrument


def test_frozen_numpy_step_matches_torch_step():
    cfg = default_v1_config()
    torch.manual_seed(2)
    model = ThreeSurfaceCell(cfg).double()
    frozen = FrozenV1Cell.from_torch(model)
    h = np.linspace(-0.2, 0.2, cfg.hidden_size)
    cue = np.eye(8)[3]
    content = np.array([1.0]); go = np.array([0.0]); clock = np.array([1.0, 0.0])
    th = model.step_from_hidden(torch.tensor(h), torch.tensor(cue), torch.tensor(content),
                                torch.tensor(go), torch.tensor(clock)).hidden.detach().numpy()
    nh = frozen.step(h, cue, content, go, clock)
    assert np.allclose(th, nh, atol=1e-12, rtol=1e-12)


def test_gauge_wrapper_preserves_observable_outputs():
    cfg = default_v1_config(); torch.manual_seed(4)
    model = ThreeSurfaceCell(cfg).double(); frozen = FrozenV1Cell.from_torch(model)
    ep = generate_episode(44, cfg, factors=(1,0,1), hold_steps=4)
    rng=np.random.default_rng(4); q,_=np.linalg.qr(rng.normal(size=(cfg.hidden_size,cfg.hidden_size)))
    s=np.linspace(1.0,3.0,cfg.hidden_size); G=q@np.diag(s)@q.T
    twin=GaugeWrappedV1(frozen,G)
    h=np.zeros(cfg.hidden_size); hp=G@h; maxerr=0.0
    for t in range(ep.go_index+1):
        h=frozen.step(h,ep.cue[t],ep.content[t],ep.go[t],ep.clock[t])
        hp=twin.step(hp,ep.cue[t],ep.content[t],ep.go[t],ep.clock[t])
        maxerr=max(maxerr,float(np.max(np.abs(frozen.readout(h,ep.go[t])-twin.readout(hp,ep.go[t])))))
    assert maxerr <= 1e-10


def test_v1_gauge_instrument_reports_small_invariant_drift():
    cfg=tiny_v1_config(); torch.manual_seed(8)
    model=ThreeSurfaceCell(cfg).double()
    result=run_v1_gauge_instrument(model,cfg,seed=8,condition_numbers=(1.0,3.0))
    assert result['conditions']['1.0']['output_max_abs'] <= 1e-10
    assert result['conditions']['3.0']['median_invariant_relative_drift'] <= 1e-6
