import numpy as np
import torch
from gatgrils.v1_config import default_v1_config
from gatgrils.v1_data import generate_episode
from gatgrils.v1_model import ThreeSurfaceCell, ControlOverride
from gatgrils.v1_interventions import make_matched_pair, expected_vector, is_informative_pair, run_episode


def test_matched_pair_changes_one_factor_only():
    cfg = default_v1_config()
    a, b = make_matched_pair(20, cfg, factor="operator")
    assert (a.p, a.q, a.hold_steps) == (b.p, b.q, b.hold_steps)
    assert a.r != b.r
    assert np.array_equal(a.stream_a, b.stream_a)
    assert np.array_equal(a.stream_b, b.stream_b)


def test_counterfactual_filter_rejects_identical_expected_vectors():
    cfg = default_v1_config()
    a = generate_episode(1, cfg, factors=(0,0,0), hold_steps=4,
                         streams=([1,1,1,1,1], [-1,-1,-1,-1,-1]))
    b = generate_episode(1, cfg, factors=(1,0,0), hold_steps=4,
                         streams=([1,1,1,1,1], [-1,-1,-1,-1,-1]))
    assert expected_vector(a) == (1.0, 0.0)
    assert expected_vector(b) == (1.0, 0.0)
    assert not is_informative_pair(a, b)


def test_publication_override_does_not_change_hidden_state():
    cfg = default_v1_config()
    torch.manual_seed(3)
    model = ThreeSurfaceCell(cfg).double()
    ep = generate_episode(33, cfg, factors=(0,0,0), hold_steps=4)
    base = run_episode(model, ep)
    zero_pub = torch.zeros_like(base.publication)
    changed = run_episode(model, ep, ControlOverride(publication=zero_pub))
    assert torch.allclose(base.hidden, changed.hidden)
    assert not torch.allclose(base.emitted, changed.emitted)
