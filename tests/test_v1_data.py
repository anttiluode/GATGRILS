import numpy as np
from gatgrils.v1_config import default_v1_config
from gatgrils.v1_data import generate_episode


def test_v1_episode_uses_one_cue_and_shared_content_channel():
    cfg = default_v1_config()
    ep = generate_episode(7, cfg=cfg, factors=(1, 0, 1), hold_steps=8,
                          streams=([1, -1, 1, 1, -1], [-1, -1, 1, -1, 1]))
    assert ep.cue.shape == (cfg.max_steps, 8)
    assert ep.content.shape == (cfg.max_steps, 1)
    assert ep.cue.sum() == 1.0
    assert ep.cue[0, 5] == 1.0
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
    assert ep.target[ep.go_index].tolist() == [0.0, 1.0]
    assert ep.loss_mask[:ep.go_index + 1].all()
    assert not ep.loss_mask[ep.go_index + 1:].any()
