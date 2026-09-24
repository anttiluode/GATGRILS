import numpy as np

from gatgrils.data import SymbolStream, make_stream


def test_stream_is_exactly_reproducible_and_aligned():
    a = make_stream(7, length=400, alphabet_size=6, mode_count=3, switch_rate=0.03)
    b = make_stream(7, length=400, alphabet_size=6, mode_count=3, switch_rate=0.03)
    assert isinstance(a, SymbolStream)
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.targets, b.targets)
    assert np.array_equal(a.mode_labels, b.mode_labels)
    assert np.array_equal(a.mode_switch_positions, b.mode_switch_positions)
    assert a.inputs.shape == a.targets.shape == a.mode_labels.shape == (400,)
    assert np.all(a.inputs[1:] == a.targets[:-1])
    assert a.inputs.dtype.kind in "iu"
    assert a.targets.min() >= 0 and a.targets.max() < 6


def test_modes_are_persistent_and_switch_positions_match_labels():
    stream = make_stream(11, length=1200, alphabet_size=6, mode_count=3, switch_rate=0.02)
    switches = np.flatnonzero(stream.mode_labels[1:] != stream.mode_labels[:-1]) + 1
    assert np.array_equal(switches, stream.mode_switch_positions)
    assert 3 <= len(switches) <= 80
    segment_lengths = np.diff(np.r_[0, switches, len(stream.mode_labels)])
    assert np.median(segment_lengths) >= 8
    assert set(np.unique(stream.mode_labels)).issubset({0, 1, 2})


def test_generator_modes_induce_distinct_symbol_transition_maps():
    # With switches disabled, changing the latent mode seed should still leave
    # the generator's fixed transition family non-degenerate.
    transitions = set()
    for seed in range(30):
        stream = make_stream(seed, length=20, alphabet_size=6, mode_count=3, switch_rate=0.0)
        transitions.add(tuple(zip(stream.inputs[:6].tolist(), stream.targets[:6].tolist())))
    assert len(transitions) >= 2
