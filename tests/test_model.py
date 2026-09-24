from dataclasses import replace

import numpy as np

from gatgrils.config import default_config
from gatgrils.data import make_stream
from gatgrils.model import TanhRNN, cross_entropy, train_rnn


def test_rnn_shapes_finite_values_and_deterministic_initialization():
    a = TanhRNN.random(seed=3, input_size=6, hidden_size=7, output_size=6)
    b = TanhRNN.random(seed=3, input_size=6, hidden_size=7, output_size=6)
    assert np.array_equal(a.W_hh, b.W_hh)
    h = np.zeros(7, dtype=np.float64)
    x = np.eye(6, dtype=np.float64)[2]
    next_h = a.step(h, x)
    assert next_h.shape == (7,)
    assert np.all(np.isfinite(next_h))
    symbols = np.array([0, 1, 2, 3], dtype=np.int64)
    logits = a.sequence_logits(symbols)
    trajectory = a.hidden_trajectory(symbols)
    assert logits.shape == (4, 6)
    assert trajectory.shape == (5, 7)
    assert np.all(np.isfinite(logits))


def test_training_is_deterministic_and_reduces_heldout_cross_entropy():
    cfg = replace(
        default_config(),
        hidden_size=8,
        training_epochs=12,
        learning_rate=0.025,
        bptt_steps=32,
    )
    train = make_stream(100, 650, cfg.alphabet_size, cfg.mode_count, cfg.switch_rate)
    held = make_stream(101, 300, cfg.alphabet_size, cfg.mode_count, cfg.switch_rate)
    untrained = TanhRNN.random(17, cfg.alphabet_size, cfg.hidden_size, cfg.alphabet_size)
    before = cross_entropy(untrained.sequence_logits(held.inputs), held.targets)
    model_a, receipt_a = train_rnn([(train.inputs, train.targets)], cfg, seed=17)
    model_b, receipt_b = train_rnn([(train.inputs, train.targets)], cfg, seed=17)
    after = cross_entropy(model_a.sequence_logits(held.inputs), held.targets)
    assert after < before - 0.05
    assert receipt_a["final_training_loss"] < receipt_a["initial_training_loss"]
    assert receipt_a == receipt_b
    for name in ("W_hh", "W_xh", "b_h", "W_hy", "b_y"):
        assert np.array_equal(getattr(model_a, name), getattr(model_b, name))


def test_train_signature_uses_only_learner_facing_sequences():
    cfg = replace(default_config(), training_epochs=1, bptt_steps=16)
    stream = make_stream(9, 80, cfg.alphabet_size, cfg.mode_count, cfg.switch_rate)
    model, receipt = train_rnn([(stream.inputs, stream.targets)], cfg, seed=2)
    assert "mode" not in " ".join(receipt).lower()
    assert model.hidden_size == cfg.hidden_size
