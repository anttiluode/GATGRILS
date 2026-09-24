import numpy as np
import pytest

from gatgrils.gauge import GaugeTwin, make_gauge_matrix, validate_gauge_matrix
from gatgrils.model import TanhRNN


def test_gauge_matrix_has_requested_condition_and_rejects_bad_draws():
    for cond in (1.0, 3.0, 10.0):
        G = make_gauge_matrix(12, 7, cond)
        assert G.dtype == np.float64
        assert np.isclose(np.linalg.cond(G), cond, rtol=1e-10, atol=1e-10)
        validate_gauge_matrix(G, cond * (1 + 1e-9))
    with pytest.raises(ValueError):
        make_gauge_matrix(1, 4, 0.5)
    bad = np.diag([1.0, 1e-8])
    with pytest.raises(ValueError):
        validate_gauge_matrix(bad, 10.0)


def test_conjugate_twin_matches_base_outputs_and_jacobian_similarity():
    model = TanhRNN.random(4, 6, 6, 6)
    G = make_gauge_matrix(9, 6, 3.0)
    twin = GaugeTwin(model, G)
    symbols = np.array([0, 2, 4, 1, 3, 5, 2, 1] * 4, dtype=np.int64)
    base_logits = model.sequence_logits(symbols)
    twin_logits = twin.sequence_logits(symbols)
    assert np.max(np.abs(base_logits - twin_logits)) <= 1e-10

    base_h = model.hidden_trajectory(symbols[:5])[-1]
    twin_h = G @ base_h
    x = np.eye(6)[3]
    J = model.analytic_step_jacobian(base_h, x)
    J_twin = twin.analytic_step_jacobian(twin_h, x)
    expected = G @ J @ np.linalg.inv(G)
    assert np.allclose(J_twin, expected, rtol=1e-11, atol=1e-11)
