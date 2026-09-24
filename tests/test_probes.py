import numpy as np

from gatgrils.model import TanhRNN
from gatgrils.probes import central_difference_jacobian, finite_difference_jvp, make_probe_bank


def test_central_difference_recovers_analytic_jacobian_and_heldout_jvps():
    model = TanhRNN.random(5, 6, 8, 6)
    symbols = np.array([1, 2, 4, 0, 3, 1], dtype=np.int64)
    h = model.hidden_trajectory(symbols)[-1]
    x = np.eye(6, dtype=np.float64)[2]
    discovery = make_probe_bank(20, 8, 8)
    heldout = make_probe_bank(21, 8, 8)
    assert not np.allclose(discovery, heldout)
    J_hat = central_difference_jacobian(lambda z: model.step(z, x), h, discovery, 1e-5)
    J_true = model.analytic_step_jacobian(h, x)
    assert np.linalg.norm(J_hat - J_true) / np.linalg.norm(J_true) < 1e-7
    errors = []
    for v in heldout:
        fd = finite_difference_jvp(lambda z: model.step(z, x), h, v, 1e-5)
        pred = J_hat @ v
        errors.append(np.linalg.norm(fd - pred) / (np.linalg.norm(fd) + 1e-12))
    assert np.median(errors) < 1e-7
