import numpy as np

from gatgrils.gauge import GaugeTwin, make_gauge_matrix
from gatgrils.invariants import ordered_eigenvalues, response_invariants
from gatgrils.model import TanhRNN
from gatgrils.probes import central_difference_jacobian, make_probe_bank


def _jacobian_sequence(system, symbols, epsilon=1e-5):
    hs = system.hidden_trajectory(symbols)
    basis = make_probe_bank(123, system.hidden_size, system.hidden_size)
    eye = np.eye(system.input_size, dtype=np.float64)
    out = []
    for t, symbol in enumerate(symbols):
        x = eye[int(symbol)]
        out.append(central_difference_jacobian(lambda z, x=x: system.step(z, x), hs[t], basis, epsilon))
    return np.asarray(out)


def test_response_features_are_stable_under_orthogonal_and_general_gauges():
    model = TanhRNN.random(8, 6, 6, 6)
    symbols = np.array(([0, 1, 3, 2, 5, 4] * 5), dtype=np.int64)
    base = response_invariants(_jacobian_sequence(model, symbols))
    for cond, tol in ((1.0, 1e-6), (3.0, 1e-6), (10.0, 1e-4)):
        G = make_gauge_matrix(100 + int(cond), 6, cond)
        twin = GaugeTwin(model, G)
        transformed = response_invariants(_jacobian_sequence(twin, symbols))
        rel = np.linalg.norm(transformed - base, axis=1) / (np.linalg.norm(base, axis=1) + 1e-12)
        assert np.median(rel) <= tol


def test_eigenvalue_order_and_invariants_remain_finite_near_repeated_roots():
    J = np.diag([0.5, 0.5 + 1e-12, -0.2, -0.2 + 1e-12]).astype(np.float64)
    eig = ordered_eigenvalues(J)
    feats = response_invariants(np.stack([J, J + np.eye(4) * 1e-13, J]))
    assert eig.shape == (4,)
    assert np.all(np.isfinite(eig.real))
    assert np.all(np.isfinite(eig.imag))
    assert np.all(np.isfinite(feats))


def test_ordered_trace_features_use_only_present_and_past_jacobians():
    rng = np.random.default_rng(99)
    Js = rng.normal(size=(6, 4, 4)) * 0.1
    base = response_invariants(Js)
    changed = Js.copy()
    changed[4:] += rng.normal(size=changed[4:].shape)
    altered = response_invariants(changed)
    # A change beginning at t=4 must not change descriptors at t<=3.
    assert np.allclose(base[:4], altered[:4])


def test_characteristic_coefficients_are_dimension_normalized():
    J = np.eye(4, dtype=np.float64) * 0.5
    feats = response_invariants(np.stack([J, J, J]))
    # For (x-lambda)^d, dividing coefficient k by C(d,k) gives (-lambda)^k.
    assert np.allclose(feats[1, 4:8], [-0.5, 0.25, -0.125, 0.0625], atol=1e-12)
