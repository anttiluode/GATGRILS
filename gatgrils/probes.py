from __future__ import annotations

from collections.abc import Callable

import numpy as np


def make_probe_bank(seed: int, dimension: int, count: int) -> np.ndarray:
    if dimension <= 0 or count <= 0:
        raise ValueError("dimension and count must be positive")
    rng = np.random.default_rng(seed)
    if count <= dimension:
        q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        bank = q[:, :count].T
    else:
        bank = rng.normal(size=(count, dimension))
        bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-15
    return np.asarray(bank, dtype=np.float64)


def finite_difference_jvp(
    step_fn: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    h = np.asarray(state, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    plus = np.asarray(step_fn(h + epsilon * v), dtype=np.float64)
    minus = np.asarray(step_fn(h - epsilon * v), dtype=np.float64)
    return (plus - minus) / (2.0 * epsilon)


def central_difference_jacobian(
    step_fn: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    probe_basis: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    probes = np.asarray(probe_basis, dtype=np.float64)
    h = np.asarray(state, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[1] != h.size:
        raise ValueError("probe bank must have shape (probe_count, state_dimension)")
    if np.linalg.matrix_rank(probes) < h.size:
        raise ValueError("probe bank must span the hidden state")
    responses = np.stack(
        [finite_difference_jvp(step_fn, h, direction, epsilon) for direction in probes],
        axis=0,
    )
    # responses[p] = J @ probes[p], hence responses = probes @ J.T.
    J_t, *_ = np.linalg.lstsq(probes, responses, rcond=None)
    return np.asarray(J_t.T, dtype=np.float64)
