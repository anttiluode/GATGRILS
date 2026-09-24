from __future__ import annotations

from math import comb

import numpy as np


def ordered_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    eig = np.asarray(np.linalg.eigvals(np.asarray(matrix, dtype=np.float64)), dtype=np.complex128)
    # Round only for deterministic tie grouping; retain the original values.
    real_key = np.round(eig.real, 12)
    imag_key = np.round(eig.imag, 12)
    order = np.lexsort((imag_key, real_key))
    return eig[order]


def _single_invariants(J: np.ndarray) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    d = J.shape[0]
    if J.ndim != 2 or J.shape[1] != d:
        raise ValueError("Jacobians must be square")
    kmax = min(4, d)
    traces = []
    power = np.eye(d, dtype=np.float64)
    for _k in range(1, kmax + 1):
        power = power @ J
        traces.append(float(np.trace(power).real / d))
    while len(traces) < 4:
        traces.append(0.0)

    coeffs = np.real_if_close(np.poly(J), tol=1000)
    coeffs = np.asarray(coeffs[1 : 1 + kmax].real, dtype=np.float64)
    char = [float(coeffs[k - 1] / comb(d, k)) for k in range(1, kmax + 1)]
    while len(char) < 4:
        char.append(0.0)

    rank_ratio = float(np.linalg.matrix_rank(J, tol=1e-10) / d)
    return np.asarray(traces + char + [rank_ratio], dtype=np.float64)


def response_invariants(J_sequence: np.ndarray) -> np.ndarray:
    Js = np.asarray(J_sequence, dtype=np.float64)
    if Js.ndim != 3 or Js.shape[1] != Js.shape[2]:
        raise ValueError("J_sequence must have shape (time, dimension, dimension)")
    T, d, _ = Js.shape
    features = []
    for t in range(T):
        base = _single_invariants(Js[t])
        pair = float(np.trace(Js[t - 1] @ Js[t]).real / d) if t >= 1 else 0.0
        triple = (
            float(np.trace(Js[t - 2] @ Js[t - 1] @ Js[t]).real / d)
            if t >= 2
            else 0.0
        )
        features.append(np.concatenate([base, np.asarray([pair, triple], dtype=np.float64)]))
    out = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite invariant descriptor")
    return out
