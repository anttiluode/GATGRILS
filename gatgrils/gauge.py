from __future__ import annotations

import numpy as np

from .model import TanhRNN, _input_matrix


def make_gauge_matrix(seed: int, dimension: int, condition_number: float) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not np.isfinite(condition_number) or condition_number < 1.0:
        raise ValueError("condition_number must be finite and at least 1")
    rng = np.random.default_rng(seed)
    q_left, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q_right, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    if dimension == 1:
        singular = np.ones(1, dtype=np.float64)
    else:
        singular = np.geomspace(1.0, float(condition_number), dimension, dtype=np.float64)
    G = q_left @ np.diag(singular) @ q_right.T
    G = np.asarray(G, dtype=np.float64)
    actual = float(np.linalg.cond(G))
    if not np.isfinite(actual) or actual > condition_number * (1.0 + 1e-10):
        raise ValueError(f"generated gauge condition {actual} exceeds assigned bound {condition_number}")
    return G


def validate_gauge_matrix(matrix: np.ndarray, max_condition: float) -> float:
    G = np.asarray(matrix, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("gauge matrix must be square")
    cond = float(np.linalg.cond(G))
    if not np.isfinite(cond) or cond > max_condition * (1.0 + 1e-12):
        raise ValueError(f"gauge condition {cond} exceeds bound {max_condition}")
    return cond


class GaugeTwin:
    """Exact hidden-coordinate conjugate of a frozen ``TanhRNN``."""

    def __init__(self, base_model: TanhRNN, G: np.ndarray):
        self.base_model = base_model
        self.G = np.asarray(G, dtype=np.float64)
        if self.G.shape != (base_model.hidden_size, base_model.hidden_size):
            raise ValueError("gauge dimension must match hidden size")
        self.G_inv = np.linalg.inv(self.G)

    @property
    def hidden_size(self) -> int:
        return self.base_model.hidden_size

    @property
    def input_size(self) -> int:
        return self.base_model.input_size

    @property
    def output_size(self) -> int:
        return self.base_model.output_size

    def step(self, hidden: np.ndarray, input_vector: np.ndarray) -> np.ndarray:
        base_hidden = self.G_inv @ np.asarray(hidden, dtype=np.float64)
        return self.G @ self.base_model.step(base_hidden, input_vector)

    def readout(self, hidden: np.ndarray) -> np.ndarray:
        return self.base_model.readout(self.G_inv @ np.asarray(hidden, dtype=np.float64))

    def hidden_trajectory(self, inputs: np.ndarray) -> np.ndarray:
        X = _input_matrix(inputs, self.input_size)
        hs = np.zeros((len(X) + 1, self.hidden_size), dtype=np.float64)
        for t, x in enumerate(X):
            hs[t + 1] = self.step(hs[t], x)
        return hs

    def sequence_logits(self, inputs: np.ndarray) -> np.ndarray:
        hs = self.hidden_trajectory(inputs)
        return np.stack([self.readout(h) for h in hs[1:]], axis=0)

    def analytic_step_jacobian(self, hidden: np.ndarray, input_vector: np.ndarray) -> np.ndarray:
        base_hidden = self.G_inv @ np.asarray(hidden, dtype=np.float64)
        J = self.base_model.analytic_step_jacobian(base_hidden, input_vector)
        return self.G @ J @ self.G_inv
