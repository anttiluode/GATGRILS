from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import ExperimentConfig


def _one_hot(symbol: int, size: int) -> np.ndarray:
    x = np.zeros(size, dtype=np.float64)
    x[int(symbol)] = 1.0
    return x


def _input_matrix(inputs: np.ndarray | Sequence[int], input_size: int) -> np.ndarray:
    arr = np.asarray(inputs)
    if arr.ndim == 2:
        if arr.shape[1] != input_size:
            raise ValueError("input matrix has wrong width")
        return np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("inputs must be symbol ids or a 2-D input matrix")
    ids = np.asarray(arr, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= input_size):
        raise ValueError("symbol id outside input vocabulary")
    return np.eye(input_size, dtype=np.float64)[ids]


def cross_entropy(logits: np.ndarray, targets: np.ndarray | Sequence[int]) -> float:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if z.ndim != 2 or y.ndim != 1 or len(z) != len(y):
        raise ValueError("logits and targets have incompatible shapes")
    shifted = z - np.max(z, axis=1, keepdims=True)
    logsum = np.log(np.sum(np.exp(shifted), axis=1))
    chosen = shifted[np.arange(len(y)), y]
    return float(np.mean(logsum - chosen))


@dataclass
class TanhRNN:
    W_hh: np.ndarray
    W_xh: np.ndarray
    b_h: np.ndarray
    W_hy: np.ndarray
    b_y: np.ndarray

    @property
    def hidden_size(self) -> int:
        return int(self.W_hh.shape[0])

    @property
    def input_size(self) -> int:
        return int(self.W_xh.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.W_hy.shape[0])

    @classmethod
    def random(cls, seed: int, input_size: int, hidden_size: int, output_size: int) -> "TanhRNN":
        rng = np.random.default_rng(seed)
        W_hh = rng.normal(0.0, 1.0 / np.sqrt(hidden_size), (hidden_size, hidden_size))
        radius = float(np.max(np.abs(np.linalg.eigvals(W_hh))))
        if radius > 0:
            W_hh *= 0.72 / radius
        W_xh = rng.normal(0.0, 0.45, (hidden_size, input_size))
        W_hy = rng.normal(0.0, 0.35, (output_size, hidden_size))
        return cls(
            W_hh=np.asarray(W_hh, dtype=np.float64),
            W_xh=np.asarray(W_xh, dtype=np.float64),
            b_h=np.zeros(hidden_size, dtype=np.float64),
            W_hy=np.asarray(W_hy, dtype=np.float64),
            b_y=np.zeros(output_size, dtype=np.float64),
        )

    def copy(self) -> "TanhRNN":
        return TanhRNN(*(np.array(a, copy=True) for a in (self.W_hh, self.W_xh, self.b_h, self.W_hy, self.b_y)))

    def step(self, hidden: np.ndarray, input_vector: np.ndarray) -> np.ndarray:
        h = np.asarray(hidden, dtype=np.float64)
        x = np.asarray(input_vector, dtype=np.float64)
        return np.tanh(self.W_hh @ h + self.W_xh @ x + self.b_h)

    def readout(self, hidden: np.ndarray) -> np.ndarray:
        return self.W_hy @ np.asarray(hidden, dtype=np.float64) + self.b_y

    def hidden_trajectory(self, inputs: np.ndarray | Sequence[int]) -> np.ndarray:
        X = _input_matrix(inputs, self.input_size)
        hs = np.zeros((len(X) + 1, self.hidden_size), dtype=np.float64)
        for t, x in enumerate(X):
            hs[t + 1] = self.step(hs[t], x)
        return hs

    def sequence_logits(self, inputs: np.ndarray | Sequence[int]) -> np.ndarray:
        hs = self.hidden_trajectory(inputs)
        return hs[1:] @ self.W_hy.T + self.b_y

    def analytic_step_jacobian(self, hidden: np.ndarray, input_vector: np.ndarray) -> np.ndarray:
        next_h = self.step(hidden, input_vector)
        return (1.0 - next_h * next_h)[:, None] * self.W_hh


def analytic_step_jacobian(model: TanhRNN, hidden: np.ndarray, input_vector: np.ndarray) -> np.ndarray:
    return model.analytic_step_jacobian(hidden, input_vector)


def _sequence_loss(model: TanhRNN, sequences: Iterable[tuple[np.ndarray, np.ndarray]]) -> float:
    losses = []
    weights = []
    for inputs, targets in sequences:
        logits = model.sequence_logits(inputs)
        losses.append(cross_entropy(logits, targets))
        weights.append(len(targets))
    if not losses:
        raise ValueError("at least one training sequence is required")
    return float(np.average(losses, weights=weights))


def _clip_gradients(grads: list[np.ndarray], clip: float) -> None:
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if norm > clip > 0:
        scale = clip / (norm + 1e-12)
        for g in grads:
            g *= scale


def train_rnn(
    streams: Sequence[tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
    seed: int,
) -> tuple[TanhRNN, dict[str, float | int]]:
    sequences = [(np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)) for x, y in streams]
    if not sequences:
        raise ValueError("at least one training sequence is required")
    model = TanhRNN.random(seed, config.alphabet_size, config.hidden_size, config.alphabet_size)
    initial_loss = _sequence_loss(model, sequences)

    # Deterministic Adam over explicit truncated-BPTT chunks.
    params = [model.W_hh, model.W_xh, model.b_h, model.W_hy, model.b_y]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2 = 0.9, 0.999
    adam_step = 0

    for _epoch in range(config.training_epochs):
        for symbols, targets in sequences:
            X = _input_matrix(symbols, model.input_size)
            hidden = np.zeros(model.hidden_size, dtype=np.float64)
            for start in range(0, len(X), config.bptt_steps):
                stop = min(start + config.bptt_steps, len(X))
                chunk_x = X[start:stop]
                chunk_y = targets[start:stop]
                T = len(chunk_x)
                hs = np.zeros((T + 1, model.hidden_size), dtype=np.float64)
                hs[0] = hidden
                logits = np.zeros((T, model.output_size), dtype=np.float64)
                for t in range(T):
                    hs[t + 1] = model.step(hs[t], chunk_x[t])
                    logits[t] = model.readout(hs[t + 1])

                shifted = logits - np.max(logits, axis=1, keepdims=True)
                probs = np.exp(shifted)
                probs /= np.sum(probs, axis=1, keepdims=True)
                dlogits = probs
                dlogits[np.arange(T), chunk_y] -= 1.0
                dlogits /= max(T, 1)

                dW_hh = np.zeros_like(model.W_hh)
                dW_xh = np.zeros_like(model.W_xh)
                db_h = np.zeros_like(model.b_h)
                dW_hy = np.zeros_like(model.W_hy)
                db_y = np.zeros_like(model.b_y)
                dh_next = np.zeros(model.hidden_size, dtype=np.float64)

                for t in range(T - 1, -1, -1):
                    dW_hy += np.outer(dlogits[t], hs[t + 1])
                    db_y += dlogits[t]
                    dh = model.W_hy.T @ dlogits[t] + dh_next
                    da = dh * (1.0 - hs[t + 1] * hs[t + 1])
                    dW_hh += np.outer(da, hs[t])
                    dW_xh += np.outer(da, chunk_x[t])
                    db_h += da
                    dh_next = model.W_hh.T @ da

                grads = [dW_hh, dW_xh, db_h, dW_hy, db_y]
                if config.weight_decay:
                    grads[0] += config.weight_decay * model.W_hh
                    grads[1] += config.weight_decay * model.W_xh
                    grads[3] += config.weight_decay * model.W_hy
                _clip_gradients(grads, config.gradient_clip)

                adam_step += 1
                for i, (param, grad) in enumerate(zip(params, grads)):
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (grad * grad)
                    m_hat = m[i] / (1.0 - beta1**adam_step)
                    v_hat = v[i] / (1.0 - beta2**adam_step)
                    param -= config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

                hidden = hs[-1].copy()

    final_loss = _sequence_loss(model, sequences)
    receipt: dict[str, float | int] = {
        "seed": int(seed),
        "initial_training_loss": float(initial_loss),
        "final_training_loss": float(final_loss),
        "epochs": int(config.training_epochs),
        "optimizer_steps": int(adam_step),
    }
    return model, receipt
