from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditionalDescriptorForecast:
    means: np.ndarray  # [coordinate, symbol, feature]


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("truth and prediction must have the same nonempty shape")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    centered = y - np.mean(y, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(centered**2)))
    if scale < 1e-12:
        scale = float(np.sqrt(np.mean(y**2)))
    return rmse / max(scale, 1e-12)


def fit_conditional_descriptor_forecast(
    coordinates: np.ndarray,
    input_symbols: np.ndarray,
    next_descriptors: np.ndarray,
    coordinate_count: int,
    alphabet_size: int,
    shrinkage: float = 1.0,
) -> ConditionalDescriptorForecast:
    coords = np.asarray(coordinates, dtype=np.int64)
    symbols = np.asarray(input_symbols, dtype=np.int64)
    targets = np.asarray(next_descriptors, dtype=np.float64)
    if coords.ndim != 1 or symbols.ndim != 1 or targets.ndim != 2:
        raise ValueError("invalid forecast training arrays")
    if len(coords) != len(symbols) or len(coords) != len(targets):
        raise ValueError("forecast training arrays must be aligned")
    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative")
    global_mean = np.mean(targets, axis=0)
    sums = np.zeros((coordinate_count, alphabet_size, targets.shape[1]), dtype=np.float64)
    counts = np.zeros((coordinate_count, alphabet_size), dtype=np.float64)
    for c, s, target in zip(coords, symbols, targets):
        sums[int(c), int(s)] += target
        counts[int(c), int(s)] += 1.0
    means = np.empty_like(sums)
    for c in range(coordinate_count):
        for s in range(alphabet_size):
            denom = counts[c, s] + shrinkage
            if denom <= 0:
                means[c, s] = global_mean
            else:
                means[c, s] = (sums[c, s] + shrinkage * global_mean) / denom
    return ConditionalDescriptorForecast(means=means)


def predict_conditional_descriptor(
    model: ConditionalDescriptorForecast,
    coordinates: np.ndarray,
    input_symbols: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64)
    symbols = np.asarray(input_symbols, dtype=np.int64)
    if coords.shape != symbols.shape:
        raise ValueError("coordinates and symbols must align")
    return model.means[coords, symbols]


def transition_total_variation(a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(a, dtype=np.float64)
    q = np.asarray(b, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 3:
        raise ValueError("transition distributions must have the same 3-D shape")
    return float(np.mean(0.5 * np.sum(np.abs(p - q), axis=-1)))


def mutual_information(labels: np.ndarray, modes: np.ndarray) -> float:
    x = np.asarray(labels, dtype=np.int64)
    y = np.asarray(modes, dtype=np.int64)
    if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("labels and modes must be aligned nonempty vectors")
    kx = int(np.max(x)) + 1
    ky = int(np.max(y)) + 1
    joint = np.zeros((kx, ky), dtype=np.float64)
    for a, b in zip(x, y):
        joint[int(a), int(b)] += 1.0
    joint /= np.sum(joint)
    px = np.sum(joint, axis=1, keepdims=True)
    py = np.sum(joint, axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def switch_agreement(labels: np.ndarray, modes: np.ndarray) -> float:
    x = np.asarray(labels, dtype=np.int64)
    y = np.asarray(modes, dtype=np.int64)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("labels and modes must align and contain at least two points")
    return float(np.mean((x[1:] != x[:-1]) == (y[1:] != y[:-1])))
