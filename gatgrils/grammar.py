from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class CoordinateModel:
    centers: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    k: int
    validation_description_length: float

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        return (x - self.mean) / self.scale

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = self.transform(features)
        distances = np.sum((z[:, None, :] - self.centers[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1).astype(np.int64)


@dataclass(frozen=True)
class TransitionModel:
    probabilities: np.ndarray  # [coordinate, input_symbol, next_coordinate]

    @property
    def coordinate_count(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def alphabet_size(self) -> int:
        return int(self.probabilities.shape[1])


def _validate_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("features must be a nonempty 2-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must be finite")
    return x


def _kmeans_pp_init(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k > len(x):
        raise ValueError("invalid cluster count")
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))].copy()]
    min_dist = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(np.sum(min_dist))
        if total <= 1e-18:
            idx = next((i for i, row in enumerate(x) if not any(np.array_equal(row, c) for c in centers)), 0)
        else:
            probs = min_dist / total
            idx = int(rng.choice(len(x), p=probs))
        centers.append(x[idx].copy())
        min_dist = np.minimum(min_dist, np.sum((x - centers[-1]) ** 2, axis=1))
    return np.asarray(centers, dtype=np.float64)


def _fit_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int = 100) -> np.ndarray:
    centers = _kmeans_pp_init(x, k, seed)
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(max_iter):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = np.mean(x[mask], axis=0)
            else:
                nearest = np.min(distances, axis=1)
                new_centers[j] = x[int(np.argmax(nearest))]
        centers = new_centers
    return centers


def fit_response_coordinates(
    features: np.ndarray,
    validation_features: np.ndarray,
    candidate_k: tuple[int, ...] | list[int],
    seed: int,
) -> CoordinateModel:
    train = _validate_features(features)
    val = _validate_features(validation_features)
    if train.shape[1] != val.shape[1]:
        raise ValueError("training and validation feature widths differ")
    candidates = tuple(int(k) for k in candidate_k)
    if not candidates:
        raise ValueError("candidate_k must not be empty")

    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    z_train = (train - mean) / scale
    z_val = (val - mean) / scale

    best: tuple[float, int, np.ndarray] | None = None
    n_val, d = z_val.shape
    for k in candidates:
        if k > len(z_train):
            continue
        centers = _fit_kmeans(z_train, k, seed + 1009 * k)
        distances = np.sum((z_val[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        distortion = float(np.mean(np.min(distances, axis=1)))
        penalty = float(k * d * np.log(max(n_val, 2)) / max(n_val, 1))
        score = distortion + penalty
        candidate = (score, k, centers)
        if best is None or score < best[0] - 1e-15 or (abs(score - best[0]) <= 1e-15 and k < best[1]):
            best = candidate
    if best is None:
        raise ValueError("no candidate cluster count can be fit")
    score, k, centers = best
    return CoordinateModel(
        centers=np.asarray(centers, dtype=np.float64),
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        k=int(k),
        validation_description_length=float(score),
    )


def fit_transition_model(
    coordinates: np.ndarray,
    input_symbols: np.ndarray,
    smoothing: float = 0.5,
    coordinate_count: int | None = None,
    alphabet_size: int | None = None,
) -> TransitionModel:
    coords = np.asarray(coordinates, dtype=np.int64)
    symbols = np.asarray(input_symbols, dtype=np.int64)
    if coords.ndim != 1 or symbols.ndim != 1 or len(coords) != len(symbols) or len(coords) < 2:
        raise ValueError("coordinates and symbols must be aligned 1-D arrays of length >= 2")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    k = int(coordinate_count or (int(np.max(coords)) + 1))
    a = int(alphabet_size or (int(np.max(symbols)) + 1))
    if np.any(coords < 0) or np.any(coords >= k) or np.any(symbols < 0) or np.any(symbols >= a):
        raise ValueError("coordinate or symbol outside declared range")
    counts = np.full((k, a, k), float(smoothing), dtype=np.float64)
    for t in range(len(coords) - 1):
        counts[coords[t], symbols[t], coords[t + 1]] += 1.0
    probs = counts / np.sum(counts, axis=2, keepdims=True)
    return TransitionModel(probabilities=probs)


def transition_nll(model: TransitionModel, coordinates: np.ndarray, input_symbols: np.ndarray) -> float:
    coords = np.asarray(coordinates, dtype=np.int64)
    symbols = np.asarray(input_symbols, dtype=np.int64)
    if len(coords) != len(symbols) or len(coords) < 2:
        raise ValueError("coordinates and symbols must be aligned")
    logps = []
    for t in range(len(coords) - 1):
        c = int(coords[t])
        s = int(symbols[t])
        nxt = int(coords[t + 1])
        if c >= model.coordinate_count or s >= model.alphabet_size or nxt >= model.coordinate_count:
            logps.append(np.log(1e-15))
        else:
            logps.append(np.log(max(float(model.probabilities[c, s, nxt]), 1e-15)))
    return float(-np.mean(logps))


def fit_bag_coordinate_model(
    coordinates: np.ndarray,
    coordinate_count: int,
    smoothing: float = 0.5,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.int64)
    if smoothing <= 0 or coordinate_count <= 0:
        raise ValueError("invalid bag model parameters")
    counts = np.full(coordinate_count, float(smoothing), dtype=np.float64)
    for c in coords:
        if c < 0 or c >= coordinate_count:
            raise ValueError("coordinate outside declared range")
        counts[int(c)] += 1.0
    return counts / np.sum(counts)


def bag_coordinate_nll(probabilities: np.ndarray, coordinates: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.int64)
    return float(-np.mean(np.log(np.maximum(probs[coords], 1e-15))))


def align_labels(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    ref = np.asarray(reference, dtype=np.int64)
    cand = np.asarray(candidate, dtype=np.int64)
    if ref.shape != cand.shape or ref.ndim != 1:
        raise ValueError("reference and candidate labels must be aligned 1-D arrays")
    k = int(max(np.max(ref), np.max(cand)) + 1)
    confusion = np.zeros((k, k), dtype=np.int64)
    for c, r in zip(cand, ref):
        confusion[int(c), int(r)] += 1

    best_score = -1
    best_perm: tuple[int, ...] | None = None
    for perm in permutations(range(k)):
        score = sum(int(confusion[c, perm[c]]) for c in range(k))
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    mapping = {c: int(best_perm[c]) for c in range(k)}
    aligned = np.asarray([mapping[int(c)] for c in cand], dtype=np.int64)
    return aligned, mapping


def remap_transition_probabilities(probabilities: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    k, a, k2 = probs.shape
    if k != k2 or set(mapping) != set(range(k)):
        raise ValueError("mapping must cover every coordinate label")
    out = np.zeros_like(probs)
    for old_from in range(k):
        new_from = mapping[old_from]
        for old_to in range(k):
            new_to = mapping[old_to]
            out[new_from, :, new_to] = probs[old_from, :, old_to]
    return out
