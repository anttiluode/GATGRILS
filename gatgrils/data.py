from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymbolStream:
    inputs: np.ndarray
    targets: np.ndarray
    mode_labels: np.ndarray
    mode_switch_positions: np.ndarray


def _transition_tables(alphabet_size: int, mode_count: int) -> np.ndarray:
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2")
    if mode_count < 1:
        raise ValueError("mode_count must be positive")
    symbols = np.arange(alphabet_size, dtype=np.int64)
    tables = np.empty((mode_count, alphabet_size), dtype=np.int64)
    for mode in range(mode_count):
        shift = 1 + (mode % (alphabet_size - 1))
        tables[mode] = (symbols + shift) % alphabet_size
    return tables


def make_stream(
    seed: int,
    length: int,
    alphabet_size: int,
    mode_count: int,
    switch_rate: float,
) -> SymbolStream:
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0.0 <= switch_rate <= 1.0:
        raise ValueError("switch_rate must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    tables = _transition_tables(alphabet_size, mode_count)
    inputs = np.empty(length, dtype=np.int64)
    targets = np.empty(length, dtype=np.int64)
    modes = np.empty(length, dtype=np.int64)

    symbol = int(rng.integers(alphabet_size))
    mode = int(rng.integers(mode_count))
    for t in range(length):
        inputs[t] = symbol
        modes[t] = mode
        next_symbol = int(tables[mode, symbol])
        targets[t] = next_symbol
        symbol = next_symbol
        if t + 1 < length and mode_count > 1 and rng.random() < switch_rate:
            offset = int(rng.integers(1, mode_count))
            mode = (mode + offset) % mode_count

    switches = np.flatnonzero(modes[1:] != modes[:-1]).astype(np.int64) + 1
    return SymbolStream(
        inputs=inputs,
        targets=targets,
        mode_labels=modes,
        mode_switch_positions=switches,
    )
