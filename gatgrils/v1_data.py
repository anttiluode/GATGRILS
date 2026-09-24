from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from .v1_config import V1Config, default_v1_config


@dataclass(frozen=True)
class Episode:
    cue: np.ndarray
    content: np.ndarray
    go: np.ndarray
    clock: np.ndarray
    target: np.ndarray
    loss_mask: np.ndarray
    r: int
    p: int
    q: int
    hold_steps: int
    go_index: int
    stream_a: np.ndarray
    stream_b: np.ndarray

    @property
    def cue_index(self) -> int:
        return (self.r << 2) | (self.p << 1) | self.q

    @property
    def selected_stream(self) -> np.ndarray:
        return self.stream_a if self.p == 0 else self.stream_b

    @property
    def target_sign(self) -> int:
        x = self.selected_stream
        value = int(np.sum(x)) if self.r == 0 else int(x[0] - x[1] + x[2] - x[3] + x[4])
        return 1 if value > 0 else -1


def _stream(values: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != (5,) or not np.all(np.isin(arr, [-1.0, 1.0])):
        raise ValueError("each stream must contain exactly five values in {-1,+1}")
    return arr


def _clock(max_steps: int, microcycle: int) -> np.ndarray:
    phase = 2.0 * np.pi * np.arange(max_steps, dtype=np.float64) / microcycle
    return np.stack([np.sin(phase), np.cos(phase)], axis=1)


def generate_episode(seed: int, cfg: V1Config | None = None,
                     factors: tuple[int, int, int] | None = None,
                     hold_steps: int | None = None,
                     streams: tuple[Iterable[int], Iterable[int]] | None = None) -> Episode:
    cfg = cfg or default_v1_config()
    rng = np.random.default_rng(int(seed))
    if factors is None:
        factors = tuple(int(v) for v in rng.integers(0, 2, size=3))
    if len(factors) != 3 or any(v not in (0, 1) for v in factors):
        raise ValueError("factors must be a binary (r,p,q) triple")
    r, p, q = map(int, factors)
    if hold_steps is None:
        hold_steps = int(rng.choice(cfg.hold_steps))
    if hold_steps not in cfg.hold_steps:
        raise ValueError("hold_steps must be one of the frozen hold durations")
    if streams is None:
        a = rng.choice(np.array([-1.0, 1.0]), size=cfg.cycles)
        b = rng.choice(np.array([-1.0, 1.0]), size=cfg.cycles)
    else:
        a, b = (_stream(streams[0]), _stream(streams[1]))
    if len(a) != cfg.cycles or len(b) != cfg.cycles:
        raise ValueError("stream length must equal cfg.cycles")

    cue = np.zeros((cfg.max_steps, 8), dtype=np.float64)
    content = np.zeros((cfg.max_steps, 1), dtype=np.float64)
    go = np.zeros((cfg.max_steps, 1), dtype=np.float64)
    target = np.zeros((cfg.max_steps, 2), dtype=np.float64)
    mask = np.zeros((cfg.max_steps, 1), dtype=bool)
    cue[0, (r << 2) | (p << 1) | q] = 1.0
    start = 1 + hold_steps
    for k in range(cfg.cycles):
        content[start + cfg.microcycle * k, 0] = a[k]
        content[start + cfg.microcycle * k + 2, 0] = b[k]
    go_index = start + cfg.cycles * cfg.microcycle
    if go_index >= cfg.max_steps:
        raise ValueError("episode exceeds cfg.max_steps")
    go[go_index, 0] = 1.0
    selected = a if p == 0 else b
    score = float(np.sum(selected)) if r == 0 else float(selected[0] - selected[1] + selected[2] - selected[3] + selected[4])
    sign = 1.0 if score > 0.0 else -1.0
    target[go_index, q] = sign
    mask[:go_index + 1] = True
    return Episode(cue=cue, content=content, go=go, clock=_clock(cfg.max_steps, cfg.microcycle),
                   target=target, loss_mask=mask, r=r, p=p, q=q, hold_steps=int(hold_steps),
                   go_index=int(go_index), stream_a=np.asarray(a, dtype=np.float64),
                   stream_b=np.asarray(b, dtype=np.float64))


def generate_dataset(seed: int, count: int, cfg: V1Config | None = None) -> list[Episode]:
    cfg = cfg or default_v1_config()
    if count < 1:
        raise ValueError("count must be positive")
    root = np.random.default_rng(int(seed))
    episode_seeds = root.integers(0, np.iinfo(np.int64).max, size=count, dtype=np.int64)
    return [generate_episode(int(s), cfg=cfg) for s in episode_seeds]
