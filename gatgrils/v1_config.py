from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class V1Config:
    canonical_seeds: tuple[int, ...] = tuple(range(12))
    development_seeds: tuple[int, ...] = (1000, 1001)
    hidden_size: int = 16
    operator_rank: int = 4
    hold_steps: tuple[int, ...] = (4, 8, 12)
    cycles: int = 5
    microcycle: int = 4
    train_episodes: int = 768
    validation_episodes: int = 192
    test_episodes: int = 384
    batch_size: int = 768
    training_epochs: int = 140
    learning_rate: float = 0.008
    weight_decay: float = 1e-5
    go_loss_weight: float = 16.0
    gradient_clip: float = 5.0
    task_accuracy_threshold: float = 0.90
    silence_mae_threshold: float = 0.10
    transplant_threshold: float = 0.80
    specificity_margin: float = 0.30
    publication_latent_threshold: float = 0.85
    publication_release_threshold: float = 0.80
    admission_damage_margin: float = 0.20
    rhythmic_damage_margin: float = 0.15
    max_steps: int = 34
    transplant_pairs: int = 96
    clamp_pairs: int = 96
    ping_train_episodes: int = 128
    ping_test_episodes: int = 128
    gauge_steps: int = 96
    probe_epsilon: float = 1e-5


def default_v1_config() -> V1Config:
    return V1Config()


def tiny_v1_config() -> V1Config:
    return replace(V1Config(), train_episodes=96, validation_episodes=48, test_episodes=64,
                   batch_size=32, training_epochs=2, transplant_pairs=12, clamp_pairs=12,
                   ping_train_episodes=32, ping_test_episodes=32, gauge_steps=24)


def v1_config_to_dict(config: V1Config) -> dict[str, Any]:
    raw = asdict(config)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in raw.items()}
