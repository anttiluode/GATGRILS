from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(12))
    dtype: str = "float64"
    alphabet_size: int = 6
    mode_count: int = 3
    hidden_size: int = 8
    train_length: int = 1400
    validation_length: int = 500
    test_length: int = 700
    switch_rate: float = 0.025
    training_epochs: int = 28
    learning_rate: float = 0.035
    bptt_steps: int = 32
    gradient_clip: float = 5.0
    weight_decay: float = 1e-5
    probe_epsilon: float = 1e-5
    discovery_probe_count: int = 8
    heldout_probe_count: int = 8
    gauge_condition_numbers: tuple[float, ...] = (1.0, 3.0, 10.0)
    candidate_clusters: tuple[int, ...] = (2, 3, 4, 5, 6)
    transition_smoothing: float = 0.5
    output_path: str = "results/gatgrils_v0.json"


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return _json_safe(asdict(config))
