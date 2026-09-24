import json
from pathlib import Path

from gatgrils.config import ExperimentConfig, config_to_dict, default_config


def test_default_config_contract_and_json_serialization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = default_config()
    assert isinstance(cfg, ExperimentConfig)
    assert tuple(cfg.seeds) == tuple(range(12))
    assert cfg.dtype == "float64"
    assert tuple(cfg.gauge_condition_numbers) == (1.0, 3.0, 10.0)
    assert cfg.probe_epsilon > 0.0
    assert cfg.output_path
    payload = config_to_dict(cfg)
    encoded = json.dumps(payload, sort_keys=True)
    assert '"dtype": "float64"' in encoded
    assert payload["output_path"] == cfg.output_path
    assert not Path(cfg.output_path).exists(), "import/default config must not write files"


def test_config_contains_required_experiment_dimensions_and_training_settings():
    cfg = default_config()
    assert cfg.hidden_size >= 4
    assert cfg.alphabet_size >= 3
    assert cfg.mode_count >= 2
    assert cfg.train_length > cfg.validation_length > 0
    assert cfg.test_length > 0
    assert cfg.learning_rate > 0
    assert cfg.training_epochs > 0
    assert cfg.candidate_clusters
    assert cfg.discovery_probe_count > 0
    assert cfg.heldout_probe_count > 0
