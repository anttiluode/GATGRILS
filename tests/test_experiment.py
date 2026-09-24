import json
from dataclasses import replace

from gatgrils.config import default_config
from gatgrils.experiment import _fit_coordinates_on_validation, _forecast_score, calculate_gates, run_experiment, run_seed


def _smoke_config(tmp_path):
    return replace(
        default_config(),
        seeds=(0,),
        hidden_size=6,
        train_length=220,
        validation_length=120,
        test_length=140,
        training_epochs=4,
        bptt_steps=24,
        discovery_probe_count=6,
        heldout_probe_count=6,
        candidate_clusters=(2, 3),
        output_path=str(tmp_path / "smoke.json"),
    )


def test_run_seed_receipt_is_json_safe_and_records_all_controls(tmp_path):
    result = run_seed(_smoke_config(tmp_path), seed=0)
    encoded = json.dumps(result, sort_keys=True)
    assert encoded
    assert result["seed"] == 0
    assert set(result["gauge_twins"]) == {"cond_1", "cond_3", "cond_10"}
    forecast = result["forecast_nrmse"]
    for key in ("grammar", "activation", "shuffled", "per_step"):
        assert key in forecast
    nll = result["transition_nll"]
    for key in ("ordered", "shuffled_order", "bag"):
        assert key in nll
    assert "heldout_jvp_relative_error_median" in result
    assert "mode_mutual_information" in result["diagnostics"]


def test_gate_calculation_does_not_turn_failed_criteria_into_passes():
    bad = []
    for seed in range(12):
        bad.append(
            {
                "seed": seed,
                "valid": True,
                "max_output_discrepancy": 0.0,
                "gauge_twins": {
                    "cond_1": {"invariant_relative_drift_median": 0.0, "transition_tv": 0.0},
                    "cond_3": {"invariant_relative_drift_median": 0.0, "transition_tv": 0.0},
                    "cond_10": {"invariant_relative_drift_median": 0.0, "transition_tv": 0.0},
                },
                "forecast_nrmse": {"grammar": 1.0, "activation": 0.5, "shuffled": 0.6, "per_step": 0.9},
                "transition_nll": {"ordered": 1.0, "shuffled_order": 0.5, "bag": 0.6},
            }
        )
    gates = calculate_gates(bad)
    assert gates["exact_twin_equivalence"]["pass"] is True
    assert gates["predictive_forecast"]["pass"] is False
    assert gates["ordered_transition_nll"]["pass"] is False
    assert gates["overall_pass"] is False
    assert gates["failure_reasons"]


def test_run_experiment_writes_deterministic_schema_valid_receipt(tmp_path):
    cfg = _smoke_config(tmp_path)
    first = run_experiment(cfg)
    text_a = (tmp_path / "smoke.json").read_text()
    second = run_experiment(cfg)
    text_b = (tmp_path / "smoke.json").read_text()
    assert first == second
    assert text_a == text_b
    payload = json.loads(text_a)
    assert payload["schema_version"] == 1
    assert payload["config"]["seeds"] == [0]
    assert len(payload["per_seed"]) == 1
    assert "gates" in payload and "aggregate" in payload


def test_exact_gauge_twins_do_not_change_recovered_grammar_due_to_cluster_rng(tmp_path):
    result = run_seed(_smoke_config(tmp_path), seed=0)
    for gauge in result["gauge_twins"].values():
        assert gauge["selected_k"] == result["selected_k"]
        assert gauge["transition_tv"] < 1e-8


def test_control_coordinate_space_can_be_scored_against_primary_descriptor_target(tmp_path):
    import numpy as np
    cfg = _smoke_config(tmp_path)
    rng = np.random.default_rng(123)
    coordinate_val = rng.normal(size=(80, 3))
    coordinate_test = rng.normal(size=(60, 3))
    full_val = np.column_stack([coordinate_val, rng.normal(size=(80, 2))])
    full_test = np.column_stack([coordinate_test, rng.normal(size=(60, 2))])
    val_symbols = rng.integers(0, cfg.alphabet_size, size=80)
    test_symbols = rng.integers(0, cfg.alphabet_size, size=60)
    model = _fit_coordinates_on_validation(coordinate_val, cfg, seed=17)
    score = _forecast_score(
        model, coordinate_val, val_symbols, coordinate_test, test_symbols, cfg,
        target_validation_features=full_val, target_test_features=full_test,
    )
    assert np.isfinite(score)
