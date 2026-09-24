from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, config_to_dict, default_config
from .data import SymbolStream, make_stream
from .gauge import GaugeTwin, make_gauge_matrix, validate_gauge_matrix
from .grammar import (
    align_labels,
    bag_coordinate_nll,
    fit_bag_coordinate_model,
    fit_response_coordinates,
    fit_transition_model,
    remap_transition_probabilities,
    transition_nll,
)
from .invariants import response_invariants
from .metrics import (
    fit_conditional_descriptor_forecast,
    mutual_information,
    nrmse,
    predict_conditional_descriptor,
    switch_agreement,
    transition_total_variation,
)
from .model import TanhRNN, train_rnn
from .probes import central_difference_jacobian, finite_difference_jvp, make_probe_bank


def _condition_key(condition: float) -> str:
    if float(condition).is_integer():
        return f"cond_{int(condition)}"
    return f"cond_{condition:g}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _make_splits(config: ExperimentConfig, seed: int) -> tuple[SymbolStream, SymbolStream, SymbolStream]:
    base = 100_000 + 97 * int(seed)
    train = make_stream(base + 1, config.train_length, config.alphabet_size, config.mode_count, config.switch_rate)
    validation = make_stream(base + 2, config.validation_length, config.alphabet_size, config.mode_count, config.switch_rate)
    test = make_stream(base + 3, config.test_length, config.alphabet_size, config.mode_count, config.switch_rate)
    return train, validation, test


def _local_response_sequence(
    system: TanhRNN | GaugeTwin,
    symbols: np.ndarray,
    config: ExperimentConfig,
    probe_seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if config.discovery_probe_count < system.hidden_size:
        raise ValueError("discovery probe bank must span hidden state")
    discovery = make_probe_bank(probe_seed, system.hidden_size, config.discovery_probe_count)
    heldout = make_probe_bank(probe_seed + 1_000_003, system.hidden_size, config.heldout_probe_count)
    if np.allclose(discovery[: min(len(discovery), len(heldout))], heldout[: min(len(discovery), len(heldout))]):
        raise RuntimeError("held-out probe bank collided with discovery bank")
    states = system.hidden_trajectory(symbols)
    inputs = np.eye(system.input_size, dtype=np.float64)
    Js = np.empty((len(symbols), system.hidden_size, system.hidden_size), dtype=np.float64)
    calibration_errors: list[float] = []
    calibration_indices = set(np.linspace(0, len(symbols) - 1, min(48, len(symbols)), dtype=int).tolist())
    for t, symbol in enumerate(symbols):
        x = inputs[int(symbol)]
        step_fn = lambda z, x=x: system.step(z, x)
        J_hat = central_difference_jacobian(step_fn, states[t], discovery, config.probe_epsilon)
        Js[t] = J_hat
        if t in calibration_indices:
            for direction in heldout:
                observed = finite_difference_jvp(step_fn, states[t], direction, config.probe_epsilon)
                predicted = J_hat @ direction
                error = np.linalg.norm(observed - predicted) / (np.linalg.norm(observed) + 1e-12)
                calibration_errors.append(float(error))
    features = response_invariants(Js)
    calibration = float(np.median(calibration_errors)) if calibration_errors else float("nan")
    return Js, features, calibration


def _fit_coordinates_on_validation(
    validation_features: np.ndarray,
    config: ExperimentConfig,
    seed: int,
):
    cut = max(2, int(round(len(validation_features) * 0.67)))
    cut = min(cut, len(validation_features) - 1)
    return fit_response_coordinates(
        validation_features[:cut],
        validation_features[cut:],
        config.candidate_clusters,
        seed,
    )


def _forecast_score(
    coordinate_model,
    validation_features: np.ndarray,
    validation_symbols: np.ndarray,
    test_features: np.ndarray,
    test_symbols: np.ndarray,
    config: ExperimentConfig,
    *,
    target_validation_features: np.ndarray | None = None,
    target_test_features: np.ndarray | None = None,
) -> float:
    target_validation = validation_features if target_validation_features is None else target_validation_features
    target_test = test_features if target_test_features is None else target_test_features
    if len(target_validation) != len(validation_features) or len(target_test) != len(test_features):
        raise ValueError("forecast target and coordinate feature sequences must have matching lengths")
    val_coords = coordinate_model.predict(validation_features)
    test_coords = coordinate_model.predict(test_features)
    predictor = fit_conditional_descriptor_forecast(
        val_coords[:-1],
        validation_symbols[:-1],
        target_validation[1:],
        coordinate_model.k,
        config.alphabet_size,
        shrinkage=1.0,
    )
    predicted = predict_conditional_descriptor(predictor, test_coords[:-1], test_symbols[:-1])
    return float(nrmse(target_test[1:], predicted))


def _activation_forecast_score(
    validation_states: np.ndarray,
    test_states: np.ndarray,
    validation_features: np.ndarray,
    validation_symbols: np.ndarray,
    test_features: np.ndarray,
    test_symbols: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> float:
    model = _fit_coordinates_on_validation(validation_states, config, seed)
    val_coords = model.predict(validation_states)
    test_coords = model.predict(test_states)
    predictor = fit_conditional_descriptor_forecast(
        val_coords[:-1],
        validation_symbols[:-1],
        validation_features[1:],
        model.k,
        config.alphabet_size,
        shrinkage=1.0,
    )
    predicted = predict_conditional_descriptor(predictor, test_coords[:-1], test_symbols[:-1])
    return float(nrmse(test_features[1:], predicted))


def _shuffled_response_forecast_score(
    validation_features: np.ndarray,
    validation_symbols: np.ndarray,
    test_features: np.ndarray,
    test_symbols: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(validation_features))
    shuffled = validation_features[order]
    model = _fit_coordinates_on_validation(shuffled, config, seed + 17)
    shuffled_coords = model.predict(shuffled)
    predictor = fit_conditional_descriptor_forecast(
        shuffled_coords[:-1],
        validation_symbols[:-1],
        shuffled[1:],
        model.k,
        config.alphabet_size,
        shrinkage=1.0,
    )
    test_coords = model.predict(test_features)
    predicted = predict_conditional_descriptor(predictor, test_coords[:-1], test_symbols[:-1])
    return float(nrmse(test_features[1:], predicted))


def _ordered_transition_scores(
    coordinate_model,
    validation_features: np.ndarray,
    validation_symbols: np.ndarray,
    test_features: np.ndarray,
    test_symbols: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, float]:
    val_coords = coordinate_model.predict(validation_features)
    test_coords = coordinate_model.predict(test_features)
    ordered_model = fit_transition_model(
        val_coords,
        validation_symbols,
        smoothing=config.transition_smoothing,
        coordinate_count=coordinate_model.k,
        alphabet_size=config.alphabet_size,
    )
    ordered = transition_nll(ordered_model, test_coords, test_symbols)

    rng = np.random.default_rng(seed)
    shuffled_coords = val_coords[rng.permutation(len(val_coords))]
    shuffled_model = fit_transition_model(
        shuffled_coords,
        validation_symbols,
        smoothing=config.transition_smoothing,
        coordinate_count=coordinate_model.k,
        alphabet_size=config.alphabet_size,
    )
    shuffled = transition_nll(shuffled_model, test_coords, test_symbols)

    bag = fit_bag_coordinate_model(val_coords, coordinate_model.k, smoothing=config.transition_smoothing)
    bag_score = bag_coordinate_nll(bag, test_coords[1:])
    return {"ordered": float(ordered), "shuffled_order": float(shuffled), "bag": float(bag_score)}


def _gauge_metrics(
    model: TanhRNN,
    G: np.ndarray,
    condition: float,
    validation: SymbolStream,
    test: SymbolStream,
    base_validation_features: np.ndarray,
    base_test_features: np.ndarray,
    base_coordinate_model,
    base_validation_coords: np.ndarray,
    base_transition,
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    validate_gauge_matrix(G, condition * (1.0 + 1e-10))
    twin = GaugeTwin(model, G)
    base_logits = model.sequence_logits(test.inputs)
    twin_logits = twin.sequence_logits(test.inputs)
    discrepancy = float(np.max(np.abs(base_logits - twin_logits)))

    _, twin_val_features, _ = _local_response_sequence(
        twin, validation.inputs, config, probe_seed=500_000 + seed * 100 + int(round(condition * 10))
    )
    _, twin_test_features, twin_jvp = _local_response_sequence(
        twin, test.inputs, config, probe_seed=600_000 + seed * 100 + int(round(condition * 10))
    )
    rel = np.linalg.norm(twin_test_features - base_test_features, axis=1) / (
        np.linalg.norm(base_test_features, axis=1) + 1e-12
    )
    drift = float(np.median(rel))

    twin_coordinate_model = _fit_coordinates_on_validation(twin_val_features, config, 400_000 + seed)
    twin_val_coords = twin_coordinate_model.predict(twin_val_features)
    if twin_coordinate_model.k != base_coordinate_model.k:
        tv = 1.0
        alignment_ok = False
        selected_k = int(twin_coordinate_model.k)
    else:
        aligned, mapping = align_labels(base_validation_coords, twin_val_coords)
        # Refit transition in the candidate's native labels, then remap both axes.
        twin_transition = fit_transition_model(
            twin_val_coords,
            validation.inputs,
            smoothing=config.transition_smoothing,
            coordinate_count=twin_coordinate_model.k,
            alphabet_size=config.alphabet_size,
        )
        remapped = remap_transition_probabilities(twin_transition.probabilities, mapping)
        tv = transition_total_variation(base_transition.probabilities, remapped)
        alignment_ok = bool(np.mean(aligned == base_validation_coords) >= 0.0)  # alignment always defined
        selected_k = int(twin_coordinate_model.k)

    return {
        "condition_number": float(np.linalg.cond(G)),
        "max_output_discrepancy": discrepancy,
        "invariant_relative_drift_median": drift,
        "transition_tv": float(tv),
        "alignment_defined": alignment_ok,
        "selected_k": selected_k,
        "heldout_jvp_relative_error_median": float(twin_jvp),
    }


def run_seed(config: ExperimentConfig, seed: int) -> dict[str, Any]:
    train_stream, validation_stream, test_stream = _make_splits(config, seed)
    model, training = train_rnn([(train_stream.inputs, train_stream.targets)], config, seed=200_000 + seed)

    _, validation_features, _ = _local_response_sequence(
        model, validation_stream.inputs, config, probe_seed=300_000 + seed * 2
    )
    _, test_features, heldout_jvp = _local_response_sequence(
        model, test_stream.inputs, config, probe_seed=300_001 + seed * 2
    )
    coordinate_model = _fit_coordinates_on_validation(validation_features, config, seed=400_000 + seed)
    val_coords = coordinate_model.predict(validation_features)
    test_coords = coordinate_model.predict(test_features)
    base_transition = fit_transition_model(
        val_coords,
        validation_stream.inputs,
        smoothing=config.transition_smoothing,
        coordinate_count=coordinate_model.k,
        alphabet_size=config.alphabet_size,
    )

    grammar_score = _forecast_score(
        coordinate_model,
        validation_features,
        validation_stream.inputs,
        test_features,
        test_stream.inputs,
        config,
    )
    validation_states = model.hidden_trajectory(validation_stream.inputs)[:-1]
    test_states = model.hidden_trajectory(test_stream.inputs)[:-1]
    activation_score = _activation_forecast_score(
        validation_states,
        test_states,
        validation_features,
        validation_stream.inputs,
        test_features,
        test_stream.inputs,
        config,
        seed=410_000 + seed,
    )
    shuffled_score = _shuffled_response_forecast_score(
        validation_features,
        validation_stream.inputs,
        test_features,
        test_stream.inputs,
        config,
        seed=420_000 + seed,
    )
    per_step_model = _fit_coordinates_on_validation(validation_features[:, :-2], config, seed=430_000 + seed)
    per_step_score = _forecast_score(
        per_step_model,
        validation_features[:, :-2],
        validation_stream.inputs,
        test_features[:, :-2],
        test_stream.inputs,
        config,
        target_validation_features=validation_features,
        target_test_features=test_features,
    )

    transition_scores = _ordered_transition_scores(
        coordinate_model,
        validation_features,
        validation_stream.inputs,
        test_features,
        test_stream.inputs,
        config,
        seed=440_000 + seed,
    )

    gauge_results: dict[str, Any] = {}
    max_output = 0.0
    for index, condition in enumerate(config.gauge_condition_numbers):
        G = make_gauge_matrix(450_000 + seed * 31 + index, config.hidden_size, float(condition))
        result = _gauge_metrics(
            model,
            G,
            float(condition),
            validation_stream,
            test_stream,
            validation_features,
            test_features,
            coordinate_model,
            val_coords,
            base_transition,
            config,
            seed=seed,
        )
        gauge_results[_condition_key(float(condition))] = result
        max_output = max(max_output, result["max_output_discrepancy"])

    # Diagnostic labels are touched only after model fitting, coordinate selection,
    # forecast scoring, transition scoring, and gauge scoring are complete.
    diagnostic_modes = test_stream.mode_labels
    diagnostics = {
        "mode_mutual_information": mutual_information(test_coords, diagnostic_modes),
        "mode_switch_agreement": switch_agreement(test_coords, diagnostic_modes),
        "mode_switch_count": int(len(test_stream.mode_switch_positions)),
    }

    result = {
        "seed": int(seed),
        "valid": bool(max_output <= 1e-10),
        "training": training,
        "selected_k": int(coordinate_model.k),
        "validation_description_length": float(coordinate_model.validation_description_length),
        "max_output_discrepancy": float(max_output),
        "heldout_jvp_relative_error_median": float(heldout_jvp),
        "gauge_twins": gauge_results,
        "forecast_nrmse": {
            "grammar": float(grammar_score),
            "activation": float(activation_score),
            "shuffled": float(shuffled_score),
            "per_step": float(per_step_score),
        },
        "transition_nll": transition_scores,
        "diagnostics": diagnostics,
    }
    return _json_safe(result)


def calculate_gates(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    expected_seed_count = 12
    full_panel = len(per_seed) == expected_seed_count
    exact = full_panel and all(bool(r.get("valid", False)) and r["max_output_discrepancy"] <= 1e-10 for r in per_seed)

    invariant_details: dict[str, Any] = {}
    invariant_pass = full_panel
    for key, tolerance in (("cond_1", 1e-6), ("cond_3", 1e-6), ("cond_10", 1e-4)):
        values = [float(r["gauge_twins"][key]["invariant_relative_drift_median"]) for r in per_seed if key in r["gauge_twins"]]
        median = float(np.median(values)) if values else float("inf")
        passed = len(values) == expected_seed_count and median <= tolerance
        invariant_details[key] = {"median": median, "threshold": tolerance, "pass": bool(passed)}
        invariant_pass = invariant_pass and passed

    tv_values = [
        float(g["transition_tv"])
        for r in per_seed
        for g in r.get("gauge_twins", {}).values()
    ]
    max_tv = float(np.max(tv_values)) if tv_values else float("inf")
    tv_pass = full_panel and len(tv_values) == expected_seed_count * 3 and max_tv <= 0.02

    forecast_wins = sum(
        float(r["forecast_nrmse"]["grammar"]) < float(r["forecast_nrmse"]["activation"])
        and float(r["forecast_nrmse"]["grammar"]) < float(r["forecast_nrmse"]["shuffled"])
        for r in per_seed
    )
    grammar_values = [float(r["forecast_nrmse"]["grammar"]) for r in per_seed]
    activation_values = [float(r["forecast_nrmse"]["activation"]) for r in per_seed]
    median_grammar = float(np.median(grammar_values)) if grammar_values else float("inf")
    median_activation = float(np.median(activation_values)) if activation_values else float("inf")
    forecast_pass = (
        full_panel
        and forecast_wins >= 9
        and median_grammar <= 0.90 * median_activation
    )

    nll_wins = sum(
        float(r["transition_nll"]["ordered"]) < float(r["transition_nll"]["shuffled_order"])
        and float(r["transition_nll"]["ordered"]) < float(r["transition_nll"]["bag"])
        for r in per_seed
    )
    nll_pass = full_panel and nll_wins >= 9

    failure_reasons = []
    if not full_panel:
        failure_reasons.append(f"canonical panel requires 12 seeds; receipt contains {len(per_seed)}")
    if not exact:
        failure_reasons.append("exact gauge-twin equivalence gate failed")
    if not invariant_pass:
        failure_reasons.append("invariant-feature drift gate failed")
    if not tv_pass:
        failure_reasons.append("aligned gauge transition-TV gate failed")
    if not forecast_pass:
        failure_reasons.append("grammar-conditioned forecast gate failed")
    if not nll_pass:
        failure_reasons.append("ordered transition-NLL gate failed")

    overall = exact and invariant_pass and tv_pass and forecast_pass and nll_pass
    return _json_safe(
        {
            "exact_twin_equivalence": {"pass": bool(exact), "threshold": 1e-10},
            "invariant_feature_drift": {"pass": bool(invariant_pass), "conditions": invariant_details},
            "gauge_transition_tv": {"pass": bool(tv_pass), "max": max_tv, "threshold": 0.02},
            "predictive_forecast": {
                "pass": bool(forecast_pass),
                "wins_vs_activation_and_shuffled": int(forecast_wins),
                "required_wins": 9,
                "median_grammar": median_grammar,
                "median_activation": median_activation,
                "required_relative_ratio": 0.90,
            },
            "ordered_transition_nll": {
                "pass": bool(nll_pass),
                "wins_vs_shuffled_and_bag": int(nll_wins),
                "required_wins": 9,
            },
            "overall_pass": bool(overall),
            "failure_reasons": failure_reasons,
        }
    )


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed:
        return {}
    return _json_safe(
        {
            "seed_count": len(per_seed),
            "median_training_final_loss": float(np.median([r["training"]["final_training_loss"] for r in per_seed])),
            "median_heldout_jvp_relative_error": float(np.median([r["heldout_jvp_relative_error_median"] for r in per_seed])),
            "median_forecast_nrmse": {
                key: float(np.median([r["forecast_nrmse"][key] for r in per_seed]))
                for key in ("grammar", "activation", "shuffled", "per_step")
            },
            "median_transition_nll": {
                key: float(np.median([r["transition_nll"][key] for r in per_seed]))
                for key in ("ordered", "shuffled_order", "bag")
            },
            "selected_k": [int(r["selected_k"]) for r in per_seed],
            "median_mode_mutual_information": float(np.median([r["diagnostics"]["mode_mutual_information"] for r in per_seed])),
        }
    )


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    per_seed = [run_seed(config, int(seed)) for seed in config.seeds]
    payload = {
        "schema_version": 1,
        "experiment": "GATGRILS v0",
        "config": config_to_dict(config),
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
        "gates": calculate_gates(per_seed),
    }
    payload = _json_safe(payload)
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen GATGRILS v0 experiment")
    parser.add_argument("--config", default="default", choices=["default"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = default_config()
    if args.output:
        config = replace(config, output_path=args.output)
    result = run_experiment(config)
    print(json.dumps({"output": config.output_path, "gates": result["gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
