from __future__ import annotations

from typing import Any


def _status(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def render_receipt_summary(receipt: dict[str, Any]) -> str:
    gates = receipt["gates"]
    agg = receipt["aggregate"]
    forecast = gates["predictive_forecast"]
    nll = gates["ordered_transition_nll"]
    drift = gates["invariant_feature_drift"]["conditions"]
    lines = [
        "<!-- canonical-summary:start -->",
        "### Canonical 12-seed receipt summary",
        "",
        f"- **Overall:** {_status(gates['overall_pass'])}",
        f"- Exact gauge-twin output equivalence (≤ 1e-10): **{_status(gates['exact_twin_equivalence']['pass'])}**",
        (
            "- Median invariant drift: "
            f"cond 1 = {drift['cond_1']['median']:.3e}, "
            f"cond 3 = {drift['cond_3']['median']:.3e}, "
            f"cond 10 = {drift['cond_10']['median']:.3e} "
            f"→ **{_status(gates['invariant_feature_drift']['pass'])}**"
        ),
        (
            f"- Maximum aligned grammar transition TV = {gates['gauge_transition_tv']['max']:.6g} "
            f"(≤ 0.02) → **{_status(gates['gauge_transition_tv']['pass'])}**"
        ),
        (
            f"- Forecast wins vs both activation and shuffled controls = "
            f"{forecast['wins_vs_activation_and_shuffled']}/12; "
            f"median grammar NRMSE = {forecast['median_grammar']:.6g}, "
            f"activation = {forecast['median_activation']:.6g} "
            f"→ **{_status(forecast['pass'])}**"
        ),
        (
            f"- Ordered-transition NLL wins vs both shuffled-order and bag controls = "
            f"{nll['wins_vs_shuffled_and_bag']}/12 → **{_status(nll['pass'])}**"
        ),
        f"- Median held-out JVP relative error (calibration only) = {agg['median_heldout_jvp_relative_error']:.3e}",
        "",
        "This block is rendered from `results/gatgrils_v0.json`; failing gates are retained rather than retuned.",
        "<!-- canonical-summary:end -->",
    ]
    return "\n".join(lines)
