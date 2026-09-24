import json
from pathlib import Path

import pytest

from gatgrils.report import render_receipt_summary


def test_readme_documents_run_command_math_gates_and_claim_boundaries():
    text = Path("README.md").read_text(encoding="utf-8")
    required = [
        "python -m gatgrils.experiment --config default --output results/gatgrils_v0.json",
        "F_G(h', u) = G F(G^-1 h', u)",
        "1e-10",
        "1e-6",
        "1e-4",
        "0.02",
        "9/12",
        "10%",
        "synthetic recurrent model",
        "does not claim",
        "held-out probe",
        "mode labels",
    ]
    for phrase in required:
        assert phrase.lower() in text.lower()


def test_readme_canonical_summary_matches_receipt_when_present():
    receipt = Path("results/gatgrils_v0.json")
    if not receipt.exists():
        pytest.skip("canonical receipt has not been generated yet")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    summary = render_receipt_summary(payload)
    text = Path("README.md").read_text(encoding="utf-8")
    assert summary in text
