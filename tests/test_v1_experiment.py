import numpy as np
from gatgrils.v1_config import tiny_v1_config
from gatgrils.v1_experiment import run_seed, aggregate_receipts


def test_v1_seed_receipt_contains_primary_and_secondary_sections():
    cfg=tiny_v1_config()
    receipt=run_seed(1000,cfg)
    assert set(receipt) >= {"seed","training","task","specificity","clamps","rhythm","ping","window","gauge"}
    assert np.asarray(receipt["specificity"]["matrix"]).shape == (3,3)


def test_aggregate_cannot_be_rescued_by_secondary_metrics():
    fake=[]
    for seed in range(12):
        fake.append({"seed":seed,"task":{"selected_route_sign_accuracy":0.5,"silent_mae":0.0},
                     "specificity":{"matrix":[[1,0,0],[0,1,0],[0,0,1]]},
                     "clamps":{"publication":{"latent_correct_before_release":1.0,"first_release_correct":1.0},
                               "admission":{"latent_correct_before_release":0.0},"latent_accuracy_damage":1.0},
                     "rhythm":{"cycle_mean_damage":1.0},"ping":{"factor_accuracy":{"operator":1,"phase":1,"route":1}},
                     "window":{},"gauge":{}})
    agg=aggregate_receipts(fake,tiny_v1_config())
    assert agg['overall_pass'] is False
    assert agg['gates']['task_valid_count']['pass'] is False


def test_readme_keeps_frozen_v0_and_separates_v1():
    from pathlib import Path
    text = Path("README.md").read_text()
    assert "## GATGRILS v0 — frozen result" in text
    assert "## GATGRILS v1 — learned temporal control surfaces" in text
    assert "8/12" in text
    assert "0.75229" in text
    assert "0.797405" in text


def test_check_receipt_loader_follows_canonical_index_to_gzip(tmp_path):
    import gzip, json
    from gatgrils.v1_experiment import _load_expected_receipt
    full = {"schema": "gatgrils-v1-temporal-control-v1", "seeds": [{"seed": 0}], "aggregate": {"overall_pass": False}}
    raw = (json.dumps(full, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    gz = tmp_path / "gatgrils_v1.json.gz"
    with gzip.GzipFile(filename=str(gz), mode="wb", mtime=0) as f:
        f.write(raw)
    index = tmp_path / "gatgrils_v1.json"
    index.write_text(json.dumps({"schema":"gatgrils-v1-canonical-index-v1", "full_receipt_gzip": gz.name}))
    assert _load_expected_receipt(index) == full
