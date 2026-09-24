import sys, numpy as np
sys.path.insert(0, '/home/claude/GATGRILS')
from gatgrils.config import default_config
from gatgrils.experiment import _make_splits, _local_response_sequence, _fit_coordinates_on_validation, _forecast_score
from gatgrils.model import train_rnn
from gatgrils.metrics import nrmse
from gatgrils.grammar import fit_response_coordinates
from sklearn.linear_model import Ridge
cfg = default_config(); A = cfg.alphabet_size
def mi(a, b):
    j = np.zeros((a.max() + 1, b.max() + 1))
    for x, y in zip(a, b): j[x, y] += 1
    j /= j.sum(); pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True); nz = j > 0
    return float((j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum())
def design(F, u):          # descriptor x input interactions: lets the map depend on which event arrives
    oh = np.eye(A)[u]
    return np.c_[oh, (F[:, :, None] * oh[:, None, :]).reshape(len(F), -1)]
rows = []
for seed in range(12):
    tr, va, te = _make_splits(cfg, seed)
    model, _ = train_rnn([(tr.inputs, tr.targets)], cfg, seed=200_000 + seed)
    _, Fv, _ = _local_response_sequence(model, va.inputs, cfg, probe_seed=300_000 + seed * 2)
    _, Ft, _ = _local_response_sequence(model, te.inputs, cfg, probe_seed=300_001 + seed * 2)
    cm = _fit_coordinates_on_validation(Fv, cfg, seed=400_000 + seed)
    canon = _forecast_score(cm, Fv, va.inputs, Ft, te.inputs, cfg)
    # continuous ceiling: no quantizer at all
    rg = Ridge(alpha=1e-2).fit(design(Fv[:-1], va.inputs[:-1]), Fv[1:])
    cont = float(nrmse(Ft[1:], rg.predict(design(Ft[:-1], te.inputs[:-1]))))
    # predictive quantizer: cluster states by WHAT THEY PREDICT (each state's predicted next
    # descriptor for every possible event), not by what they look like. causal-state style.
    def futures(F):
        return np.concatenate([rg.predict(design(F, np.full(len(F), s))) for s in range(A)], 1)
    Pv, Pt = futures(Fv), futures(Ft)
    cp = _fit_coordinates_on_validation(Pv, cfg, seed=400_000 + seed)
    pred_q = _forecast_score(cp, Pv, va.inputs, Pt, te.inputs, cfg, target_validation_features=Fv, target_test_features=Ft)
    cut = int(round(len(Pv) * 0.67))
    c3 = fit_response_coordinates(Pv[:cut], Pv[cut:], (3,), 400_000 + seed)
    pred_q3 = _forecast_score(c3, Pv, va.inputs, Pt, te.inputs, cfg, target_validation_features=Fv, target_test_features=Ft)
    rows.append(dict(canonical_kmeans=canon, continuous_no_quantizer=cont, predictive_quantizer=pred_q, predictive_quantizer_k3=pred_q3,
                     MI_mode_canonical=mi(cm.predict(Ft), te.mode_labels), MI_mode_predictive=mi(cp.predict(Pt), te.mode_labels),
                     MI_mode_predictive_k3=mi(c3.predict(Pt), te.mode_labels), k_pred=cp.k))
    print(seed, {k: round(v, 3) for k, v in rows[-1].items()}, flush=True)
ks = list(rows[0])
print('median', {k: round(float(np.median([r[k] for r in rows])), 4) for k in ks})
print('predictive quantizer beats canonical on', sum(r['predictive_quantizer'] < r['canonical_kmeans'] for r in rows), '/12;',
      'k3 version', sum(r['predictive_quantizer_k3'] < r['canonical_kmeans'] for r in rows), '/12')
