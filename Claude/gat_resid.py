import sys, json, numpy as np
sys.path.insert(0, '/home/claude/GATGRILS')
from gatgrils.config import default_config
from gatgrils.experiment import _make_splits, _local_response_sequence, _fit_coordinates_on_validation, _forecast_score
from gatgrils.model import train_rnn
from gatgrils.grammar import fit_response_coordinates
cfg = default_config(); A = cfg.alphabet_size
def mi(a, b):
    j = np.zeros((a.max() + 1, b.max() + 1))
    for x, y in zip(a, b): j[x, y] += 1
    j /= j.sum(); pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True); nz = j > 0
    return float((j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum())
out = []
for seed in range(12):
    tr, va, te = _make_splits(cfg, seed)
    model, _ = train_rnn([(tr.inputs, tr.targets)], cfg, seed=200_000 + seed)
    _, Fv, _ = _local_response_sequence(model, va.inputs, cfg, probe_seed=300_000 + seed * 2)
    _, Ft, _ = _local_response_sequence(model, te.inputs, cfg, probe_seed=300_001 + seed * 2)
    cm = _fit_coordinates_on_validation(Fv, cfg, seed=400_000 + seed)
    # same ping, different resident state: remove what the current event explains (per-input mean,
    # fitted on validation only), cluster what is left
    mu = np.stack([Fv[va.inputs == s].mean(0) for s in range(A)])
    Rv, Rt = Fv - mu[va.inputs], Ft - mu[te.inputs]
    cr = _fit_coordinates_on_validation(Rv, cfg, seed=400_000 + seed)
    # also: force k = mode_count (3) to see whether 3 residual coordinates find the modes
    cut = int(round(len(Rv) * 0.67))
    c3 = fit_response_coordinates(Rv[:cut], Rv[cut:], (3,), 400_000 + seed)
    tgt = dict(target_validation_features=Fv, target_test_features=Ft)
    r = dict(seed=seed,
             forecast_canonical=_forecast_score(cm, Fv, va.inputs, Ft, te.inputs, cfg),
             forecast_event_residual=_forecast_score(cr, Rv, va.inputs, Rt, te.inputs, cfg, **tgt),
             forecast_event_residual_k3=_forecast_score(c3, Rv, va.inputs, Rt, te.inputs, cfg, **tgt),
             MI_mode_canonical=mi(cm.predict(Ft), te.mode_labels),
             MI_mode_event_residual=mi(cr.predict(Rt), te.mode_labels),
             MI_mode_event_residual_k3=mi(c3.predict(Rt), te.mode_labels),
             k_residual=cr.k)
    out.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open('/home/claude/gat_resid.json', 'w'), indent=1)
ks = [k for k in out[0] if k not in ('seed', 'k_residual')]
print('median', {k: round(float(np.median([o[k] for o in out])), 4) for k in ks})
print('residual beats canonical forecast', sum(o['forecast_event_residual'] < o['forecast_canonical'] for o in out), '/12;',
      'k3 residual beats canonical', sum(o['forecast_event_residual_k3'] < o['forecast_canonical'] for o in out), '/12')
