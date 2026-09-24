import sys, json, numpy as np
sys.path.insert(0, '/home/claude/GATGRILS')
from gatgrils.config import default_config
from gatgrils.experiment import _make_splits, _local_response_sequence, _fit_coordinates_on_validation, _forecast_score
from gatgrils.model import train_rnn
from gatgrils.gauge import GaugeTwin, make_gauge_matrix
cfg = default_config(); A = cfg.alphabet_size

def io_table(system, symbols):
    """Counterfactual input->output ping: from the current resident state h_t, what would the
    system output for EACH possible input symbol? Enters and exits through the fixed interfaces,
    so it needs no similarity invariants: a hidden-basis change cannot touch it."""
    hs = system.hidden_trajectory(symbols)[:-1]
    E = np.eye(system.input_size)
    rows = []
    for h in hs:
        L = np.stack([system.readout(system.step(h, E[s])) for s in range(A)])
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        rows.append(P.ravel())
    return np.array(rows)

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
    Iv, It = io_table(model, va.inputs), io_table(model, te.inputs)
    cio = _fit_coordinates_on_validation(Iv, cfg, seed=400_000 + seed)
    # forecast the SAME target as the canonical grammar (next invariant Jacobian descriptor)
    g = _forecast_score(cm, Fv, va.inputs, Ft, te.inputs, cfg)
    gio = _forecast_score(cio, Iv, va.inputs, It, te.inputs, cfg, target_validation_features=Fv, target_test_features=Ft)
    twin = GaugeTwin(model, make_gauge_matrix(700 + seed, model.hidden_size, 10.0))
    drift = float(np.abs(io_table(twin, te.inputs) - It).max())
    r = dict(seed=seed, k_jac=cm.k, k_io=cio.k, forecast_jac=g, forecast_io=gio,
             MI_mode_jac=mi(cm.predict(Ft), te.mode_labels), MI_mode_io=mi(cio.predict(It), te.mode_labels),
             MI_input_jac=mi(cm.predict(Ft), te.inputs), MI_input_io=mi(cio.predict(It), te.inputs),
             io_gauge_drift_cond10=drift)
    out.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open('/home/claude/gat_io.json', 'w'), indent=1)
ks = ['forecast_jac', 'forecast_io', 'MI_mode_jac', 'MI_mode_io', 'MI_input_jac', 'MI_input_io', 'io_gauge_drift_cond10']
print('median', {k: float(np.median([o[k] for o in out])) for k in ks})
print('io beats jac forecast on', sum(o['forecast_io'] < o['forecast_jac'] for o in out), '/12; ln3 =', np.log(3))
