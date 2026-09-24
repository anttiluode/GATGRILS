import sys, time, json, numpy as np
sys.path.insert(0, '/home/claude/GATGRILS')
from gatgrils.config import default_config
from gatgrils.experiment import _make_splits, _local_response_sequence, _fit_coordinates_on_validation, _forecast_score
from gatgrils.model import train_rnn
from gatgrils.metrics import fit_conditional_descriptor_forecast, predict_conditional_descriptor, nrmse

cfg = default_config(); A = cfg.alphabet_size
def fc(ctx_val, sym_val, tgt_val, ctx_test, sym_test, tgt_test, k):
    p = fit_conditional_descriptor_forecast(ctx_val[:-1], sym_val[:-1], tgt_val[1:], k, A, shrinkage=1.0)
    return float(nrmse(tgt_test[1:], predict_conditional_descriptor(p, ctx_test[:-1], sym_test[:-1])))
out = []
for seed in range(12):
    t0 = time.time()
    tr, va, te = _make_splits(cfg, seed)
    model, _ = train_rnn([(tr.inputs, tr.targets)], cfg, seed=200_000 + seed)
    _, Fv, _ = _local_response_sequence(model, va.inputs, cfg, probe_seed=300_000 + seed * 2)
    _, Ft, _ = _local_response_sequence(model, te.inputs, cfg, probe_seed=300_001 + seed * 2)
    cm = _fit_coordinates_on_validation(Fv, cfg, seed=400_000 + seed)
    g = _forecast_score(cm, Fv, va.inputs, Ft, te.inputs, cfg)
    zeros_v, zeros_t = np.zeros(len(va.inputs), int), np.zeros(len(te.inputs), int)
    inp = fc(zeros_v, va.inputs, Fv, zeros_t, te.inputs, Ft, 1)                       # input only
    prev = lambda s: np.r_[0, s[:-1]]
    bigram = fc(prev(va.inputs), va.inputs, Fv, prev(te.inputs), te.inputs, Ft, A)     # (u_{t-1}, u_t): reveals the shift = mode
    oracle = fc(va.mode_labels, va.inputs, Fv, te.mode_labels, te.inputs, Ft, cfg.mode_count)  # true mode + input
    # does the next-symbol task get solved? (sanity) accuracy of the frozen RNN on test
    logits = np.stack([model.readout(h) for h in model.hidden_trajectory(te.inputs)[1:]])
    acc = float((logits.argmax(1) == te.targets).mean())
    # how much of the coordinate is just "which symbol"? MI(coord; input) vs MI(coord; mode)
    tc = cm.predict(Ft)
    def mi(a, b):
        j = np.zeros((a.max() + 1, b.max() + 1))
        for x, y in zip(a, b): j[x, y] += 1
        j /= j.sum(); pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
        nz = j > 0; return float((j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum())
    out.append(dict(seed=seed, k=cm.k, grammar=g, input_only=inp, input_bigram=bigram, oracle_mode_plus_input=oracle,
                    rnn_test_acc=acc, MI_coord_input=mi(tc, te.inputs), MI_coord_mode=mi(tc, te.mode_labels)))
    print(out[-1], f'{time.time()-t0:.0f}s', flush=True)
json.dump(out, open('/home/claude/gat_extra.json', 'w'), indent=1)
keys = ['grammar', 'input_only', 'input_bigram', 'oracle_mode_plus_input', 'rnn_test_acc', 'MI_coord_input', 'MI_coord_mode']
print({k: round(float(np.median([o[k] for o in out])), 4) for k in keys})
print('bigram beats grammar on', sum(o['input_bigram'] < o['grammar'] for o in out), '/12; input-only beats grammar on',
      sum(o['input_only'] < o['grammar'] for o in out), '/12')
