import sys, numpy as np
sys.path.insert(0, '/home/claude/GATGRILS')
from gatgrils.config import default_config
from gatgrils.experiment import _make_splits, _local_response_sequence
from gatgrils.model import train_rnn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
cfg = default_config(); A = cfg.alphabet_size
def acc(Xv, yv, Xt, yt):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xv, yv).score(Xt, yt)
def eigfeat(Js):
    out = []
    for J in Js:
        e = np.linalg.eigvals(J); e = e[np.lexsort((e.imag, np.abs(e)))]
        out.append(np.r_[np.abs(e), np.abs(e.imag)])
    return np.array(out)
onehot = lambda s: np.eye(A)[s]
rows = []
for seed in range(12):
    tr, va, te = _make_splits(cfg, seed)
    model, _ = train_rnn([(tr.inputs, tr.targets)], cfg, seed=200_000 + seed)
    Jv, Fv, _ = _local_response_sequence(model, va.inputs, cfg, probe_seed=300_000 + seed * 2)
    Jt, Ft, _ = _local_response_sequence(model, te.inputs, cfg, probe_seed=300_001 + seed * 2)
    hv, ht = model.hidden_trajectory(va.inputs)[1:], model.hidden_trajectory(te.inputs)[1:]   # state AFTER reading u_t
    yv, yt = va.mode_labels, te.mode_labels
    rows.append([acc(onehot(va.inputs), yv, onehot(te.inputs), yt),
                 acc(hv, yv, ht, yt),
                 acc(np.c_[Fv, onehot(va.inputs)], yv, np.c_[Ft, onehot(te.inputs)], yt),
                 acc(np.c_[eigfeat(Jv), onehot(va.inputs)], yv, np.c_[eigfeat(Jt), onehot(te.inputs)], yt),
                 acc(np.c_[Jv.reshape(len(Jv), -1)], yv, np.c_[Jt.reshape(len(Jt), -1)], yt)])
    print(seed, np.round(rows[-1], 3), flush=True)
r = np.median(np.array(rows), 0)
print('median mode-decoding accuracy (chance ~0.33):')
for n, v in zip(['input symbol only', 'hidden state h (gauge-dependent)', 'canonical invariant descriptor + input',
                 'full |eigenvalue| set of J + input', 'raw J entries (gauge-dependent)'], r):
    print(f'  {n:42s} {v:.3f}')
