# GATGRILS v0 audit (extra baselines, run against commit e0ff267, all 12 seeds)

Drop these scripts next to a GATGRILS checkout (the path is set at the top of each: `/home/claude/GATGRILS`) and run them with python.

| script | question | median result |
|---|---|---|
| gat_extra.py | Can the next response descriptor be forecast from the input history alone? | input only 0.954, **input bigram (u_{t-1}, u_t) 0.451**, true mode + input (oracle) 0.445, canonical grammar 0.752. The bigram beats the grammar on 12/12 seeds |
| gat_decode.py | Where does the hidden mode live? Linear decoding accuracy, chance about 0.33 | input only 0.365, hidden state h 0.579, **invariant descriptor + input 0.828**, full \|eig\| + input 0.859, raw J 0.876 |
| gat_io.py | Coordinates from a counterfactual input→output ping (from h_t, feed each symbol and read the output) | forecast 0.643 (beats the canonical grammar on 9/12); gauge drift 3e-15 by construction; but mode MI 0.17 nats, *lower* than canonical |
| gat_resid.py | Cluster what the current event does not explain (per-input mean removed) | forecast 0.744, mode MI 0.364. No real change |
| gat_causal.py | Ceiling without a quantizer (ridge on descriptor × input), and a predictive-state quantizer | continuous 0.670; predictive quantizer 0.747 |

For reference, ln 3 = 1.10 nats is the full mode information.
