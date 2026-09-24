from __future__ import annotations
import numpy as np
import torch
from .v1_config import V1Config
from .v1_data import Episode, generate_episode
from .v1_model import ControlOverride, ThreeSurfaceCell
from .v1_interventions import run_episode


def _clock_at(t: int, microcycle: int = 4) -> torch.Tensor:
    phase = 2.0 * np.pi * float(t) / microcycle
    return torch.tensor([np.sin(phase), np.cos(phase)], dtype=torch.float64)


def ping_response(model: ThreeSurfaceCell, ep: Episode, reset: bool = False) -> np.ndarray:
    start = 1 + ep.hold_steps
    if reset:
        h = torch.zeros(model.cfg.hidden_size, dtype=torch.float64)
    else:
        tr = run_episode(model, ep)
        h = tr.hidden[start - 1].clone()
    cue = torch.zeros(8, dtype=torch.float64)
    go = torch.zeros(1, dtype=torch.float64)
    feats=[]
    with torch.no_grad():
        for phase_idx in range(4):
            clock = _clock_at(phase_idx, model.cfg.microcycle)
            open_pub = ControlOverride(publication=torch.ones(2, dtype=torch.float64))
            ping = model.step_from_hidden(h, cue, torch.ones(1, dtype=torch.float64), go, clock, open_pub)
            null = model.step_from_hidden(h, cue, torch.zeros(1, dtype=torch.float64), go, clock, open_pub)
            feats.extend((ping.hidden-null.hidden).numpy().tolist())
            feats.extend((ping.latent-null.latent).numpy().tolist())
            feats.extend((ping.emitted-null.emitted).numpy().tolist())
    return np.asarray(feats, dtype=np.float64)


def _balanced_episodes(seed: int, count: int, cfg: V1Config) -> list[Episode]:
    if count % 8 != 0:
        count = count - (count % 8)
    out=[]
    rng=np.random.default_rng(seed)
    for i in range(count):
        idx=i % 8
        factors=((idx >> 2) & 1, (idx >> 1) & 1, idx & 1)
        out.append(generate_episode(int(rng.integers(0, 2**62)), cfg, factors=factors))
    return out


def _labels(episodes: list[Episode]) -> np.ndarray:
    return np.asarray([[2*e.r-1, 2*e.p-1, 2*e.q-1] for e in episodes], dtype=np.float64)


def _decode(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> list[float]:
    X=np.c_[np.ones(len(train_x)), train_x]
    Z=np.c_[np.ones(len(test_x)), test_x]
    W=np.linalg.lstsq(X, train_y, rcond=None)[0]
    pred=np.where(Z@W >= 0.0, 1.0, -1.0)
    return [float(np.mean(pred[:,j] == test_y[:,j])) for j in range(3)]


def silent_ping_tomography(model: ThreeSurfaceCell, cfg: V1Config, seed: int) -> dict:
    train_eps=_balanced_episodes(seed*100003+607, cfg.ping_train_episodes, cfg)
    test_eps=_balanced_episodes(seed*100003+613, cfg.ping_test_episodes, cfg)
    Ytr=_labels(train_eps); Yte=_labels(test_eps)
    Xtr=np.stack([ping_response(model,e,False) for e in train_eps])
    Xte=np.stack([ping_response(model,e,False) for e in test_eps])
    Rtr=np.stack([ping_response(model,e,True) for e in train_eps])
    Rte=np.stack([ping_response(model,e,True) for e in test_eps])
    names=("operator","phase","route")
    acc=_decode(Xtr,Ytr,Xte,Yte); reset=_decode(Rtr,Ytr,Rte,Yte)
    return {"factor_accuracy": dict(zip(names,acc)), "reset_accuracy": dict(zip(names,reset)),
            "feature_dim": int(Xtr.shape[1])}


def _state_after_context(model: ThreeSurfaceCell, cue_index: int, delay_steps: int):
    h=torch.zeros(model.cfg.hidden_size,dtype=torch.float64)
    with torch.no_grad():
        for t in range(delay_steps):
            cue=torch.zeros(8,dtype=torch.float64)
            if t == 0: cue[cue_index]=1.0
            st=model.step_from_hidden(h,cue,torch.zeros(1,dtype=torch.float64),
                                      torch.zeros(1,dtype=torch.float64),_clock_at(t,model.cfg.microcycle))
            h=st.hidden
    return h


def temporal_context_curve(model: ThreeSurfaceCell, cfg: V1Config,
                           lags: tuple[int,...]=(4,8,12,16,24,32),
                           time_scale: float=1.0) -> dict:
    raw={k:[] for k in ("hidden","operator","admission","publication")}
    actual=[]
    with torch.no_grad():
        for lag in lags:
            steps=max(1,int(round(lag*time_scale))); actual.append(steps)
            h0=_state_after_context(model,0,steps)
            h1=_state_after_context(model,7,steps)
            clock=_clock_at(steps, cfg.microcycle)
            z8=torch.zeros(8,dtype=torch.float64); z1=torch.zeros(1,dtype=torch.float64)
            s0=model.step_from_hidden(h0,z8,torch.ones(1,dtype=torch.float64),z1,clock)
            s1=model.step_from_hidden(h1,z8,torch.ones(1,dtype=torch.float64),z1,clock)
            raw["hidden"].append(float(torch.linalg.vector_norm(s0.hidden-s1.hidden)))
            raw["operator"].append(float(torch.linalg.vector_norm(s0.g_op-s1.g_op)))
            raw["admission"].append(float(torch.linalg.vector_norm(s0.admission-s1.admission)))
            raw["publication"].append(float(torch.linalg.vector_norm(s0.publication-s1.publication)))
    result={"lags":list(lags),"actual_steps":actual,"time_scale":float(time_scale)}
    for key,vals in raw.items():
        base=vals[0]
        norm=[float(v/base) if base > 1e-15 else 0.0 for v in vals]
        half=None
        for lag,v in zip(lags,norm):
            if v <= 0.5:
                half=int(lag); break
        result[key]={"distance":vals,"normalized":norm,"half_decay_lag":half}
    return result
