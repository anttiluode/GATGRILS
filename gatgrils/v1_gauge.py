from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .v1_config import V1Config
from .v1_data import generate_episode
from .invariants import response_invariants
from .grammar import fit_response_coordinates, fit_transition_model, align_labels, remap_transition_probabilities


def _arr(t): return t.detach().cpu().numpy().astype(np.float64,copy=True)
def _sigmoid(x): return 1.0/(1.0+np.exp(-x))

@dataclass
class FrozenV1Cell:
    cfg: V1Config; w: dict[str,np.ndarray]
    @classmethod
    def from_torch(cls,model):
        w={name:_arr(p) for name,p in model.named_parameters()}
        return cls(model.cfg,w)
    @property
    def hidden_size(self): return self.cfg.hidden_size
    def controls(self,h,clock):
        w=self.w
        g=np.tanh(w['op_gate.weight']@h+w['op_gate.bias'])
        phase=w['adm_phase_base']+w['adm_phase_h.weight']@h
        adm=_sigmoid(w['adm_bias']+(w['adm_h.weight']@h)+(clock*phase).sum())
        return g,float(np.asarray(adm).reshape(-1)[0])
    def step(self,h,cue,content,go,clock):
        w=self.w; h=np.asarray(h,dtype=np.float64); cue=np.asarray(cue); content=np.asarray(content); go=np.asarray(go); clock=np.asarray(clock)
        g,adm=self.controls(h,clock)
        op=w['op_b.weight']@(g*(w['op_a.weight']@h))
        pre=w['recurrent.weight']@h+w['recurrent.bias']+w['cue_proj.weight']@cue+adm*float(content[0])*w['content_weight']+w['go_proj.weight']@go+op
        return np.tanh(pre)
    def readout(self,h,go):
        w=self.w; h=np.asarray(h,dtype=np.float64); go=np.asarray(go,dtype=np.float64)
        latent=np.tanh(w['latent_head.weight']@h+w['latent_head.bias'])
        pub=_sigmoid(w['pub_h.weight']@h+w['pub_h.bias']+float(go[0])*w['pub_go'])
        return pub*latent

class GaugeWrappedV1:
    def __init__(self,base,G):
        self.base=base; self.G=np.asarray(G,dtype=np.float64); self.Ginv=np.linalg.inv(self.G)
    def step(self,hp,cue,content,go,clock): return self.G@self.base.step(self.Ginv@hp,cue,content,go,clock)
    def readout(self,hp,go): return self.base.readout(self.Ginv@hp,go)

def _gauge(seed,d,cond):
    rng=np.random.default_rng(seed); ql,_=np.linalg.qr(rng.normal(size=(d,d))); qr,_=np.linalg.qr(rng.normal(size=(d,d)))
    s=np.geomspace(1.0,float(cond),d); return ql@np.diag(s)@qr.T

def _jac(step_fn,h,eps):
    d=len(h); J=np.empty((d,d),dtype=np.float64)
    for j in range(d):
        e=np.zeros(d); e[j]=1.0
        J[:,j]=(step_fn(h+eps*e)-step_fn(h-eps*e))/(2*eps)
    return J

def _event_symbol(ep,t):
    if ep.cue[t].any(): return 0
    if ep.go[t,0]!=0: return 3
    if ep.content[t,0]!=0: return 1 if t%4==1 else 2
    return 4

def _trajectory(cell,ep,n,G=None):
    d=cell.base.hidden_size if isinstance(cell,GaugeWrappedV1) else cell.hidden_size
    h=np.zeros(d) if G is None else G@np.zeros(d); states=[]; outs=[]
    for t in range(n):
        states.append(h.copy())
        h=cell.step(h,ep.cue[t],ep.content[t],ep.go[t],ep.clock[t]); outs.append(cell.readout(h,ep.go[t]))
    return np.asarray(states),np.asarray(outs)

def _grammar(features,symbols,seed):
    n=len(features); split=max(6,int(0.67*n)); split=min(split,n-3)
    model=fit_response_coordinates(features[:split],features[split:],(2,3,4,5,6),seed)
    labels=model.predict(features); tm=fit_transition_model(labels,symbols,smoothing=0.5,coordinate_count=model.k,alphabet_size=5)
    return model,labels,tm

def run_v1_gauge_instrument(model,cfg:V1Config,seed:int,condition_numbers=(1.0,3.0,10.0)):
    base=FrozenV1Cell.from_torch(model); ep=generate_episode(seed*100003+701,cfg); n=min(cfg.gauge_steps,ep.go_index+1)
    bstates,bouts=_trajectory(base,ep,n); symbols=np.asarray([_event_symbol(ep,t) for t in range(n)],dtype=int)
    bJ=np.stack([_jac(lambda z,t=t: base.step(z,ep.cue[t],ep.content[t],ep.go[t],ep.clock[t]),bstates[t],cfg.probe_epsilon) for t in range(n)])
    bf=response_invariants(bJ); result={"steps":n,"conditions":{}}
    for ci,cond in enumerate(condition_numbers):
        G=_gauge(seed*1009+ci+3,cfg.hidden_size,float(cond)); twin=GaugeWrappedV1(base,G)
        tstates,touts=_trajectory(twin,ep,n,G=G)
        tJ=np.stack([_jac(lambda z,t=t: twin.step(z,ep.cue[t],ep.content[t],ep.go[t],ep.clock[t]),tstates[t],cfg.probe_epsilon) for t in range(n)])
        tf=response_invariants(tJ)
        denom=np.linalg.norm(bf,axis=1)+1e-12; drift=np.linalg.norm(tf-bf,axis=1)/denom
        bm,bl,bt=_grammar(bf,symbols,seed+11); tm,tl,tt=_grammar(tf,symbols,seed+11)
        tv=None
        if bm.k==tm.k:
            aligned,mapping=align_labels(bl,tl); rem=remap_transition_probabilities(tt.probabilities,mapping)
            tv=float(np.max(0.5*np.sum(np.abs(bt.probabilities-rem),axis=2)))
        result["conditions"][str(float(cond))]={"output_max_abs":float(np.max(np.abs(bouts-touts))),
            "median_invariant_relative_drift":float(np.median(drift)),"transition_tv":tv,"coordinate_count_base":bm.k,"coordinate_count_twin":tm.k}
    return result
