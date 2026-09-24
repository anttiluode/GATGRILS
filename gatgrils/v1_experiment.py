from __future__ import annotations
import argparse, gzip, json, platform
from pathlib import Path
import numpy as np
import torch
from .v1_config import V1Config, default_v1_config, v1_config_to_dict
from .v1_data import generate_dataset
from .v1_train import train_v1, evaluate_task
from .v1_interventions import score_specificity_matrix, publication_vs_admission_clamp, rhythmic_admission_controls
from .v1_instrument import silent_ping_tomography, temporal_context_curve
from .v1_gauge import run_v1_gauge_instrument


def run_seed(seed:int,cfg:V1Config)->dict:
    model,training=train_v1(seed,cfg)
    test_eps=generate_dataset(seed*100003+31,cfg.test_episodes,cfg)
    task=evaluate_task(model,test_eps,cfg)
    specificity=score_specificity_matrix(model,cfg,seed)
    clamps=publication_vs_admission_clamp(model,cfg,seed)
    rhythm=rhythmic_admission_controls(model,cfg,seed)
    ping=silent_ping_tomography(model,cfg,seed)
    window={str(scale):temporal_context_curve(model,cfg,time_scale=scale) for scale in (0.5,1.0,2.0)}
    gauge=run_v1_gauge_instrument(model,cfg,seed)
    return {'seed':int(seed),'training':{**training,'parameter_count':int(sum(p.numel() for p in model.parameters()))},
            'task':task,'specificity':specificity,'clamps':clamps,'rhythm':rhythm,
            'ping':ping,'window':window,'gauge':gauge}


def _median(values):
    return None if not values else float(np.median(np.asarray(values,dtype=float)))


def aggregate_receipts(seeds:list[dict],cfg:V1Config)->dict:
    valid=[r for r in seeds if r['task']['selected_route_sign_accuracy']>=cfg.task_accuracy_threshold and r['task']['silent_mae']<=cfg.silence_mae_threshold]
    count=len(valid)
    surface_medians=[]; margins=[]
    for j in range(3):
        meds=[]
        for i in range(3): meds.append(_median([r['specificity']['matrix'][i][j] for r in valid]))
        own=meds[j]; others=[m for i,m in enumerate(meds) if i!=j and m is not None]
        surface_medians.append(own)
        margins.append(None if own is None or not others else float(own-max(others)))
    pub_lat=_median([r['clamps']['publication']['latent_correct_before_release'] for r in valid])
    pub_rel=_median([r['clamps']['publication']['first_release_correct'] for r in valid])
    adm_damage=_median([r['clamps']['latent_accuracy_damage'] for r in valid])
    rhythm_damage=_median([r['rhythm']['cycle_mean_damage'] for r in valid])
    gates={
        'task_valid_count':{'value':count,'threshold':9,'pass':count>=9},
        'own_surface_transplants':{'values':surface_medians,'threshold':cfg.transplant_threshold,
            'pass':all(v is not None and v>=cfg.transplant_threshold for v in surface_medians)},
        'specificity_margin':{'values':margins,'threshold':cfg.specificity_margin,
            'pass':all(v is not None and v>=cfg.specificity_margin for v in margins)},
        'publication_state':{'latent':pub_lat,'release':pub_rel,'latent_threshold':cfg.publication_latent_threshold,
            'release_threshold':cfg.publication_release_threshold,
            'pass':pub_lat is not None and pub_rel is not None and pub_lat>=cfg.publication_latent_threshold and pub_rel>=cfg.publication_release_threshold},
        'admission_damage':{'value':adm_damage,'threshold':cfg.admission_damage_margin,
            'pass':adm_damage is not None and adm_damage>=cfg.admission_damage_margin},
        'rhythmic_admission_damage':{'value':rhythm_damage,'threshold':cfg.rhythmic_damage_margin,
            'pass':rhythm_damage is not None and rhythm_damage>=cfg.rhythmic_damage_margin},
    }
    return {'overall_pass':bool(all(g['pass'] for g in gates.values())),'task_valid_count':count,
            'own_surface_medians':surface_medians,'specificity_margins':margins,
            'publication_latent_median':pub_lat,'publication_release_median':pub_rel,
            'admission_damage_median':adm_damage,'rhythmic_damage_median':rhythm_damage,'gates':gates}


def run_panel(cfg:V1Config,seeds:tuple[int,...])->dict:
    rows=[run_seed(int(seed),cfg) for seed in seeds]
    return {'schema':'gatgrils-v1-temporal-control-v1','config':v1_config_to_dict(cfg),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
            'seeds':rows,'aggregate':aggregate_receipts(rows,cfg)}


def _canonical_json(obj): return json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n'

def _load_expected_receipt(path):
    p=Path(path); obj=json.loads(p.read_text())
    if obj.get('schema')!='gatgrils-v1-canonical-index-v1': return obj
    ref=Path(obj['full_receipt_gzip'])
    candidates=(ref, p.parent/ref.name, p.parent/ref)
    gz=next((q for q in candidates if q.exists()), None)
    if gz is None: raise FileNotFoundError(f'full receipt gzip not found: {ref}')
    with gzip.open(gz,'rt') as f: return json.load(f)

def write_receipt(receipt,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(_canonical_json(receipt))

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output',default='results/gatgrils_v1.json'); p.add_argument('--development',action='store_true'); p.add_argument('--check-receipt')
    args=p.parse_args(argv); cfg=default_v1_config(); seeds=cfg.development_seeds if args.development else cfg.canonical_seeds
    receipt=run_panel(cfg,seeds)
    if args.check_receipt:
        expected=_load_expected_receipt(args.check_receipt)
        if _canonical_json(receipt)!=_canonical_json(expected): raise SystemExit('receipt mismatch')
        print('receipt check: PASS'); return 0
    write_receipt(receipt,args.output); print(json.dumps(receipt['aggregate'],indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
