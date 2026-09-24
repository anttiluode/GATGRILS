from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from .v1_config import V1Config
from .v1_data import Episode, generate_episode, generate_dataset
from .v1_model import ControlOverride, ThreeSurfaceCell

SURFACES = ("operator", "admission", "publication")
FACTORS = ("operator", "phase", "route")


def expected_vector(ep: Episode) -> tuple[float, float]:
    out = [0.0, 0.0]
    out[ep.q] = float(ep.target_sign)
    return tuple(out)


def is_informative_pair(a: Episode, b: Episode) -> bool:
    return expected_vector(a) != expected_vector(b)


def _episode_inputs(ep: Episode):
    return tuple(torch.as_tensor(x[None], dtype=torch.float64)
                 for x in (ep.cue, ep.content, ep.go, ep.clock))


def _batch_override(override: ControlOverride | None) -> ControlOverride | None:
    if override is None:
        return None
    def b(x):
        if x is None: return None
        return x.unsqueeze(0) if x.ndim in (1, 2) else x
    return ControlOverride(g_op=b(override.g_op), admission=b(override.admission),
                           publication=b(override.publication), g_op_mask=b(override.g_op_mask),
                           admission_mask=b(override.admission_mask), publication_mask=b(override.publication_mask))


def run_episode(model: ThreeSurfaceCell, ep: Episode,
                override: ControlOverride | None = None):
    cue, content, go, clock = _episode_inputs(ep)
    with torch.no_grad():
        tr = model(cue, content, go, clock, override=_batch_override(override))
    return SimpleNamespace(**{k: getattr(tr, k).squeeze(0) for k in
                              ("hidden", "g_op", "admission", "latent", "publication", "emitted")})


def make_matched_pair(seed: int, cfg: V1Config, factor: str) -> tuple[Episode, Episode]:
    if factor not in FACTORS:
        raise ValueError(f"unknown factor {factor}")
    rng = np.random.default_rng(int(seed))
    factors = [int(x) for x in rng.integers(0, 2, size=3)]
    hold = int(rng.choice(cfg.hold_steps))
    streams = (rng.choice([-1, 1], size=cfg.cycles), rng.choice([-1, 1], size=cfg.cycles))
    other = factors.copy()
    other[{"operator": 0, "phase": 1, "route": 2}[factor]] ^= 1
    a = generate_episode(seed, cfg, tuple(factors), hold, streams)
    b = generate_episode(seed, cfg, tuple(other), hold, streams)
    return a, b


def _matches(y: np.ndarray, ep: Episode) -> bool:
    y = np.asarray(y, dtype=float)
    sign_ok = (1 if y[ep.q] >= 0.0 else -1) == ep.target_sign
    route_ok = abs(float(y[ep.q])) >= abs(float(y[1 - ep.q]))
    return bool(sign_ok and route_ok)


def _transplant_override(recipient, donor, surface: str, ep: Episode) -> ControlOverride:
    start = 1 + ep.hold_steps
    if surface == "operator":
        mask = torch.zeros((ep.content.shape[0], 1), dtype=torch.bool); mask[start:ep.go_index] = True
        return ControlOverride(g_op=donor.g_op, g_op_mask=mask)
    if surface == "admission":
        mask = torch.zeros((ep.content.shape[0], 1), dtype=torch.bool); mask[start:ep.go_index] = True
        return ControlOverride(admission=donor.admission, admission_mask=mask)
    if surface == "publication":
        mask = torch.zeros((ep.content.shape[0], 1), dtype=torch.bool); mask[ep.go_index] = True
        return ControlOverride(publication=donor.publication, publication_mask=mask)
    raise ValueError(surface)


def score_specificity_matrix(model: ThreeSurfaceCell, cfg: V1Config, seed: int,
                             pair_count: int | None = None) -> dict:
    wanted = int(pair_count or cfg.transplant_pairs)
    hits = np.zeros((3, 3), dtype=float)
    counts = np.zeros((3, 3), dtype=int)
    attempts = {f: 0 for f in FACTORS}
    informative = {f: 0 for f in FACTORS}
    for j, factor in enumerate(FACTORS):
        k = 0
        while informative[factor] < wanted and k < wanted * 30:
            a, b = make_matched_pair(seed * 100000 + j * 10000 + k, cfg, factor)
            attempts[factor] += 1; k += 1
            if not is_informative_pair(a, b):
                continue
            informative[factor] += 1
            recipient = run_episode(model, a); donor = run_episode(model, b)
            for i, surface in enumerate(SURFACES):
                ov = _transplant_override(recipient, donor, surface, a)
                changed = run_episode(model, a, ov)
                hits[i, j] += float(_matches(changed.emitted[a.go_index].numpy(), b))
                counts[i, j] += 1
        if informative[factor] < wanted:
            raise RuntimeError(f"could not construct {wanted} informative {factor} pairs")
    matrix = np.divide(hits, counts, out=np.zeros_like(hits), where=counts > 0)
    return {"surfaces": list(SURFACES), "factors": list(FACTORS), "matrix": matrix.tolist(),
            "attempted_pairs": attempts, "informative_pairs": informative}


def _latent_sign_ok(latent: torch.Tensor, ep: Episode) -> bool:
    return (1 if float(latent[ep.q]) >= 0 else -1) == ep.target_sign


def publication_vs_admission_clamp(model: ThreeSurfaceCell, cfg: V1Config, seed: int,
                                   episode_count: int | None = None) -> dict:
    episodes = generate_dataset(seed * 100003 + 401, int(episode_count or cfg.clamp_pairs), cfg)
    pub_lat=[]; pub_release=[]; adm_lat=[]; adm_release=[]; pub_sil=[]
    for ep in episodes:
        base = run_episode(model, ep)
        start = max(1 + ep.hold_steps, ep.go_index - 8)
        p = torch.zeros_like(base.publication); pmask = torch.zeros_like(base.publication, dtype=torch.bool); pmask[start:ep.go_index, ep.q] = True
        pub = run_episode(model, ep, ControlOverride(publication=p, publication_mask=pmask))
        a = torch.zeros_like(base.admission); amask = torch.zeros_like(base.admission, dtype=torch.bool); amask[start:ep.go_index] = True
        adm = run_episode(model, ep, ControlOverride(admission=a, admission_mask=amask))
        pub_lat.append(_latent_sign_ok(pub.latent[ep.go_index - 1], ep))
        adm_lat.append(_latent_sign_ok(adm.latent[ep.go_index - 1], ep))
        pub_release.append(_matches(pub.emitted[ep.go_index].numpy(), ep))
        adm_release.append(_matches(adm.emitted[ep.go_index].numpy(), ep))
        pub_sil.extend(torch.abs(pub.emitted[start:ep.go_index, ep.q]).numpy().tolist())
    return {"publication": {"latent_correct_before_release": float(np.mean(pub_lat)),
                             "first_release_correct": float(np.mean(pub_release)),
                             "selected_output_mae_during_clamp": float(np.mean(pub_sil))},
            "admission": {"latent_correct_before_release": float(np.mean(adm_lat)),
                           "first_release_correct": float(np.mean(adm_release))},
            "latent_accuracy_damage": float(np.mean(pub_lat) - np.mean(adm_lat))}


def _retimed_episode(ep: Episode, cfg: V1Config) -> Episode:
    content = np.zeros_like(ep.content)
    start = 1 + ep.hold_steps
    for k in range(cfg.cycles):
        content[start + 4*k + 2, 0] = ep.stream_a[k]
        content[start + 4*k, 0] = ep.stream_b[k]
    return replace(ep, content=content)


def rhythmic_admission_controls(model: ThreeSurfaceCell, cfg: V1Config, seed: int,
                                episode_count: int | None = None) -> dict:
    episodes = generate_dataset(seed * 100003 + 503, int(episode_count or cfg.clamp_pairs), cfg)
    base_ok=[]; mean_ok=[]; retime_ok=[]
    for ep in episodes:
        base = run_episode(model, ep)
        base_ok.append(_matches(base.emitted[ep.go_index].numpy(), ep))
        a = base.admission.clone(); start = 1 + ep.hold_steps
        amask = torch.zeros_like(a, dtype=torch.bool)
        for k in range(cfg.cycles):
            sl = slice(start + 4*k, start + 4*k + 4)
            a[sl] = a[sl].mean(dim=0, keepdim=True); amask[sl] = True
        mean = run_episode(model, ep, ControlOverride(admission=a, admission_mask=amask))
        mean_ok.append(_matches(mean.emitted[ep.go_index].numpy(), ep))
        rt = _retimed_episode(ep, cfg)
        retimed = run_episode(model, rt)
        retime_ok.append(_matches(retimed.emitted[rt.go_index].numpy(), rt))
    return {"baseline_accuracy": float(np.mean(base_ok)),
            "cycle_mean_accuracy": float(np.mean(mean_ok)),
            "retimed_accuracy": float(np.mean(retime_ok)),
            "cycle_mean_damage": float(np.mean(base_ok) - np.mean(mean_ok)),
            "retiming_damage": float(np.mean(base_ok) - np.mean(retime_ok))}
