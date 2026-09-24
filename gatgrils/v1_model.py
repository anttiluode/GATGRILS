from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .v1_config import V1Config


@dataclass
class ControlOverride:
    g_op: torch.Tensor | None = None
    admission: torch.Tensor | None = None
    publication: torch.Tensor | None = None
    g_op_mask: torch.Tensor | None = None
    admission_mask: torch.Tensor | None = None
    publication_mask: torch.Tensor | None = None


@dataclass
class StepTrace:
    hidden: torch.Tensor
    g_op: torch.Tensor
    admission: torch.Tensor
    latent: torch.Tensor
    publication: torch.Tensor
    emitted: torch.Tensor


@dataclass
class ForwardTrace:
    hidden: torch.Tensor
    g_op: torch.Tensor
    admission: torch.Tensor
    latent: torch.Tensor
    publication: torch.Tensor
    emitted: torch.Tensor


def _override_at(value: torch.Tensor | None, t: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim >= 3:
        return value[:, t]
    return value


class ThreeSurfaceCell(nn.Module):
    def __init__(self, cfg: V1Config):
        super().__init__()
        self.cfg = cfg
        h, r = cfg.hidden_size, cfg.operator_rank
        self.recurrent = nn.Linear(h, h)
        self.cue_proj = nn.Linear(8, h, bias=False)
        self.content_weight = nn.Parameter(torch.empty(h))
        self.go_proj = nn.Linear(1, h, bias=False)
        self.op_gate = nn.Linear(h, r)
        self.op_a = nn.Linear(h, r, bias=False)
        self.op_b = nn.Linear(r, h, bias=False)
        self.adm_h = nn.Linear(h, 1, bias=False)
        self.adm_phase_base = nn.Parameter(torch.zeros(2))
        self.adm_phase_h = nn.Linear(h, 2, bias=False)
        self.adm_bias = nn.Parameter(torch.zeros(1))
        self.latent_head = nn.Linear(h, 2)
        self.pub_h = nn.Linear(h, 2)
        self.pub_go = nn.Parameter(torch.zeros(2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.recurrent.weight)
            self.recurrent.weight.mul_(0.78)
            self.recurrent.bias.zero_()
            nn.init.xavier_uniform_(self.cue_proj.weight, gain=0.7)
            nn.init.normal_(self.content_weight, mean=0.0, std=0.25)
            nn.init.xavier_uniform_(self.go_proj.weight, gain=0.5)
            for layer in (self.op_gate, self.op_a, self.op_b, self.adm_h, self.adm_phase_h,
                          self.latent_head, self.pub_h):
                if hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if getattr(layer, 'bias', None) is not None:
                    layer.bias.zero_()
            self.adm_phase_base.zero_()
            self.adm_bias.zero_()
            self.pub_go.zero_()

    def step_from_hidden(self, hidden: torch.Tensor, cue: torch.Tensor, content: torch.Tensor,
                         go: torch.Tensor, clock: torch.Tensor,
                         override: ControlOverride | None = None) -> StepTrace:
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
            cue = cue.unsqueeze(0) if cue.ndim == 1 else cue
            content = content.unsqueeze(0) if content.ndim == 1 else content
            go = go.unsqueeze(0) if go.ndim == 1 else go
            clock = clock.unsqueeze(0) if clock.ndim == 1 else clock
            squeeze = True
        else:
            squeeze = False
        g_op = torch.tanh(self.op_gate(hidden))
        if override is not None and override.g_op is not None:
            g_op = override.g_op if override.g_op_mask is None else torch.where(override.g_op_mask, override.g_op, g_op)
        operator_delta = self.op_b(g_op * self.op_a(hidden))
        phase_vector = self.adm_phase_base + self.adm_phase_h(hidden)
        admission = torch.sigmoid(self.adm_bias + self.adm_h(hidden) +
                                  (clock * phase_vector).sum(dim=-1, keepdim=True))
        if override is not None and override.admission is not None:
            admission = override.admission if override.admission_mask is None else torch.where(override.admission_mask, override.admission, admission)
        x_eff = admission * content
        pre = (self.recurrent(hidden) + self.cue_proj(cue) +
               x_eff * self.content_weight.unsqueeze(0) + self.go_proj(go) + operator_delta)
        hidden_next = torch.tanh(pre)
        latent = torch.tanh(self.latent_head(hidden_next))
        publication = torch.sigmoid(self.pub_h(hidden_next) + go * self.pub_go.unsqueeze(0))
        if override is not None and override.publication is not None:
            publication = override.publication if override.publication_mask is None else torch.where(override.publication_mask, override.publication, publication)
        emitted = publication * latent
        if squeeze:
            return StepTrace(*(x.squeeze(0) for x in (hidden_next, g_op, admission, latent, publication, emitted)))
        return StepTrace(hidden_next, g_op, admission, latent, publication, emitted)

    def forward(self, cue: torch.Tensor, content: torch.Tensor, go: torch.Tensor,
                clock: torch.Tensor, override: ControlOverride | None = None,
                initial_hidden: torch.Tensor | None = None) -> ForwardTrace:
        if cue.ndim != 3 or content.ndim != 3 or go.ndim != 3 or clock.ndim != 3:
            raise ValueError('forward inputs must have shape [B,T,D]')
        B, T, _ = cue.shape
        hidden = (torch.zeros(B, self.cfg.hidden_size, dtype=cue.dtype, device=cue.device)
                  if initial_hidden is None else initial_hidden)
        hs=[]; gs=[]; ads=[]; lats=[]; pubs=[]; outs=[]
        for t in range(T):
            o = None
            if override is not None:
                o = ControlOverride(g_op=_override_at(override.g_op, t),
                                    admission=_override_at(override.admission, t),
                                    publication=_override_at(override.publication, t),
                                    g_op_mask=_override_at(override.g_op_mask, t),
                                    admission_mask=_override_at(override.admission_mask, t),
                                    publication_mask=_override_at(override.publication_mask, t))
            st = self.step_from_hidden(hidden, cue[:, t], content[:, t], go[:, t], clock[:, t], o)
            hidden = st.hidden
            hs.append(st.hidden); gs.append(st.g_op); ads.append(st.admission)
            lats.append(st.latent); pubs.append(st.publication); outs.append(st.emitted)
        stack=lambda xs: torch.stack(xs, dim=1)
        return ForwardTrace(stack(hs), stack(gs), stack(ads), stack(lats), stack(pubs), stack(outs))
