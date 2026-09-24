from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .v1_config import V1Config
from .v1_data import Episode, generate_dataset
from .v1_model import ThreeSurfaceCell


@dataclass
class EpisodeTensors:
    cue: torch.Tensor; content: torch.Tensor; go: torch.Tensor; clock: torch.Tensor
    target: torch.Tensor; mask: torch.Tensor


def stack_episodes(episodes: list[Episode]) -> EpisodeTensors:
    def t(attr, dtype=torch.float64):
        return torch.as_tensor(np.stack([getattr(e, attr) for e in episodes]), dtype=dtype)
    return EpisodeTensors(t('cue'), t('content'), t('go'), t('clock'), t('target'), t('loss_mask'))


def task_loss(model: ThreeSurfaceCell, batch: EpisodeTensors, cfg: V1Config) -> torch.Tensor:
    tr = model(batch.cue, batch.content, batch.go, batch.clock)
    weight = batch.mask * (1.0 + cfg.go_loss_weight * batch.go)
    sq = (tr.emitted - batch.target).square().mean(dim=-1, keepdim=True)
    return (sq * weight).sum() / weight.sum().clamp_min(1.0)


def _subset(batch: EpisodeTensors, idx: np.ndarray) -> EpisodeTensors:
    ti = torch.as_tensor(idx, dtype=torch.long)
    return EpisodeTensors(*(getattr(batch, k)[ti] for k in ('cue','content','go','clock','target','mask')))


def train_v1(seed: int, cfg: V1Config, epochs: int | None = None,
             episode_count: int | None = None) -> tuple[ThreeSurfaceCell, dict[str, float | int]]:
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    model = ThreeSurfaceCell(cfg).double()
    n = int(episode_count or cfg.train_episodes)
    episodes = generate_dataset(seed * 100003 + 17, n, cfg)
    batch = stack_episodes(episodes)
    with torch.no_grad():
        initial = float(task_loss(model, batch, cfg))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed * 100003 + 23)
    epochs_run = int(epochs if epochs is not None else cfg.training_epochs)
    steps = 0
    model.train()
    for _ in range(epochs_run):
        order = rng.permutation(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            b = _subset(batch, idx)
            opt.zero_grad(set_to_none=True)
            loss = task_loss(model, b, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            opt.step(); steps += 1
    model.eval()
    with torch.no_grad():
        final = float(task_loss(model, batch, cfg))
    receipt = {'seed': int(seed), 'initial_loss': initial, 'final_loss': final,
               'epochs': epochs_run, 'optimizer_steps': int(steps), 'episodes': n}
    return model, receipt


def evaluate_task(model: ThreeSurfaceCell, episodes: list[Episode], cfg: V1Config) -> dict[str, float]:
    batch = stack_episodes(episodes)
    with torch.no_grad():
        tr = model(batch.cue, batch.content, batch.go, batch.clock)
    outs = tr.emitted.detach().cpu().numpy()
    correct=[]; silent=[]; nonselected=[]
    for i,e in enumerate(episodes):
        y = outs[i, e.go_index]
        correct.append((1 if y[e.q] >= 0 else -1) == e.target_sign)
        nonselected.append(abs(float(y[1-e.q])))
        valid = e.loss_mask[:,0].copy(); valid[e.go_index] = False
        silent.extend(np.abs(outs[i, valid]).ravel().tolist())
    return {'selected_route_sign_accuracy': float(np.mean(correct)),
            'silent_mae': float(np.mean(silent)),
            'go_nonselected_mae': float(np.mean(nonselected))}
