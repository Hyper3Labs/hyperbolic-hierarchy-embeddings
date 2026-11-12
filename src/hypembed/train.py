"""Shallow (lookup-table) embeddings of a graph, Lorentz or Euclidean.

Same loss and sampler for both geometries, so the only difference is the
geometry (Nickel & Kiela 2017/2018 protocol):

  loss = cross_entropy over [-d(u, v_pos), -d(u, v_neg_1), ..., -d(u, v_neg_K)]

Updates are sparse: each step touches only the rows in the batch, so the
table can be much larger than a batch (the starting point for web-scale graphs).
Lorentz rows are updated with Riemannian SGD via the exponential map.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .lorentz import egrad_to_rgrad, expmap, expmap0, lorentz_distance, project_to_hyperboloid


@dataclass
class TrainConfig:
    geometry: str = "lorentz"  # "lorentz" | "euclidean"
    dim: int = 10
    epochs: int = 300
    batch_size: int = 128  # more update steps per epoch; 1024 underfits at 300 epochs
    n_neg: int = 50
    lr: float = 0.1
    burnin_epochs: int = 10  # first epochs run at lr * burnin_factor (Nickel & Kiela)
    burnin_factor: float = 0.1
    max_step: float = 1.0  # per-row step-size clip (tangent norm for Lorentz)
    filter_negatives: bool = True  # resample negatives that are known positives
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64  # Lorentz coordinates lose precision in float32 far from the origin
    max_radius: float = 15.0  # cap on hyperbolic distance from the origin (keeps x_0 <= cosh(15))
    log_every: int = 10


def init_embeddings(n_nodes: int, cfg: TrainConfig, g: torch.Generator) -> torch.Tensor:
    u0 = (torch.randn((n_nodes, cfg.dim), generator=g) * 1e-3).to(cfg.dtype)
    if cfg.geometry == "lorentz":
        return project_to_hyperboloid(expmap0(u0)).to(cfg.device)
    return u0.to(cfg.device)


def clip_radius(x: torch.Tensor, max_radius: float) -> torch.Tensor:
    """Pull Lorentz points back so their distance from the origin is <= max_radius."""
    spatial = x[..., 1:]
    norm = spatial.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    limit = torch.sinh(torch.tensor(max_radius, dtype=x.dtype, device=x.device))
    spatial = spatial * (limit / norm).clamp(max=1.0)
    return project_to_hyperboloid(torch.cat([x[..., :1], spatial], dim=-1))


def distance(geometry: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if geometry == "lorentz":
        return lorentz_distance(a, b)
    return (a - b).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()


def _edge_keys(edges: torch.Tensor, n_nodes: int) -> torch.Tensor:
    return torch.sort(edges[:, 0] * n_nodes + edges[:, 1]).values


def sample_negatives(
    anchors: torch.Tensor,
    n_nodes: int,
    n_neg: int,
    g: torch.Generator,
    positive_keys: torch.Tensor | None,
    tries: int = 3,
) -> torch.Tensor:
    """Uniform node negatives; optionally resample those that are known positives."""
    neg = torch.randint(0, n_nodes, (anchors.shape[0], n_neg), generator=g, device=anchors.device)
    if positive_keys is None:
        return neg
    for _ in range(tries):
        keys = anchors.unsqueeze(1) * n_nodes + neg
        pos = torch.searchsorted(positive_keys, keys).clamp_max(positive_keys.numel() - 1)
        bad = (positive_keys[pos] == keys) | (neg == anchors.unsqueeze(1))
        if not bad.any():
            break
        neg[bad] = torch.randint(0, n_nodes, (int(bad.sum()),), generator=g, device=anchors.device)
    return neg


def train(edges: torch.Tensor, n_nodes: int, cfg: TrainConfig) -> torch.Tensor:
    """Train embeddings on directed (u, v) pairs; v is the positive for anchor u.

    Pass both directions if you want the graph treated as undirected.
    """
    g = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    g_cpu = torch.Generator().manual_seed(cfg.seed)
    edges = edges.to(cfg.device)
    emb = init_embeddings(n_nodes, cfg, g_cpu)
    pos_keys = _edge_keys(edges, n_nodes) if cfg.filter_negatives else None

    for epoch in range(cfg.epochs):
        lr = cfg.lr * (cfg.burnin_factor if epoch < cfg.burnin_epochs else 1.0)
        perm = torch.randperm(edges.shape[0], generator=g, device=cfg.device)
        total, seen = 0.0, 0
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = edges[perm[start : start + cfg.batch_size]]
            u, v = batch[:, 0], batch[:, 1]
            neg = sample_negatives(u, n_nodes, cfg.n_neg, g, pos_keys)
            cand = torch.cat([v.unsqueeze(1), neg], dim=1)  # (B, 1 + K)

            # gather only the rows this batch touches
            idx = torch.cat([u, cand.reshape(-1)])
            uniq, inv = torch.unique(idx, return_inverse=True)
            rows = emb[uniq].clone().requires_grad_(True)
            u_e = rows[inv[: u.numel()]].unsqueeze(1)
            c_e = rows[inv[u.numel() :]].view(cand.shape[0], cand.shape[1], -1)

            logits = -distance(cfg.geometry, u_e, c_e)
            target = torch.zeros(u.numel(), dtype=torch.long, device=cfg.device)
            # sum, not mean: each row's step size should not shrink with batch size
            loss = F.cross_entropy(logits, target, reduction="sum")
            loss.backward()

            with torch.no_grad():
                grad = rows.grad
                if cfg.geometry == "lorentz":
                    new = expmap(rows, -lr * egrad_to_rgrad(rows, grad), max_step=cfg.max_step)
                    emb[uniq] = clip_radius(project_to_hyperboloid(new), cfg.max_radius)
                else:
                    step = lr * grad
                    norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    emb[uniq] = rows - step * (cfg.max_step / norm).clamp(max=1.0)
            total += loss.item()
            seen += u.numel()

        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"  [{cfg.geometry} d={cfg.dim}] epoch {epoch:4d}  loss {total / max(1, seen):.4f}")

    return emb
