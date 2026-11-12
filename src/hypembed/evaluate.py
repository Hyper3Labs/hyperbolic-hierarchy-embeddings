"""Evaluation.

- reconstruction: Nickel & Kiela mean rank / MAP. For each node, rank all
  other nodes by distance and see where its true neighbors land.
- link_prediction: held-out edges scored against uniform random negatives
  (MRR, Hits@k). This is the protocol most graph-embedding releases report.
- depth_radius_spearman: does distance from the origin track hierarchy depth?
"""
from __future__ import annotations

import numpy as np
import torch

from .lorentz import lorentz_to_poincare
from .train import distance


@torch.no_grad()
def reconstruction(
    emb: torch.Tensor,
    edges: torch.Tensor,
    geometry: str,
    *,
    max_nodes: int | None = 2000,
    seed: int = 0,
    chunk: int = 128,
) -> dict[str, float]:
    n = emb.shape[0]
    neigh: dict[int, set[int]] = {}
    for u, v in edges.tolist():
        neigh.setdefault(u, set()).add(v)
    nodes = sorted(neigh)
    if max_nodes is not None and len(nodes) > max_nodes:
        rng = np.random.default_rng(seed)
        nodes = sorted(rng.choice(nodes, size=max_nodes, replace=False).tolist())

    rank_sum, n_ranks, ap_sum = 0.0, 0, 0.0
    for start in range(0, len(nodes), chunk):
        batch = nodes[start : start + chunk]
        d = distance(geometry, emb[batch].unsqueeze(1), emb.unsqueeze(0))  # (b, n)
        d[torch.arange(len(batch)), torch.tensor(batch)] = float("inf")
        order = torch.argsort(d, dim=1).cpu().numpy()
        for row, u in enumerate(batch):
            pos = neigh[u]
            is_pos = np.isin(order[row], list(pos))
            ranks = np.nonzero(is_pos)[0] + 1  # 1-based, ascending
            # N&K mean rank: rank of each positive among non-positives
            rank_sum += float((ranks - np.arange(len(ranks))).sum())
            n_ranks += len(ranks)
            ap_sum += float(np.mean(np.arange(1, len(ranks) + 1) / ranks))
    return {"mean_rank": rank_sum / max(1, n_ranks), "map": ap_sum / max(1, len(nodes))}


@torch.no_grad()
def link_prediction(
    emb: torch.Tensor,
    test_edges: torch.Tensor,
    geometry: str,
    *,
    n_neg: int = 1000,
    seed: int = 0,
    chunk: int = 4096,
) -> dict[str, float]:
    g = torch.Generator(device=emb.device).manual_seed(seed)
    ranks = []
    for start in range(0, test_edges.shape[0], chunk):
        e = test_edges[start : start + chunk].to(emb.device)
        u, v = e[:, 0], e[:, 1]
        neg = torch.randint(0, emb.shape[0], (u.numel(), n_neg), generator=g, device=emb.device)
        d_pos = distance(geometry, emb[u], emb[v])
        d_neg = distance(geometry, emb[u].unsqueeze(1), emb[neg])
        ranks.append(1 + (d_neg < d_pos.unsqueeze(1)).sum(dim=1))
    r = torch.cat(ranks).float()
    return {
        "mrr": float((1.0 / r).mean()),
        "hits@1": float((r <= 1).float().mean()),
        "hits@10": float((r <= 10).float().mean()),
    }


@torch.no_grad()
def depth_radius_spearman(emb: torch.Tensor, depths: list[int], geometry: str) -> float:
    x = lorentz_to_poincare(emb) if geometry == "lorentz" else emb
    radii = torch.linalg.norm(x, dim=-1).cpu().numpy()
    d = np.asarray(depths, dtype=float)
    keep = d >= 0
    ra = np.argsort(np.argsort(radii[keep]))
    da = np.argsort(np.argsort(d[keep], kind="mergesort"))
    return float(np.corrcoef(ra, da)[0, 1])
