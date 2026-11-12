"""Lorentz (hyperboloid) model of hyperbolic space, curvature -1.

Points are (d+1)-vectors x with <x, x>_L = -1 and x_0 > 0, where
<x, y>_L = -x_0 y_0 + sum_i x_i y_i.
"""
from __future__ import annotations

import torch

EPS = 1e-6


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = (-lorentz_inner(x, y)).clamp_min(1.0 + EPS)
    return torch.acosh(z)


def project_to_hyperboloid(x: torch.Tensor) -> torch.Tensor:
    """Keep spatial coordinates and recompute x_0 = sqrt(1 + |x_{1:}|^2)."""
    spatial = x[..., 1:]
    time = torch.sqrt(1.0 + (spatial * spatial).sum(dim=-1, keepdim=True))
    return torch.cat([time, spatial], dim=-1)


def proj_tan(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Project an ambient vector u onto the tangent space at x."""
    return u + lorentz_inner(x, u).unsqueeze(-1) * x


def egrad_to_rgrad(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Euclidean gradient -> Riemannian gradient on the hyperboloid."""
    g = grad.clone()
    g[..., 0] = -g[..., 0]
    return proj_tan(x, g)


def expmap(x: torch.Tensor, v: torch.Tensor, *, max_step: float | None = None) -> torch.Tensor:
    """Exponential map at x, with optional clipping of the tangent norm."""
    v = proj_tan(x, v)
    v_norm = lorentz_inner(v, v).clamp_min(0.0).sqrt()
    if max_step is not None:
        scale = (max_step / v_norm.clamp_min(EPS)).clamp(max=1.0)
        v = v * scale.unsqueeze(-1)
        v_norm = v_norm * scale
    v_norm = v_norm.unsqueeze(-1).clamp_min(EPS)
    return torch.cosh(v_norm) * x + torch.sinh(v_norm) * (v / v_norm)


def expmap0(u: torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin (1, 0, ..., 0) of a d-dim Euclidean vector."""
    norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(EPS)
    return torch.cat([torch.cosh(norm), torch.sinh(norm) * u / norm], dim=-1)


def lorentz_to_poincare(x: torch.Tensor) -> torch.Tensor:
    """Map hyperboloid points into the Poincare ball (for plotting / radius)."""
    return x[..., 1:] / (x[..., :1] + 1.0).clamp_min(EPS)
