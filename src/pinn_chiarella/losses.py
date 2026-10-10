"""Résidu de Fokker-Planck et loss KL variationnelle sur les données.

Équation de Fokker-Planck de l'état (δ, M), de drift (a_δ, a_M) et de bruit corrélé (σ, ασ)
porté par le même dW :

    ∂p/∂t = −∂(a_δ p)/∂δ − ∂(a_M p)/∂M + ½σ² (p_δδ + 2α p_δM + α² p_MM)
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Density = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Drift = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

LOG_EPS = 1e-8


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Dérivée point par point de y par rapport à x (graphe conservé)."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def sample_uniform(lo: torch.Tensor, hi: torch.Tensor, n: int) -> torch.Tensor:
    """Tire n points uniformes dans le pavé [lo, hi]. Renvoie (n, len(lo))."""
    return lo + (hi - lo) * torch.rand(n, lo.numel(), dtype=lo.dtype, device=lo.device)


def fokker_planck_residual(
    density: Density,
    drift: Drift,
    sigma: torch.Tensor | float,
    alpha: torch.Tensor | float,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Résidu de Fokker-Planck p_t + div(a p) − ½σ²(p_δδ + 2α p_δM + α² p_MM).

    Arguments :
        density : fonction (x (n, 2), t (n, 1)) -> p (n, 1).
        drift : fonction (δ (n, 1), M (n, 1)) -> (a_δ, a_M).
        sigma, alpha : coefficients du bruit (σ sur δ, ασ sur M, même dW).
        x : points de collocation (n, 2), colonnes (δ, M).
        t : instants de collocation (n, 1).

    Renvoie :
        Le résidu en chaque point, de dimension (n, 1). Il est nul si p est solution.
    """
    delta = x[:, :1].detach().clone().requires_grad_(True)
    m = x[:, 1:2].detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    p = density(torch.cat([delta, m], dim=1), t)
    a_delta, a_m = drift(delta, m)

    p_t = _grad(p, t)
    divergence = _grad(a_delta * p, delta) + _grad(a_m * p, m)
    p_d, p_m = _grad(p, delta), _grad(p, m)
    p_dd, p_dm, p_mm = _grad(p_d, delta), _grad(p_d, m), _grad(p_m, m)
    diffusion = 0.5 * sigma**2 * (p_dd + 2 * alpha * p_dm + alpha**2 * p_mm)
    return p_t + divergence - diffusion


def pde_loss(
    density: Density,
    drift: Drift,
    sigma: torch.Tensor | float,
    alpha: torch.Tensor | float,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """L_pde : moyenne du carré du résidu de Fokker-Planck sur les points de collocation."""
    return fokker_planck_residual(density, drift, sigma, alpha, x, t).square().mean()


def data_loss(
    density: Density,
    snapshots: torch.Tensor,
    t_obs: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    n_integral: int,
) -> torch.Tensor:
    """L_data : somme sur les instants observés de la KL variationnelle.

    Pour chaque instant t_i : −mean_j log p̃(X_j, t_i) + ∫ p̃(x, t_i) dx, l'intégrale étant
    estimée par Monte-Carlo (volume du domaine × moyenne de p̃ sur des points uniformes).

    Arguments :
        density : fonction (x (n, 2), t (n, 1)) -> p (n, 1).
        snapshots : nuages observés, de dimension (T, N, 2).
        t_obs : instants observés, de dimension (T,).
        lo, hi : bornes (2,) du domaine d'intégration.
        n_integral : nombre de points uniformes par instant pour l'intégrale.
    """
    n_times, n_samples, dim = snapshots.shape
    volume = torch.prod(hi - lo)

    t_data = t_obs.repeat_interleave(n_samples).unsqueeze(1)
    log_p = torch.log(density(snapshots.reshape(-1, dim), t_data) + LOG_EPS)

    x_unif = sample_uniform(lo, hi, n_times * n_integral)
    t_unif = t_obs.repeat_interleave(n_integral).unsqueeze(1)
    mass = volume * density(x_unif, t_unif)

    # Les instants ont tous le même nombre de points : somme des moyennes = T × moyenne.
    return n_times * (mass.mean() - log_p.mean())
