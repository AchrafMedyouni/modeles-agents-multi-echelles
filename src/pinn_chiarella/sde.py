"""Drift, diffusion et simulateur Euler-Maruyama du modèle de Chiarella (données synthétiques).

État X = (δ, M), avec δ = P − V et V constant (g = 0, σ_V = 0) :

    dδ = (f(δ) + β tanh(γM)) dt + σ dW
    dM = α (f(δ) + β tanh(γM) − M) dt + α σ dW

où f(δ) = −κδ − κ₃δ³ est la demande des fondamentalistes. Le même dW entre dans les deux
équations : le bruit est parfaitement corrélé, de coefficients (σ, ασ).

Les équations viennent du rapport du groupe : ne pas les modifier sans demander.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch

Array = np.ndarray | torch.Tensor
Scalar = float | torch.Tensor


@dataclass(frozen=True)
class ChiarellaParams:
    """Paramètres du modèle de Chiarella à V constant.

    Attributs :
        kappa : rappel linéaire des fondamentalistes.
        kappa3 : rappel cubique des fondamentalistes (0 pour le modèle simple).
        beta : intensité des suiveurs de tendance.
        sigma : volatilité du bruit.
        alpha : inverse de l'horizon des suiveurs de tendance.
        gamma : sensibilité au signal de tendance.
    """

    kappa: float
    kappa3: float
    beta: float
    sigma: float
    alpha: float
    gamma: float

    def as_dict(self) -> dict[str, float]:
        """Renvoie les paramètres sous forme de dictionnaire."""
        return asdict(self)


@dataclass(frozen=True)
class InitialState:
    """Loi initiale de l'état : δ₀ et M₀ gaussiens indépendants."""

    delta_mean: float = 1.0
    delta_std: float = 0.3
    m_mean: float = 0.0
    m_std: float = 0.3


def _tanh(x: Array) -> Array:
    """Tangente hyperbolique, pour un tableau NumPy ou un tenseur PyTorch."""
    return torch.tanh(x) if isinstance(x, torch.Tensor) else np.tanh(x)


def fundamental_demand(delta: Array, kappa: Scalar, kappa3: Scalar) -> Array:
    """Demande des fondamentalistes f(δ) = −κδ − κ₃δ³."""
    return -kappa * delta - kappa3 * delta**3


def drift(
    delta: Array,
    m: Array,
    *,
    kappa: Scalar,
    kappa3: Scalar,
    beta: Scalar,
    alpha: Scalar,
    gamma: Scalar,
) -> tuple[Array, Array]:
    """Drift de l'état (δ, M).

    Fonctionne aussi bien avec des tableaux NumPy (simulation) qu'avec des tenseurs PyTorch
    (résidu de Fokker-Planck, paramètres entraînables) : c'est la seule définition du drift.

    Renvoie :
        (a_δ, a_M) avec a_δ = f(δ) + β tanh(γM) et a_M = α (a_δ − M).
    """
    a_delta = fundamental_demand(delta, kappa, kappa3) + beta * _tanh(gamma * m)
    a_m = alpha * (a_delta - m)
    return a_delta, a_m


def diffusion(*, sigma: Scalar, alpha: Scalar) -> tuple[Scalar, Scalar]:
    """Coefficients de diffusion (b_δ, b_M) = (σ, ασ) devant le même dW."""
    return sigma, alpha * sigma


def is_stable(params: ChiarellaParams) -> bool:
    """Critère de stabilité linéaire du point fixe (δ, M) = (0, 0) : α + κ − αβγ > 0."""
    return params.alpha + params.kappa - params.alpha * params.beta * params.gamma > 0


def sample_initial_state(n: int, init: InitialState, rng: np.random.Generator) -> np.ndarray:
    """Tire n états initiaux. Renvoie un tableau (n, 2) de colonnes (δ₀, M₀)."""
    delta0 = rng.normal(init.delta_mean, init.delta_std, n)
    m0 = rng.normal(init.m_mean, init.m_std, n)
    return np.stack([delta0, m0], axis=1)


def observation_steps(t_obs: Sequence[float], dt: float) -> np.ndarray:
    """Convertit des instants d'observation en indices de pas de temps.

    Les instants doivent être positifs, strictement croissants et multiples de dt.
    """
    t = np.asarray(t_obs, dtype=float)
    steps = np.rint(t / dt).astype(int)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t_obs doit être une liste non vide d'instants.")
    if np.any(t < 0) or np.any(np.diff(steps) <= 0):
        raise ValueError("t_obs doit être positif et strictement croissant.")
    if not np.allclose(steps * dt, t, rtol=0.0, atol=1e-9):
        raise ValueError("Les instants de t_obs doivent être des multiples de dt.")
    return steps


def simulate(
    params: ChiarellaParams,
    x0: np.ndarray,
    t_obs: Sequence[float],
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simule l'état (δ, M) par Euler-Maruyama et renvoie les nuages aux instants demandés.

    Arguments :
        params : paramètres du modèle.
        x0 : états initiaux, tableau (n, 2) de colonnes (δ₀, M₀). Non modifié.
        t_obs : instants d'observation, croissants et multiples de dt.
        dt : pas de temps.
        rng : générateur aléatoire (fixe la reproductibilité).

    Renvoie :
        Un tableau (len(t_obs), n, 2) : l'état des n particules à chaque instant de t_obs.
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 2 or x0.shape[1] != 2:
        raise ValueError("x0 doit être de dimension (n, 2).")
    steps = observation_steps(t_obs, dt)
    n = x0.shape[0]
    b_delta, b_m = diffusion(sigma=params.sigma, alpha=params.alpha)

    delta, m = x0[:, 0].copy(), x0[:, 1].copy()
    snapshots = np.empty((len(steps), n, 2))
    k = 0
    for s in range(steps[-1] + 1):
        if s == steps[k]:
            snapshots[k, :, 0], snapshots[k, :, 1] = delta, m
            k += 1
            if k == len(steps):
                break
        dw = rng.normal(0.0, np.sqrt(dt), n)
        a_delta, a_m = drift(
            delta,
            m,
            kappa=params.kappa,
            kappa3=params.kappa3,
            beta=params.beta,
            alpha=params.alpha,
            gamma=params.gamma,
        )
        delta, m = delta + a_delta * dt + b_delta * dw, m + a_m * dt + b_m * dw
    return snapshots
