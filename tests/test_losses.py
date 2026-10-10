"""Tests des losses : justesse de l'opérateur de Fokker-Planck, décroissance de la loss."""

import numpy as np
import pytest
import torch

from pinn_chiarella.data import generate_synthetic, save_synthetic
from pinn_chiarella.losses import fokker_planck_residual
from pinn_chiarella.sde import ChiarellaParams, drift, simulate
from pinn_chiarella.train import train

KAPPA, SIGMA, ALPHA = 1.0, 0.5, 1.5
MEAN0 = torch.tensor([1.0, 0.2], dtype=torch.float64)
COV0 = torch.tensor([[0.09, 0.01], [0.01, 0.16]], dtype=torch.float64)


def ou_moments(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Moyenne (n, 2) et covariance (n, 2, 2) exactes de l'état (δ, M) pour β = 0, f = −κδ.

    Le système est alors linéaire, dX = A X dt + B dW, donc gaussien :
    moyenne e^{At} m₀, covariance e^{At} Σ₀ e^{Aᵀt} + ∫₀ᵗ e^{As} B Bᵀ e^{Aᵀs} ds
    (intégrale calculée par l'exponentielle de matrice par blocs de Van Loan).
    """
    a = torch.tensor([[-KAPPA, 0.0], [-ALPHA * KAPPA, -ALPHA]], dtype=torch.float64)
    b = torch.tensor([[SIGMA], [ALPHA * SIGMA]], dtype=torch.float64)
    block = torch.zeros(4, 4, dtype=torch.float64)
    block[:2, :2], block[:2, 2:], block[2:, 2:] = -a, b @ b.T, a.T
    expm = torch.matrix_exp(block.unsqueeze(0) * t.reshape(-1, 1, 1))
    flow = expm[:, 2:, 2:].transpose(1, 2)
    cov = flow @ COV0 @ flow.transpose(1, 2) + flow @ expm[:, :2, 2:]
    return flow @ MEAN0, cov


def ou_density(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Densité gaussienne exacte de l'état aux points x (n, 2) et instants t (n, 1)."""
    mean, cov = ou_moments(t)
    d0, d1 = (x - mean).unbind(dim=1)
    c00, c01, c11 = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = c00 * c11 - c01**2
    quad = (c11 * d0**2 - 2 * c01 * d0 * d1 + c00 * d1**2) / det
    return (torch.exp(-0.5 * quad) / (2 * torch.pi * torch.sqrt(det))).unsqueeze(1)


def ou_drift(delta, m):
    return drift(delta, m, kappa=KAPPA, kappa3=0.0, beta=0.0, alpha=ALPHA, gamma=2.0)


def collocation(n: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(n, 2, dtype=torch.float64, generator=gen) * 3.0 - 1.5
    t = torch.rand(n, 1, dtype=torch.float64, generator=gen) * 2.9 + 0.1
    return x, t


def test_residu_nul_pour_la_densite_ornstein_uhlenbeck():
    x, t = collocation()
    residual = fokker_planck_residual(ou_density, ou_drift, SIGMA, ALPHA, x, t)
    assert residual.shape == (len(x), 1)
    assert ou_density(x, t).max() > 0.1  # la densité n'est pas négligeable sur les points testés
    assert residual.abs().max() < 1e-10


@pytest.mark.parametrize(
    ("sigma", "alpha"), [(1.2 * SIGMA, ALPHA), (SIGMA, 0.5 * ALPHA), (SIGMA, -ALPHA)]
)
def test_residu_non_nul_si_la_diffusion_est_fausse(sigma, alpha):
    # Contrôle négatif : un mauvais σ, ou un mauvais coefficient sur les dérivées croisées
    # et en M (α), doit se voir dans le résidu.
    x, t = collocation()
    residual = fokker_planck_residual(ou_density, ou_drift, sigma, alpha, x, t)
    assert residual.abs().max() > 1e-2


def test_densite_analytique_coherente_avec_le_simulateur():
    # La densité de référence décrit bien le processus simulé par sde.py.
    params = ChiarellaParams(kappa=KAPPA, kappa3=0.0, beta=0.0, sigma=SIGMA, alpha=ALPHA, gamma=2.0)
    rng = np.random.default_rng(0)
    x0 = rng.multivariate_normal(MEAN0.numpy(), COV0.numpy(), size=40_000)
    x = simulate(params, x0, [1.0], 2e-3, rng)[0]
    mean, cov = ou_moments(torch.tensor([1.0], dtype=torch.float64))
    assert np.allclose(x.mean(axis=0), mean[0].numpy(), atol=0.01)
    assert np.allclose(np.cov(x.T), cov[0].numpy(), atol=0.01)


def test_200_iterations_font_decroitre_la_loss(tmp_path):
    synth = {
        "seed": 0,
        "model": dict(kappa=1.0, kappa3=0.0, beta=0.5, sigma=0.5, alpha=1.0, gamma=2.0),
        "simulation": dict(n_samples=500, n_test=10, dt=5e-3, t_obs=[0.0, 0.5, 1.0], t_test=0.75),
    }
    data_path = tmp_path / "synth.npz"
    save_synthetic(generate_synthetic(synth), data_path)
    config = {
        "seed": 0,
        "device": "cpu",
        "model": {
            "fundamental": "linear",
            "trainable": dict(kappa=0.3, beta=0.1, sigma=1.0),
            "fixed": dict(kappa3=0.0, alpha=1.0, gamma=2.0),
        },
        "training": dict(
            iterations=200, lr=1e-3, tau=10.0, n_collocation=500, n_integral=500, log_every=10
        ),
    }
    result = train(config, data_path, tmp_path / "run", verbose=False)

    losses = [r["loss"] for r in result.log["history"]]
    assert np.all(np.isfinite(losses))
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    assert result.log["final"] != result.log["initial"]
    masses = np.array([r["mass"] for r in result.log["history"]])
    assert masses.shape == (len(losses), 3)
    assert np.all(masses > 0)
    # La masse de p̃ aux instants observés se rapproche de 1 pendant l'entraînement.
    assert np.abs(masses[-1] - 1).max() < np.abs(masses[0] - 1).max()
    for name in ("config.yaml", "log.json", "model.pt"):
        assert (tmp_path / "run" / name).exists()
