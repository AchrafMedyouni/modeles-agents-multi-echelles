"""Réseaux (densité, MLP du terme manquant) et paramètres physiques entraînables."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from pinn_chiarella.sde import drift_from_demand, fundamental_demand

# Paramètres dont le modèle a besoin, selon la forme de la demande des fondamentalistes.
REQUIRED_PARAMS = {
    "linear": {"kappa", "kappa3", "beta", "sigma", "alpha", "gamma"},
    "nn": {"beta", "sigma", "alpha", "gamma"},
}


def mlp(n_in: int, n_out: int, width: int, depth: int) -> nn.Sequential:
    """MLP à activations tanh : `depth` couches cachées de largeur `width`."""
    layers: list[nn.Module] = []
    n = n_in
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.Tanh()]
        n = width
    layers.append(nn.Linear(n, n_out))
    return nn.Sequential(*layers)


def _to_unit(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Normalisation affine de [lo, hi] vers [−1, 1]."""
    return 2 * (x - lo) / (hi - lo) - 1


class DensityNet(nn.Module):
    """Réseau de densité p̃(δ, M, t) > 0.

    Les entrées sont normalisées dans [−1, 1] à partir du domaine [lo, hi] × [0, t_max] ;
    la sortie passe par un softplus pour rester positive.
    """

    def __init__(
        self, lo: torch.Tensor, hi: torch.Tensor, t_max: float, width: int = 32, depth: int = 4
    ) -> None:
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.t_max = float(t_max)
        self.net = mlp(3, 1, width, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Densité aux points x (n, 2) = (δ, M) et aux instants t (n, 1). Renvoie (n, 1)."""
        xn = _to_unit(x, self.lo, self.hi)
        tn = 2 * t / self.t_max - 1
        return nn.functional.softplus(self.net(torch.cat([xn, tn], dim=1)))


class DemandNet(nn.Module):
    """MLP f_θ(δ) : demande des fondamentalistes apprise (terme manquant)."""

    def __init__(self, lo: float, hi: float, width: int = 20, depth: int = 3) -> None:
        super().__init__()
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))
        self.net = mlp(1, 1, width, depth)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """f_θ(δ) pour δ de dimension (n, 1), normalisé à partir du domaine de δ."""
        return self.net(_to_unit(delta, self.lo, self.hi))


class PhysicalParams(nn.Module):
    """Paramètres physiques du modèle : entraînables (en log, donc positifs) ou fixés.

    Arguments :
        trainable : valeurs initiales (> 0) des paramètres inconnus.
        fixed : valeurs des paramètres supposés connus.
    """

    def __init__(self, trainable: Mapping[str, float], fixed: Mapping[str, float]) -> None:
        super().__init__()
        common = set(trainable) & set(fixed)
        if common:
            raise ValueError(f"Paramètres à la fois entraînables et fixés : {sorted(common)}")
        if any(v <= 0 for v in trainable.values()):
            raise ValueError("Les valeurs initiales des paramètres entraînables doivent être > 0.")
        self.log_values = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(math.log(v))) for name, v in trainable.items()}
        )
        self.fixed = {name: float(v) for name, v in fixed.items()}

    def __getitem__(self, name: str) -> torch.Tensor | float:
        if name in self.log_values:
            return torch.exp(self.log_values[name])
        return self.fixed[name]

    def names(self) -> set[str]:
        """Noms de tous les paramètres (entraînables et fixés)."""
        return set(self.log_values) | set(self.fixed)

    def trainable_values(self) -> dict[str, float]:
        """Valeurs courantes des paramètres entraînables."""
        return {name: math.exp(v.item()) for name, v in self.log_values.items()}


class ChiarellaPINN(nn.Module):
    """Modèle complet : densité p̃, paramètres physiques et, en option, demande apprise f_θ.

    Le drift garde toujours la structure connue du modèle (suiveurs de tendance, mémoire M) :
    seule la demande des fondamentalistes f(δ) peut être remplacée par un réseau.
    """

    def __init__(
        self, density: DensityNet, params: PhysicalParams, demand_net: DemandNet | None = None
    ) -> None:
        super().__init__()
        self.density = density
        self.params = params
        self.demand_net = demand_net

    def demand(self, delta: torch.Tensor) -> torch.Tensor:
        """Demande des fondamentalistes f(δ) : polynôme −κδ − κ₃δ³ ou réseau f_θ."""
        if self.demand_net is not None:
            return self.demand_net(delta)
        return fundamental_demand(delta, self.params["kappa"], self.params["kappa3"])

    def drift(self, delta: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Drift (a_δ, a_M) de l'état, avec les paramètres courants."""
        p = self.params
        return drift_from_demand(
            self.demand(delta), m, beta=p["beta"], alpha=p["alpha"], gamma=p["gamma"]
        )


def build_model(
    model_config: Mapping[str, Any], lo: torch.Tensor, hi: torch.Tensor, t_max: float
) -> ChiarellaPINN:
    """Construit le modèle décrit par la section `model` d'une config.

    Arguments :
        model_config : section `model` (fundamental, trainable, fixed, density_net, demand_net).
        lo, hi : bornes (2,) du domaine en (δ, M).
        t_max : dernier instant du domaine temporel.
    """
    kind = model_config["fundamental"]
    if kind not in REQUIRED_PARAMS:
        raise ValueError(f"model.fundamental doit valoir 'linear' ou 'nn', pas {kind!r}.")
    params = PhysicalParams(model_config["trainable"], model_config["fixed"])
    if params.names() != REQUIRED_PARAMS[kind]:
        raise ValueError(
            f"Paramètres attendus pour fundamental={kind!r} : {sorted(REQUIRED_PARAMS[kind])}, "
            f"reçus : {sorted(params.names())}."
        )
    density = DensityNet(lo, hi, t_max, **model_config.get("density_net", {}))
    demand_net = None
    if kind == "nn":
        demand_net = DemandNet(float(lo[0]), float(hi[0]), **model_config.get("demand_net", {}))
    return ChiarellaPINN(density, params, demand_net)
