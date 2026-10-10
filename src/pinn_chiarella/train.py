"""Boucle d'entraînement du PINN Fokker-Planck, pilotée par une config YAML.

Loss : L = τ·L_pde + Σ_i L_data(t_i). Les sorties vont dans experiments/<nom>/ :
config copiée, log JSON (toutes les `log_every` itérations) et poids du modèle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pinn_chiarella.data import SyntheticDataset, load_synthetic
from pinn_chiarella.losses import data_loss, pde_loss, sample_uniform
from pinn_chiarella.models import ChiarellaPINN, build_model
from pinn_chiarella.sde import fundamental_demand

# Le log est réécrit sur disque à cette fréquence (en nombre d'enregistrements).
FLUSH_EVERY = 10


@dataclass
class TrainResult:
    """Résultat d'un entraînement.

    Attributs :
        model : modèle entraîné.
        log : contenu du log JSON (vraies valeurs, valeurs initiales, historique, durée…).
        dataset : données utilisées.
        out_dir : dossier de sortie de l'expérience.
    """

    model: ChiarellaPINN
    log: dict[str, Any]
    dataset: SyntheticDataset
    out_dir: Path


def resolve_device(name: str) -> torch.device:
    """Convertit le champ `device` de la config ('cpu', 'cuda', 'auto'…) en device PyTorch."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def data_domain(snapshots: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Bornes (lo, hi) du domaine en (δ, M) : étendue des données, élargie de `margin`."""
    points = snapshots.reshape(-1, snapshots.shape[-1])
    return points.min(axis=0) - margin, points.max(axis=0) + margin


def demand_profile(model: ChiarellaPINN, dataset: SyntheticDataset, n: int = 200) -> dict[str, Any]:
    """Demande des fondamentalistes apprise et vraie sur une grille couvrant le domaine de δ.

    Renvoie la grille, f apprise, f vraie, la zone couverte par les données (percentiles
    1 et 99 de δ) et l'erreur quadratique moyenne dans cette zone.
    """
    density = model.density
    grid = torch.linspace(float(density.lo[0]), float(density.hi[0]), n, device=density.lo.device)
    with torch.no_grad():
        learned = model.demand(grid.unsqueeze(1)).squeeze(1).cpu().numpy()
    grid_np = grid.cpu().numpy()
    true = fundamental_demand(grid_np, dataset.true_params["kappa"], dataset.true_params["kappa3"])
    d_lo, d_hi = np.percentile(dataset.snapshots[:, :, 0], [1, 99])
    inside = (grid_np >= d_lo) & (grid_np <= d_hi)
    return {
        "delta": grid_np.tolist(),
        "learned": learned.tolist(),
        "true": true.tolist(),
        "data_range": [float(d_lo), float(d_hi)],
        "rmse_data_range": float(np.sqrt(np.mean((learned - true)[inside] ** 2))),
    }


def train(
    config: dict[str, Any],
    data_path: Path,
    out_dir: Path,
    iterations: int | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Entraîne le PINN décrit par `config` sur le jeu de données `data_path`.

    Arguments :
        config : config de l'expérience (sections `model` et `training`, `seed`, `device`).
        data_path : fichier .npz écrit par scripts/make_synthetic.py.
        out_dir : dossier de sortie (créé si besoin).
        iterations : remplace `training.iterations` (runs courts, tests).
        verbose : affiche l'avancement toutes les 1000 itérations.
    """
    cfg_model, cfg_train = config["model"], config["training"]
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(config.get("device", "cpu"))

    dataset = load_synthetic(data_path)
    lo_np, hi_np = data_domain(dataset.snapshots, float(cfg_model.get("margin", 0.5)))
    lo = torch.tensor(lo_np, dtype=torch.float32, device=device)
    hi = torch.tensor(hi_np, dtype=torch.float32, device=device)
    snapshots = torch.tensor(dataset.snapshots, dtype=torch.float32, device=device)
    t_obs = torch.tensor(dataset.t_obs, dtype=torch.float32, device=device)
    t_max = float(dataset.t_obs.max())

    model = build_model(cfg_model, lo, hi, t_max).to(device)
    n_iter = int(cfg_train["iterations"] if iterations is None else iterations)
    tau = float(cfg_train["tau"])
    n_col, n_int = int(cfg_train["n_collocation"]), int(cfg_train["n_integral"])
    log_every = int(cfg_train.get("log_every", 100))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_train["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg_train.get("lr_step") or max(1, n_iter // 3)),
        gamma=float(cfg_train.get("lr_decay", 0.3)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    initial = model.params.trainable_values()
    log: dict[str, Any] = {
        "seed": seed,
        "data": str(data_path),
        "iterations": n_iter,
        "tau": tau,
        "true": {name: dataset.true_params[name] for name in initial},
        "initial": initial,
        "fixed": model.params.fixed,
        "domain": {"lo": lo_np.tolist(), "hi": hi_np.tolist(), "t_max": t_max},
        "history": [],
    }

    start = time.perf_counter()
    for it in range(n_iter + 1):
        optimizer.zero_grad()
        x_col = sample_uniform(lo, hi, n_col)
        t_col = t_max * torch.rand(n_col, 1, device=device)
        l_pde = pde_loss(
            model.density, model.drift, model.params["sigma"], model.params["alpha"], x_col, t_col
        )
        l_data = data_loss(model.density, snapshots, t_obs, lo, hi, n_int)
        loss = tau * l_pde + l_data
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % log_every == 0:
            record = {
                "it": it,
                "L_pde": l_pde.item(),
                "L_data": l_data.item(),
                "loss": loss.item(),
                **model.params.trainable_values(),
            }
            log["history"].append(record)
            log["duration_s"] = time.perf_counter() - start
            if len(log["history"]) % FLUSH_EVERY == 0:
                _write_log(log, out_dir)
            if verbose and it % (10 * log_every) == 0:
                print(json.dumps(record), f"{log['duration_s']:.0f}s", flush=True)

    log["duration_s"] = time.perf_counter() - start
    log["final"] = model.params.trainable_values()
    log["demand"] = demand_profile(model, dataset)
    _write_log(log, out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")
    return TrainResult(model=model, log=log, dataset=dataset, out_dir=out_dir)


def _write_log(log: dict[str, Any], out_dir: Path) -> None:
    """Écrit le log JSON de l'expérience."""
    (out_dir / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
