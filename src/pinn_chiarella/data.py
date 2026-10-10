"""Chargement et prétraitement des données (synthétiques et Telefonica) en snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pinn_chiarella.sde import (
    ChiarellaParams,
    InitialState,
    observation_steps,
    sample_initial_state,
    simulate,
)

# Pas de temps (en unités de temps) entre deux points des trajectoires de contrôle.
PATH_RESOLUTION = 0.01


@dataclass(frozen=True)
class SyntheticDataset:
    """Jeu de données synthétique : nuages de particules observés à quelques instants.

    Attributs :
        snapshots : tableau (len(t_obs), N, 2), état (δ, M) aux instants observés.
        t_obs : instants observés.
        test_snapshot : tableau (n_test, 2), état à t_test (simulation indépendante).
        t_test : instant non observé, réservé à la validation.
        true_params : vrais paramètres du modèle.
        seed : graine utilisée.
        dt : pas de temps de la simulation.
        paths : tableau (len(t_paths), N, 2), trajectoires des mêmes particules que
            `snapshots`, sur une grille fine (figure de contrôle uniquement ; absent d'un
            jeu rechargé depuis le disque).
        t_paths : instants de `paths`.
    """

    snapshots: np.ndarray
    t_obs: np.ndarray
    test_snapshot: np.ndarray
    t_test: float
    true_params: dict[str, float]
    seed: int
    dt: float
    paths: np.ndarray | None = None
    t_paths: np.ndarray | None = None


def generate_synthetic(config: dict[str, Any]) -> SyntheticDataset:
    """Simule le jeu de données synthétique décrit par une config (voir configs/).

    Les snapshots d'entraînement et le snapshot de test viennent de deux simulations
    indépendantes, toutes deux déterminées par la graine de la config.
    """
    params = ChiarellaParams(**config["model"])
    sim = config["simulation"]
    seed, dt = int(config["seed"]), float(sim["dt"])
    init = InitialState(**sim.get("init", {}))
    t_obs = np.asarray(sim["t_obs"], dtype=float)
    t_test = float(sim["t_test"])
    rng_train, rng_test = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))

    # Une seule simulation, enregistrée sur une grille fine contenant les instants observés.
    obs_steps = observation_steps(t_obs, dt)
    stride = max(1, round(PATH_RESOLUTION / dt))
    path_steps = np.union1d(np.arange(0, obs_steps[-1] + 1, stride), obs_steps)
    x0 = sample_initial_state(int(sim["n_samples"]), init, rng_train)
    paths = simulate(params, x0, path_steps * dt, dt, rng_train)

    x0_test = sample_initial_state(int(sim["n_test"]), init, rng_test)
    test_snapshot = simulate(params, x0_test, [t_test], dt, rng_test)[0]

    return SyntheticDataset(
        snapshots=paths[np.searchsorted(path_steps, obs_steps)],
        t_obs=t_obs,
        test_snapshot=test_snapshot,
        t_test=t_test,
        true_params=params.as_dict(),
        seed=seed,
        dt=dt,
        paths=paths,
        t_paths=path_steps * dt,
    )


def save_synthetic(dataset: SyntheticDataset, path: Path) -> None:
    """Sauvegarde le jeu de données en .npz (sans les trajectoires de contrôle)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snapshots=dataset.snapshots,
        t_obs=dataset.t_obs,
        test_snapshot=dataset.test_snapshot,
        t_test=dataset.t_test,
        seed=dataset.seed,
        dt=dataset.dt,
        **{f"true_{name}": value for name, value in dataset.true_params.items()},
    )


def load_synthetic(path: Path) -> SyntheticDataset:
    """Recharge un jeu de données écrit par `save_synthetic`."""
    with np.load(path) as npz:
        return SyntheticDataset(
            snapshots=npz["snapshots"],
            t_obs=npz["t_obs"],
            test_snapshot=npz["test_snapshot"],
            t_test=float(npz["t_test"]),
            true_params={k[5:]: float(npz[k]) for k in npz.files if k.startswith("true_")},
            seed=int(npz["seed"]),
            dt=float(npz["dt"]),
        )
