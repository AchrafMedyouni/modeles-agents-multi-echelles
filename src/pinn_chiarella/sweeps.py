"""Séries de runs (sweeps) : variations d'une config de base, exécution et résumé.

Un sweep est décrit par un YAML (configs/sweeps/<nom>.yaml) :

    base: configs/pinn_params.yaml     # config de base
    iterations: 5000                   # optionnel (5000 par défaut)
    seeds: [0, 1, 2]                   # optionnel (seed de la config de base par défaut)
    grid:                              # produit cartésien de valeurs...
      training.tau: [1, 10, 100]
    runs:                              # ... ou liste explicite de cas
      - name: sigma_fixe
        set:
          model.trainable: {kappa: 0.3, beta: 0.1}

Les clés sont des chemins pointés dans la config ; la valeur remplace entièrement le champ.
Chaque run écrit dans experiments/<sweep>/<run>/.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ITERATIONS = 5000


@dataclass(frozen=True)
class SweepRun:
    """Un run d'un sweep.

    Attributs :
        name : nom du run (dossier de sortie), suffixé par la seed s'il y en a plusieurs.
        case : nom du cas (commun aux seeds d'un même cas).
        seed : graine du run.
        config : config complète du run.
    """

    name: str
    case: str
    seed: int
    config: dict[str, Any]


def set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    """Remplace en place le champ désigné par un chemin pointé (ex. 'training.tau')."""
    *parents, leaf = path.split(".")
    node = config
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise KeyError(f"Chemin de config inconnu : {path!r}")
        node = node[key]
    node[leaf] = copy.deepcopy(value)


def _format_value(value: Any) -> str:
    """Valeur lisible dans un nom de dossier."""
    return Path(value).stem if isinstance(value, str) else f"{value:g}"


def expand_runs(sweep: dict[str, Any], base: dict[str, Any]) -> list[SweepRun]:
    """Développe la description d'un sweep en liste de runs (cas × seeds).

    Arguments :
        sweep : contenu du YAML du sweep (`grid` et/ou `runs`, `seeds`).
        base : config de base, non modifiée.
    """
    cases: list[tuple[str, dict[str, Any]]] = []
    grid = sweep.get("grid") or {}
    if grid:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            name = "_".join(
                f"{k.split('.')[-1]}={_format_value(v)}" for k, v in zip(keys, values, strict=True)
            )
            cases.append((name, dict(zip(keys, values, strict=True))))
    for run in sweep.get("runs") or []:
        cases.append((run["name"], run.get("set") or {}))
    if not cases:
        raise ValueError("Le sweep ne décrit aucun run (ni `grid` ni `runs`).")
    if len({name for name, _ in cases}) != len(cases):
        raise ValueError("Deux cas du sweep portent le même nom.")

    seeds = sweep.get("seeds") or [base["seed"]]
    runs = []
    for (case, overrides), seed in itertools.product(cases, seeds):
        config = copy.deepcopy(base)
        for path, value in overrides.items():
            set_by_path(config, path, value)
        config["seed"] = int(seed)
        name = case if len(seeds) == 1 else f"{case}_seed{seed}"
        runs.append(SweepRun(name=name, case=case, seed=int(seed), config=config))
    return runs


def run_one(
    run: SweepRun,
    root: Path,
    out_dir: Path,
    iterations: int,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Exécute un run de sweep (entraînement et figures standard) et renvoie son log.

    Arguments :
        run : run à exécuter.
        root : racine du dépôt (le champ `data` de la config lui est relatif).
        out_dir : dossier de sortie du run.
        iterations : nombre d'itérations.
        num_threads : nombre de threads PyTorch (1 quand plusieurs runs tournent en parallèle).
    """
    import torch

    from pinn_chiarella.plotting import (
        plot_density_check,
        plot_learned_demand,
        plot_parameter_convergence,
    )
    from pinn_chiarella.train import train

    if num_threads is not None:
        torch.set_num_threads(num_threads)
    result = train(
        run.config,
        root / run.config["data"],
        out_dir,
        iterations=iterations,
        verbose=False,
        extra_log={"case": run.case, "run": run.name},
    )
    plot_parameter_convergence(result.log, out_dir / "convergence.png")
    plot_learned_demand(result.log, out_dir / "demande.png")
    plot_density_check(result.model, result.dataset, out_dir / "densite.png")
    return result.log


def is_done(out_dir: Path, iterations: int) -> bool:
    """Vrai si le run a déjà été mené à son terme avec ce nombre d'itérations."""
    path = out_dir / "log.json"
    if not path.exists():
        return False
    log = json.loads(path.read_text(encoding="utf-8"))
    return "final" in log and log["iterations"] == iterations


def load_sweep_logs(sweep_dir: Path) -> list[dict[str, Any]]:
    """Charge les logs des runs terminés d'un sweep, triés par nom de run."""
    logs = []
    for path in sorted(sweep_dir.glob("*/log.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        if "final" in log:
            logs.append(log)
    return logs


def relative_errors(log: dict[str, Any]) -> dict[str, float]:
    """Erreur relative (appris − vrai) / vrai de chaque paramètre entraînable."""
    return {k: (v - log["true"][k]) / log["true"][k] for k, v in log["final"].items()}


def beta_demand_correlation(logs: list[dict[str, Any]]) -> tuple[float, int]:
    """Corrélation de Pearson, entre runs, de l'erreur relative sur β et de l'écart de f.

    L'écart de f est (gain − 1), le gain étant la projection de f apprise sur la vraie f
    dans la zone couverte par les données. Renvoie (corrélation, nombre de runs).
    """
    beta_err = [relative_errors(log)["beta"] for log in logs]
    f_gap = [log["demand"]["gain_data_range"] - 1 for log in logs]
    if len(logs) < 3 or np.std(beta_err) == 0 or np.std(f_gap) == 0:
        return float("nan"), len(logs)
    return float(np.corrcoef(beta_err, f_gap)[0, 1]), len(logs)
