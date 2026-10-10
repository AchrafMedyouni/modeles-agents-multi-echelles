"""Entraîne le PINN Fokker-Planck décrit par une config et produit les figures standard.

Usage :
    python scripts/train.py --config configs/pinn_params.yaml

Écrit dans experiments/<nom>/ (<nom> = nom du fichier de config) : config.yaml, log.json,
model.pt, convergence.png, demande.png, densite.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pinn_chiarella.plotting import (
    plot_density_check,
    plot_learned_demand,
    plot_parameter_convergence,
)
from pinn_chiarella.train import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="config YAML de l'expérience")
    parser.add_argument("--iterations", type=int, help="remplace training.iterations (run court)")
    parser.add_argument("--out-dir", type=Path, help="dossier de sortie")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = args.out_dir or ROOT / "experiments" / args.config.stem
    result = train(config, ROOT / config["data"], out_dir, iterations=args.iterations)

    plot_parameter_convergence(result.log, out_dir / "convergence.png")
    plot_learned_demand(result.log, out_dir / "demande.png")
    plot_density_check(result.model, result.dataset, out_dir / "densite.png")

    log = result.log
    print(f"Terminé en {log['duration_s']:.0f} s — sorties dans {out_dir}")
    for name, value in log["final"].items():
        print(f"  {name:<6} appris {value:.4f}   vrai {log['true'][name]:g}")


if __name__ == "__main__":
    main()
