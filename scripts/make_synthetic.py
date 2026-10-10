"""Génère un jeu de données synthétique et sa figure de contrôle à partir d'une config.

Usage :
    python scripts/make_synthetic.py --config configs/synth_chiarella.yaml

Écrit data/synthetic/<nom>.npz et data/synthetic/<nom>.png, <nom> étant le nom du fichier
de config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pinn_chiarella.data import generate_synthetic, save_synthetic
from pinn_chiarella.plotting import plot_synthetic_check

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="config YAML de l'expérience")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data" / "synthetic", help="dossier de sortie"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = generate_synthetic(config)

    npz_path = args.out_dir / f"{args.config.stem}.npz"
    png_path = npz_path.with_suffix(".png")
    save_synthetic(dataset, npz_path)
    plot_synthetic_check(dataset, png_path)

    print(f"Données : {npz_path}  (snapshots {dataset.snapshots.shape})")
    print(f"Figure  : {png_path}")
    for t, snap in zip(dataset.t_obs, dataset.snapshots, strict=True):
        mean, std = snap.mean(axis=0), snap.std(axis=0)
        print(
            f"  t = {t:<4g} δ : {mean[0]:+.3f} ± {std[0]:.3f}    M : {mean[1]:+.3f} ± {std[1]:.3f}"
        )


if __name__ == "__main__":
    main()
