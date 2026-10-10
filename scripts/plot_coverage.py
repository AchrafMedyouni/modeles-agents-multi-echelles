"""Trace l'étendue de δ (percentiles 1 à 99) en fonction du temps pour des jeux synthétiques.

Usage :
    python scripts/plot_coverage.py --config configs/synth_a.yaml configs/synth_b.yaml \
        --out experiments/couverture/etendue_delta.png

Les trajectoires sont resimulées en mémoire à partir des configs (mêmes seeds, donc mêmes
particules que les jeux sur disque) ; aucun fichier de données n'est réécrit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from pinn_chiarella.data import generate_synthetic
from pinn_chiarella.plotting import plot_delta_extent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="configs synth_*")
    parser.add_argument("--out", type=Path, required=True, help="figure de sortie")
    args = parser.parse_args()

    datasets = {}
    for path in args.config:
        data = generate_synthetic(yaml.safe_load(path.read_text(encoding="utf-8")))
        datasets[path.stem] = data
        print(path.stem)
        for t in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, float(data.t_paths[-1])):
            i = int(np.argmin(np.abs(data.t_paths - t)))
            lo, hi = np.percentile(data.paths[i, :, 0], [1, 99])
            print(f"  t = {data.t_paths[i]:<5g} δ entre {lo:+.2f} et {hi:+.2f}")
    plot_delta_extent(datasets, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
