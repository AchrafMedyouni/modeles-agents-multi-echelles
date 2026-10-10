"""Compare la densité PINN d'un run à un Monte-Carlo indépendant, à plusieurs instants.

Usage :
    python scripts/plot_density.py --run instable/tout_inconnu \
        --synth-config configs/synth_instable.yaml --times 4 10

Écrit experiments/<run>/densite_temps.png. Les particules Monte-Carlo sont celles du
snapshot de test (simulation indépendante des données d'entraînement), resimulées aux
instants demandés.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pinn_chiarella.data import simulate_test_particles
from pinn_chiarella.plotting import plot_density_times
from pinn_chiarella.train import load_run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run, relatif à experiments/")
    parser.add_argument("--synth-config", type=Path, required=True, help="config des données")
    parser.add_argument("--times", type=float, nargs="+", required=True, help="instants à tracer")
    args = parser.parse_args()

    out_dir = ROOT / "experiments" / args.run
    model, dataset = load_run(out_dir, ROOT)
    config = yaml.safe_load(args.synth_config.read_text(encoding="utf-8"))
    times = sorted(args.times)
    particles = simulate_test_particles(config, times)
    samples = dict(zip(times, particles, strict=True))
    plot_density_times(model, samples, dataset.t_obs, out_dir / "densite_temps.png")
    print(out_dir / "densite_temps.png")


if __name__ == "__main__":
    main()
