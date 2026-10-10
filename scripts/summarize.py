"""Résume les expériences dans experiments/summary.md (paramètres appris contre vrais).

Usage :
    python scripts/summarize.py pinn_params pinn_nn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"kappa": "κ", "kappa3": "κ₃", "beta": "β", "sigma": "σ"}


def run_section(name: str, log: dict) -> list[str]:
    """Section Markdown d'une expérience : tableau des paramètres et diagnostics."""
    last = log["history"][-1]
    lines = [
        f"## {name}",
        "",
        f"{log['iterations']} itérations, τ = {log['tau']:g}, seed {log['seed']}, "
        f"{log['duration_s'] / 60:.1f} min. Données : `{Path(log['data']).name}`. "
        f"Fin : L_pde = {last['L_pde']:.2e}, L_data = {last['L_data']:.3f}.",
        "",
        "| paramètre | vrai | initial | appris | erreur relative |",
        "|---|---|---|---|---|",
    ]
    final, true = log["final"], log["true"]
    for key, value in final.items():
        err = (value - true[key]) / true[key]
        lines.append(
            f"| {LABELS.get(key, key)} | {true[key]:g} | {log['initial'][key]:g} | "
            f"{value:.3f} | {err:+.1%} |"
        )
    lines.append("")
    if {"kappa", "sigma"} <= final.keys():
        ratio = final["sigma"] ** 2 / (2 * final["kappa"])
        ratio_true = true["sigma"] ** 2 / (2 * true["kappa"])
        lines += [
            f"Identifiabilité : σ²/(2κ) appris = {ratio:.3f}, vrai = {ratio_true:.3f} "
            f"({(ratio - ratio_true) / ratio_true:+.1%}). κ et σ se compensent si ce rapport "
            "est mieux retrouvé que κ et σ séparément.",
            "",
        ]
    if "final_mass" in log:
        masses = ", ".join(
            f"{m:.3f} (t = {t:g})" for t, m in zip(log["t_obs"], log["final_mass"], strict=True)
        )
        lines += [f"Masse totale de p̃ aux instants observés (cible : 1) : {masses}.", ""]
    demand = log["demand"]
    d_lo, d_hi = demand["data_range"]
    lines += [
        f"Demande des fondamentalistes f(δ) : RMSE = {demand['rmse_data_range']:.3f} dans la "
        f"zone couverte par les données (δ entre {d_lo:.2f} et {d_hi:.2f}).",
        "",
        f"Figures : `{name}/convergence.png`, `{name}/demande.png`, `{name}/densite.png`.",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="noms des expériences (dossiers de experiments/)")
    args = parser.parse_args()

    lines = ["# Résumé des expériences", ""]
    for name in args.runs:
        log = json.loads((ROOT / "experiments" / name / "log.json").read_text(encoding="utf-8"))
        lines += run_section(name, log)
    out = ROOT / "experiments" / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
