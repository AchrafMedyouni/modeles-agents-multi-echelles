"""Résume les sweeps dans experiments/summary_sweeps.md et trace la demande apprise par cas.

Usage :
    python scripts/summarize_sweeps.py kappa_sigma snapshots n_samples tau terme_manquant couverture
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pinn_chiarella.plotting import plot_sweep_demand
from pinn_chiarella.sweeps import beta_demand_correlation, load_sweep_logs, relative_errors
from pinn_chiarella.train import load_run

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"kappa": "κ", "kappa3": "κ₃", "beta": "β", "sigma": "σ"}
# Sweeps où la demande f est l'objet de l'étude : figure par cas.
DEMAND_SWEEPS = {"terme_manquant", "couverture"}


def relative_rmse(log: dict) -> float:
    """RMSE de f dans la zone couverte, rapportée à la moyenne quadratique de la vraie f."""
    demand = log["demand"]
    delta, true = np.array(demand["delta"]), np.array(demand["true"])
    d_lo, d_hi = demand["data_range"]
    inside = (delta >= d_lo) & (delta <= d_hi)
    return demand["rmse_data_range"] / float(np.sqrt(np.mean(true[inside] ** 2)))


def run_table(logs: list[dict]) -> list[str]:
    """Tableau Markdown : une ligne par run."""
    names = [p for p in PARAMS if any(p in log["final"] for log in logs)]
    both = all({"kappa", "sigma"} <= log["final"].keys() for log in logs)
    header = ["run", *(f"{PARAMS[p]} (erreur)" for p in names)]
    header += ["σ²/(2κ)"] if both else []
    header += ["RMSE f", "RMSE f relative", "gain f", "zone δ", "masse p̃", "NLL test", "L_pde"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for log in logs:
        err = relative_errors(log)
        row = [log["run"]]
        for p in names:
            if p in log["final"]:
                row.append(f"{log['final'][p]:.3f} ({err[p]:+.1%})")
            elif p in log["fixed"]:
                row.append(f"fixé à {log['fixed'][p]:g}")
            else:
                row.append("—")
        if both:
            ratio = log["final"]["sigma"] ** 2 / (2 * log["final"]["kappa"])
            true = log["true"]["sigma"] ** 2 / (2 * log["true"]["kappa"])
            row.append(f"{ratio:.3f} ({(ratio - true) / true:+.1%})")
        demand = log["demand"]
        l_pde = np.mean([r["L_pde"] for r in log["history"][-10:]])
        # La masse fluctue d'une itération à l'autre : moyenne sur les derniers enregistrements.
        mass = np.mean([r["mass"] for r in log["history"][-10:]], axis=0)
        row += [
            f"{demand['rmse_data_range']:.3f}",
            f"{relative_rmse(log):.1%}",
            f"{demand['gain_data_range']:.3f}",
            f"[{demand['data_range'][0]:.2f}, {demand['data_range'][1]:.2f}]",
            f"{mass.min():.3f} à {mass.max():.3f}",
            f"{log['test_nll']:.3f}",
            f"{l_pde:.1e}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def case_table(logs: list[dict]) -> list[str]:
    """Tableau Markdown : moyenne ± écart-type entre seeds, par cas."""
    cases: dict[str, list[dict]] = {}
    for log in logs:
        cases.setdefault(log["case"], []).append(log)

    def stat(values: list[float], fmt: str) -> str:
        return f"{format(np.mean(values), fmt)} ± {format(np.std(values), fmt)}"

    lines = [
        "| cas | seeds | erreur β | erreur σ | RMSE f | gain f |",
        "|---|---|---|---|---|---|",
    ]
    for case, group in cases.items():
        errs = [relative_errors(log) for log in group]
        sigma = stat([e["sigma"] for e in errs], "+.1%") if "sigma" in errs[0] else "fixé"
        lines.append(
            f"| {case} | {len(group)} | {stat([e['beta'] for e in errs], '+.1%')} | {sigma} | "
            f"{stat([g['demand']['rmse_data_range'] for g in group], '.3f')} | "
            f"{stat([g['demand']['gain_data_range'] for g in group], '.3f')} |"
        )
    return lines


def sweep_section(name: str, logs: list[dict]) -> list[str]:
    """Section Markdown d'un sweep."""
    first = logs[0]
    lines = [
        f"## {name}",
        "",
        f"{len(logs)} runs, {first['iterations']} itérations chacun, "
        f"{sum(log['duration_s'] for log in logs) / 60:.0f} min de calcul au total.",
        "",
        *run_table(logs),
        "",
    ]
    if len({log["seed"] for log in logs}) > 1:
        lines += ["Moyenne ± écart-type entre seeds :", "", *case_table(logs), ""]
    if name == "terme_manquant":
        mlp = [log for log in logs if log["case"].startswith("mlp")]
        for label, group in (("runs MLP", mlp), ("tous les runs", logs)):
            corr, n = beta_demand_correlation(group)
            lines.append(
                f"Compensation β / f : corrélation de Pearson entre l'erreur relative sur β et "
                f"(gain f − 1), sur {label} (n = {n}) : {corr:+.2f}."
            )
        lines.append("")
    if name in DEMAND_SWEEPS:
        plot_sweep_demand(logs, ROOT / "experiments" / name / "demande.png")
        lines += [f"Figure : `{name}/demande.png`.", ""]
    return lines


def variance_section(runs: list[str]) -> list[str]:
    """Diagnostic du biais de σ : écart-type de δ et de M sous p̃ contre celui des données.

    Pour chaque run et chaque instant observé, rapport (écart-type sous p̃, intégré sur la
    grille) / (écart-type empirique des données). Un rapport < 1 signifie que p̃ est trop
    étroite.
    """
    lines = [
        "## Diagnostic du biais de σ : largeur de p̃ contre largeur des données",
        "",
        "Rapport entre l'écart-type sous p̃ (densité normalisée, intégrée sur la grille du "
        "domaine) et l'écart-type empirique des données, à chaque instant observé. "
        "Un rapport inférieur à 1 signifie que p̃ est plus étroite que les données.",
        "",
    ]
    results = {}
    for run in runs:
        model, dataset = load_run(ROOT / "experiments" / run, ROOT)
        moments = [model.density.moments(float(t)) for t in dataset.t_obs]
        std_model = np.sqrt([m[2] for m in moments])
        results[run] = (dataset.t_obs, std_model / dataset.snapshots.std(axis=1))
    for col, name in enumerate(("δ", "M")):
        t_obs = next(iter(results.values()))[0]
        header = ["run", *(f"t = {t:g}" for t in t_obs), "moyenne"]
        lines += [f"Écart-type de {name} : p̃ / données", ""]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for run, (_, ratio) in results.items():
            cells = [f"{r:.3f}" for r in ratio[:, col]]
            lines.append(f"| {run} | " + " | ".join(cells) + f" | {ratio[:, col].mean():.3f} |")
        lines.append("")
    ratios = np.concatenate([ratio.ravel() for _, ratio in results.values()])
    lines += [
        f"Sur l'ensemble : rapport moyen {ratios.mean():.3f}, de {ratios.min():.3f} à "
        f"{ratios.max():.3f}. "
        + (
            "p̃ est systématiquement plus étroite que les données."
            if ratios.mean() < 0.97
            else "p̃ n'est pas systématiquement plus étroite que les données : la sous-estimation "
            "de σ ne vient pas de la largeur de p̃ aux instants observés (loss données, domaine)."
        ),
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweeps", nargs="+", help="noms des sweeps (dossiers de experiments/)")
    parser.add_argument(
        "--variance-runs",
        nargs="*",
        default=[],
        help="runs (<sweep>/<run>) pour le diagnostic de largeur de p̃ contre les données",
    )
    args = parser.parse_args()

    lines = [
        "# Résumé des sweeps",
        "",
        "Colonnes : erreur relative de chaque paramètre appris ; RMSE de f dans la zone couverte "
        "par les données (percentiles 1 à 99 de δ), en valeur absolue et rapportée à la moyenne "
        "quadratique de la vraie f ; gain f = projection de f apprise sur la vraie f dans cette "
        "zone (1 = rappel retrouvé, < 1 = sous-estimé) ; masse de p̃ aux instants observés "
        "(min à max entre instants, cible 1) ; NLL du snapshot de test (comparable seulement "
        "entre runs de même jeu de test). Masse et L_pde sont moyennées sur les 1000 dernières "
        "itérations.",
        "",
    ]
    for name in args.sweeps:
        logs = load_sweep_logs(ROOT / "experiments" / name)
        if not logs:
            lines += [f"## {name}", "", "Aucun run terminé.", ""]
            continue
        lines += sweep_section(name, logs)
    if args.variance_runs:
        lines += variance_section(args.variance_runs)
    out = ROOT / "experiments" / "summary_sweeps.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
