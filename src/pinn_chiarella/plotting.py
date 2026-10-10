"""Figures standard : contrôle des données, convergence, terme appris, densités."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pinn_chiarella.data import SyntheticDataset  # noqa: E402
from pinn_chiarella.models import ChiarellaPINN  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_DARK = "#104281"
ORANGE = "#eb6834"
NEUTRAL_FILL = "#d9d8d3"
BAND = "#efeeea"
# Rampe séquentielle du bleu, du clair au foncé.
BLUE_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")

PARAM_LABELS = {"kappa": "κ", "kappa3": "κ₃", "beta": "β", "sigma": "σ", "alpha": "α", "gamma": "γ"}

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "font.size": 9,
}


def plot_synthetic_check(dataset: SyntheticDataset, path: Path, n_paths: int = 40) -> None:
    """Figure de contrôle d'un jeu synthétique.

    À gauche : quelques trajectoires de δ et de M, leur moyenne empirique et les instants
    observés. À droite : histogramme de δ à chaque instant observé (même axe horizontal).
    """
    if dataset.paths is None or dataset.t_paths is None:
        raise ValueError(
            "La figure de contrôle a besoin des trajectoires (jeu fraîchement simulé)."
        )
    t, paths = dataset.t_paths, dataset.paths
    t_obs, snaps = dataset.t_obs, dataset.snapshots
    n_obs = len(t_obs)
    p = dataset.true_params

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 1.5 * n_obs + 1.6), layout="constrained")
        left, right = fig.subfigures(1, 2, width_ratios=[1.35, 1])

        axes = left.subplots(2, 1, sharex=True)
        for ax, col, name in zip(axes, (0, 1), ("δ = P − V", "M"), strict=True):
            ax.axhline(0.0, color=GRID, lw=1, zorder=0)
            for t_i in t_obs:
                ax.axvline(t_i, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
            ax.plot(t, paths[:, :n_paths, col], color=BLUE, lw=0.6, alpha=0.35, zorder=2)
            ax.plot(t, paths[:, :, col].mean(axis=1), color=BLUE_DARK, lw=2, zorder=3)
            ax.set_ylabel(name)
            ax.margins(x=0.01)
        axes[0].set_title(f"Trajectoires ({n_paths} sur {paths.shape[1]}) et moyenne empirique")
        axes[0].plot([], [], color=BLUE, lw=0.6, alpha=0.6, label="trajectoires")
        axes[0].plot([], [], color=BLUE_DARK, lw=2, label="moyenne sur les N particules")
        axes[0].plot([], [], color=MUTED, lw=0.8, ls=(0, (2, 3)), label="instants observés")
        axes[0].legend(frameon=False, ncols=3, loc="upper right", fontsize=8)
        axes[1].set_xlabel("temps t")

        hist_axes = right.subplots(n_obs, 1, sharex=True, sharey=True)
        delta = snaps[:, :, 0]
        bins = np.linspace(delta.min(), delta.max(), 61)
        for ax, t_i, d in zip(hist_axes, t_obs, delta, strict=True):
            ax.hist(d, bins=bins, density=True, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
            ax.axvline(0.0, color=GRID, lw=1, zorder=0)
            ax.axvline(d.mean(), color=BLUE_DARK, lw=1.5)
            ax.text(
                0.99,
                0.9,
                f"t = {t_i:g}\nmoyenne {d.mean():+.2f}\nécart-type {d.std():.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=INK_SECONDARY,
            )
            ax.set_ylabel("densité")
        hist_axes[0].set_title("Histogrammes de δ aux instants observés")
        hist_axes[-1].set_xlabel("δ = P − V   (trait foncé : moyenne)")

        fig.suptitle(
            f"Données synthétiques Chiarella — κ = {p['kappa']:g}, κ₃ = {p['kappa3']:g}, "
            f"β = {p['beta']:g}, σ = {p['sigma']:g}, α = {p['alpha']:g}, γ = {p['gamma']:g} ; "
            f"N = {paths.shape[1]}, dt = {dataset.dt:g}, seed = {dataset.seed}",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_parameter_convergence(log: dict[str, Any], path: Path) -> None:
    """Convergence des paramètres appris vers les vraies valeurs, et évolution des losses.

    Un panneau par paramètre entraînable (valeur apprise et vraie valeur), un panneau pour
    L_pde, un pour L_data, et un pour la masse totale de p̃ aux instants observés (qui doit
    tendre vers 1) si le log la contient.
    """
    history = log["history"]
    its = np.array([r["it"] for r in history])
    names = list(log["initial"])
    has_mass = "mass" in history[0]
    panels = len(names) + 2 + has_mass

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, panels, figsize=(3.3 * panels, 3.4), layout="constrained", squeeze=False
        )
        axes = axes[0]
        for ax, name in zip(axes, names, strict=False):
            values = np.array([r[name] for r in history])
            true = log["true"][name]
            ax.axhline(true, color=INK_SECONDARY, lw=1.2, ls=(0, (4, 3)), label="vraie valeur")
            ax.plot(its, values, color=BLUE, lw=2, label="apprise")
            err = (values[-1] - true) / true
            ax.set_title(
                f"{PARAM_LABELS.get(name, name)} : {values[-1]:.3f} (vrai {true:g}, {err:+.0%})"
            )
            ax.set_xlabel("itération")
            ax.grid(axis="y", color=GRID, lw=0.6)
        axes[0].legend(frameon=False, fontsize=8, loc="center right")

        tau = log["tau"]
        for ax, key, title in zip(
            axes[len(names) :], ("L_pde", "L_data"), ("L_pde (résidu)", "L_data"), strict=False
        ):
            values = np.array([r[key] for r in history])
            ax.plot(its, values, color=BLUE, lw=1.5)
            ax.set_title(title)
            ax.set_xlabel("itération")
            ax.grid(axis="y", color=GRID, lw=0.6)
        axes[len(names)].set_yscale("log")

        if has_mass:
            ax = axes[-1]
            mass = np.array([r["mass"] for r in history])
            ax.axhline(1.0, color=INK_SECONDARY, lw=1.2, ls=(0, (4, 3)))
            ax.fill_between(
                its, mass.min(axis=1), mass.max(axis=1), color=BLUE_RAMP[0], alpha=0.6, lw=0
            )
            ax.plot(its, mass.mean(axis=1), color=BLUE, lw=1.2)
            ax.set_ylim(0.8, 1.2)
            ax.set_title("Masse totale de p̃ (cible : 1)")
            ax.set_xlabel("itération\n(trait : moyenne des instants,\nbande : min à max)")
            ax.grid(axis="y", color=GRID, lw=0.6)

        fig.suptitle(
            f"Convergence des paramètres — {log['iterations']} itérations, τ = {tau:g}, "
            f"{log['duration_s']:.0f} s",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_learned_demand(log: dict[str, Any], path: Path) -> None:
    """Demande des fondamentalistes apprise contre la vraie, avec la zone couverte par les données.

    La zone grisée va du percentile 1 au percentile 99 de δ sur l'ensemble des snapshots.
    """
    demand = log["demand"]
    delta = np.array(demand["delta"])
    d_lo, d_hi = demand["data_range"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
        ax.axvspan(d_lo, d_hi, color=BAND, zorder=0)
        ax.axhline(0.0, color=GRID, lw=1, zorder=1)
        ax.plot(delta, demand["true"], color=INK_SECONDARY, lw=1.5, ls=(0, (4, 3)), label="vraie f")
        ax.plot(delta, demand["learned"], color=BLUE, lw=2, label="f apprise")
        ax.text(
            0.5 * (d_lo + d_hi),
            0.03,
            "zone couverte par les données\n(percentiles 1 à 99 de δ)",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )
        ax.set_xlabel("δ = P − V")
        ax.set_ylabel("f(δ)")
        ax.set_title("Demande des fondamentalistes : apprise et vraie")
        ax.text(
            0.5 * (d_lo + d_hi),
            0.97,
            f"RMSE dans la zone : {demand['rmse_data_range']:.3f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=INK_SECONDARY,
        )
        ax.legend(frameon=False, loc="upper right")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def density_grid(
    model: ChiarellaPINN, t: float, n_grid: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densité PINN à l'instant t sur une grille du domaine.

    Renvoie (grille δ (n,), grille M (n,), p̃ (n, n) indexée [δ, M]).
    """
    density = model.density
    d = torch.linspace(float(density.lo[0]), float(density.hi[0]), n_grid, device=density.lo.device)
    m = torch.linspace(float(density.lo[1]), float(density.hi[1]), n_grid, device=density.lo.device)
    dd, mm = torch.meshgrid(d, m, indexing="ij")
    x = torch.stack([dd.flatten(), mm.flatten()], dim=1)
    with torch.no_grad():
        p = density(x, torch.full((x.shape[0], 1), float(t), device=x.device))
    return d.cpu().numpy(), m.cpu().numpy(), p.reshape(n_grid, n_grid).cpu().numpy()


def density_marginals(
    model: ChiarellaPINN, t: float, n_grid: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Marginales en δ et en M de la densité PINN à l'instant t, par intégration sur une grille.

    Renvoie (grille δ, marginale δ, grille M, marginale M, masse totale ∫p̃).
    """
    d_np, m_np, p = density_grid(model, t, n_grid)
    marg_d = np.trapezoid(p, m_np, axis=1)
    marg_m = np.trapezoid(p, d_np, axis=0)
    return d_np, marg_d, m_np, marg_m, float(np.trapezoid(marg_d, d_np))


def plot_density_check(model: ChiarellaPINN, dataset: SyntheticDataset, path: Path) -> None:
    """Densité PINN contre histogramme Monte-Carlo à l'instant non observé t_test.

    Marginales en δ et en M : la densité PINN est intégrée sur l'autre variable.
    """
    d, marg_d, m, marg_m, mass = density_marginals(model, dataset.t_test)
    test = dataset.test_snapshot

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
        for ax, grid, marg, samples, name in zip(
            axes,
            (d, m),
            (marg_d, marg_m),
            (test[:, 0], test[:, 1]),
            ("δ = P − V", "M"),
            strict=True,
        ):
            ax.hist(
                samples,
                bins=60,
                density=True,
                color=NEUTRAL_FILL,
                edgecolor=SURFACE,
                linewidth=0.3,
                label=f"Monte-Carlo ({len(samples)} particules)",
            )
            ax.plot(grid, marg, color=BLUE, lw=2, label="PINN (marginale)")
            ax.set_xlabel(name)
        axes[0].set_ylabel("densité")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(
            f"Densité à l'instant non observé t = {dataset.t_test:g} — "
            f"masse totale de la densité PINN : {mass:.3f}",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_sweep_demand(logs: list[dict[str, Any]], path: Path) -> None:
    """Demande des fondamentalistes apprise contre la vraie, pour les runs d'un sweep.

    Un panneau par cas ; les seeds d'un même cas sont superposées. La zone grisée est la
    zone couverte par les données du cas (percentiles 1 à 99 de δ).
    """
    cases: dict[str, list[dict[str, Any]]] = {}
    for log in logs:
        cases.setdefault(log["case"], []).append(log)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, len(cases), figsize=(4.6 * len(cases), 4), layout="constrained", squeeze=False
        )
        for ax, (case, case_logs) in zip(axes[0], cases.items(), strict=True):
            demand = case_logs[0]["demand"]
            delta, true = np.array(demand["delta"]), np.array(demand["true"])
            d_lo, d_hi = demand["data_range"]
            ax.axvspan(d_lo, d_hi, color=BAND, zorder=0)
            ax.axhline(0.0, color=GRID, lw=1, zorder=1)
            ax.plot(delta, true, color=INK_SECONDARY, lw=1.5, ls=(0, (4, 3)), label="vraie f")
            for i, log in enumerate(case_logs):
                ax.plot(
                    log["demand"]["delta"],
                    log["demand"]["learned"],
                    color=BLUE,
                    lw=1.5,
                    alpha=0.8,
                    label="f apprise" + (" (une courbe par seed)" if len(case_logs) > 1 else "")
                    if i == 0
                    else None,
                )
            rmse = np.mean([log["demand"]["rmse_data_range"] for log in case_logs])
            ax.set_title(f"{case}\nRMSE dans la zone : {rmse:.3f}", fontsize=10)
            ax.set_xlabel("δ = P − V")
            # Cadrage sur la zone couverte, élargie de moitié de chaque côté : au-delà, la
            # vraie f (cubique) écraserait l'échelle.
            half = 0.5 * (d_hi - d_lo)
            shown = true[(delta >= d_lo - half) & (delta <= d_hi + half)]
            pad = 0.1 * (shown.max() - shown.min())
            ax.set_xlim(max(delta[0], d_lo - half), min(delta[-1], d_hi + half))
            ax.set_ylim(shown.min() - pad, shown.max() + pad)
        axes[0, 0].set_ylabel("f(δ)")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle(
            "Demande des fondamentalistes par cas (zone grisée : couverte par les données)",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_delta_extent(datasets: dict[str, SyntheticDataset], path: Path) -> None:
    """Étendue de δ (percentiles 1 à 99) en fonction du temps, pour plusieurs jeux simulés.

    Montre combien de temps les données occupent les grandes valeurs de |δ|. Les points
    marquent les instants observés de chaque jeu. À droite : zoom sur le début.
    """
    colors = (BLUE, ORANGE, "#1baf7a", "#eda100")
    t_end = max(float(d.t_paths[-1]) for d in datasets.values())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained", sharey=True)
        for ax, t_max in zip(axes, (t_end, 0.5), strict=True):
            ax.axhline(0.0, color=GRID, lw=1, zorder=0)
            for (name, data), color in zip(datasets.items(), colors, strict=False):
                lo, hi = np.percentile(data.paths[:, :, 0], [1, 99], axis=1)
                ax.fill_between(data.t_paths, lo, hi, color=color, alpha=0.18, lw=0)
                ax.plot(data.t_paths, lo, color=color, lw=1.5, label=name)
                ax.plot(data.t_paths, hi, color=color, lw=1.5)
                obs = np.percentile(data.snapshots[:, :, 0], [1, 99], axis=1)
                for edge in obs:
                    ax.plot(data.t_obs, edge, "o", color=color, ms=5, mec=SURFACE, mew=1)
            ax.set_xlim(-0.01 * t_max, t_max)
            ax.set_xlabel("temps t")
        axes[0].set_ylabel("δ : percentiles 1 et 99")
        axes[0].set_title("Étendue de δ sur tout l'horizon")
        axes[1].set_title("Zoom sur le transitoire (t ≤ 0,5)")
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle(
            "Zone de δ couverte par les données au cours du temps (points : instants observés)",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_density_times(
    model: ChiarellaPINN, samples: dict[float, np.ndarray], t_obs: np.ndarray, path: Path
) -> None:
    """Densité PINN contre Monte-Carlo à plusieurs instants : marginales et densité jointe.

    Une ligne par instant. Colonnes : marginale en δ, marginale en M, puis plan (δ, M) avec
    les particules Monte-Carlo et les lignes de niveau de la densité PINN.

    Arguments :
        model : modèle entraîné.
        samples : particules Monte-Carlo indépendantes, (n, 2) par instant.
        t_obs : instants observés à l'entraînement (pour l'indiquer dans les titres).
        path : figure de sortie.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(samples),
            3,
            figsize=(13, 3.6 * len(samples)),
            layout="constrained",
            squeeze=False,
        )
        for row, (t, x) in zip(axes, samples.items(), strict=True):
            d, m, p = density_grid(model, t)
            marg_d, marg_m = np.trapezoid(p, m, axis=1), np.trapezoid(p, d, axis=0)
            mass = float(np.trapezoid(marg_d, d))
            kind = "instant observé" if np.any(np.isclose(t_obs, t)) else "instant non observé"
            panels = ((d, marg_d, x[:, 0], "δ = P − V"), (m, marg_m, x[:, 1], "M"))
            for ax, (grid, marg, values, name) in zip(row, panels, strict=False):
                ax.hist(
                    values,
                    bins=60,
                    density=True,
                    color=NEUTRAL_FILL,
                    edgecolor=SURFACE,
                    linewidth=0.3,
                    label=f"Monte-Carlo ({len(values)} particules)",
                )
                ax.plot(grid, marg, color=BLUE, lw=2, label="PINN (marginale)")
                ax.set_xlabel(name)
                ax.set_ylabel("densité")
            row[0].set_title(f"t = {t:g} ({kind}) — masse de p̃ : {mass:.3f}")
            ax = row[2]
            ax.scatter(x[:, 0], x[:, 1], s=2, color=NEUTRAL_FILL, linewidths=0, rasterized=True)
            ax.contour(d, m, p.T, levels=6, colors=BLUE, linewidths=1.2)
            ax.set_xlabel("δ = P − V")
            ax.set_ylabel("M")
            ax.set_title("Plan (δ, M) : particules et niveaux de p̃")
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(
            "Densité PINN et Monte-Carlo indépendant (particules de test)",
            fontsize=10,
            x=0.01,
            ha="left",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
