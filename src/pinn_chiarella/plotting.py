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
# Rampe séquentielle (clair -> foncé) pour une grandeur ordonnée, ici le temps.
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
            colors = [BLUE_RAMP[round(i)] for i in np.linspace(0, 4, mass.shape[1])]
            ax.axhline(1.0, color=INK_SECONDARY, lw=1.2, ls=(0, (4, 3)))
            for values, t_i, color in zip(mass.T, log["t_obs"], colors, strict=True):
                ax.plot(its, values, color=color, lw=1.5, label=f"t = {t_i:g}")
            ax.set_ylim(0.8, 1.2)
            ax.set_title("Masse totale de p̃ (cible : 1)")
            ax.set_xlabel("itération")
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.legend(frameon=False, fontsize=8, ncols=2, loc="lower right")

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


def density_marginals(
    model: ChiarellaPINN, t: float, n_grid: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Marginales en δ et en M de la densité PINN à l'instant t, par intégration sur une grille.

    Renvoie (grille δ, marginale δ, grille M, marginale M, masse totale ∫p̃).
    """
    density = model.density
    d = torch.linspace(float(density.lo[0]), float(density.hi[0]), n_grid, device=density.lo.device)
    m = torch.linspace(float(density.lo[1]), float(density.hi[1]), n_grid, device=density.lo.device)
    dd, mm = torch.meshgrid(d, m, indexing="ij")
    x = torch.stack([dd.flatten(), mm.flatten()], dim=1)
    with torch.no_grad():
        p = density(x, torch.full((x.shape[0], 1), float(t), device=x.device))
    p = p.reshape(n_grid, n_grid).cpu().numpy()
    d_np, m_np = d.cpu().numpy(), m.cpu().numpy()
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
