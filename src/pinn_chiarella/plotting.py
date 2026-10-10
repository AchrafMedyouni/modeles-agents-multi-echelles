"""Figures standard : contrôle des données, convergence, terme appris, densités."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinn_chiarella.data import SyntheticDataset  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_DARK = "#104281"

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
