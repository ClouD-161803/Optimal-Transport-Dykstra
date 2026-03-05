"""Plotting utilities for the Dykstra projection project.

Provides dedicated plotter classes for disjoint plotting domains:

* ``DykstraPlotter`` for Dykstra/PGD convergence diagnostics.
* ``DistributionPlotter`` for sample-distribution visualisations.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from .projection_result import ProjectionResult

TITLE_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 11
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10


class _BasePlotter:
    """Shared plotting base with output handling and common styling."""

    def __init__(self, output_dir: str, dpi: int = 150) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _style_axis(
        ax: Axes,
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

    def _save_and_show(
        self, fig: Figure, filename: str | None, show: bool
    ) -> Figure:
        fig.tight_layout()
        if filename is not None:
            fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi)
        if show:
            plt.show()
        return fig


class DykstraPlotter(_BasePlotter):
    """Reusable plotter for Dykstra projection benchmarks and experiments.

    Parameters
    ----------
    output_dir : str
        Directory where figures are saved.  Created automatically if it
        does not exist.
    dpi : int, optional
        Resolution for saved figures (default 150).
    """

    # Public API

    def plot_convergence_comparison(
        self,
        results: Sequence[ProjectionResult],
        labels: Sequence[str],
        max_iter: int,
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot side-by-side convergence curves colour-coded by solver state.

        Each panel shows the squared error on a log scale.  Iterations are
        coloured green (converged), red (stalled), or blue (normal) based
        on the arrays stored in the ``ProjectionResult``.

        Parameters
        ----------
        results : sequence of ProjectionResult
            One result per solver to compare.  Each must have been
            produced with ``track_error=True``.
        labels : sequence of str
            Display name for each solver (same length as *results*).
        max_iter : int
            Number of Dykstra cycles that were run (used to build the
            iteration axis).
        filename : str, optional
            If given the figure is saved to ``output_dir / filename``.
        show : bool, optional
            Whether to call ``plt.show()`` (default ``True``).

        Returns
        -------
        matplotlib.figure.Figure
            The figure object, useful for further customisation.
        """
        n_panels = len(results)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(7 * n_panels, 5), sharex=True, sharey=True,
        )
        if n_panels == 1:
            axes = [axes]

        iters = np.arange(max_iter + 1)

        for ax, label, result in zip(axes, labels, results):
            self._draw_convergence_panel(ax, result, iters, label)

        return self._save_and_show(fig, filename, show)

    def plot_outer_iteration_solver_comparison(
        self,
        vanilla_results: Sequence[ProjectionResult],
        fast_forward_results: Sequence[ProjectionResult],
        filename_prefix: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot all outer-iteration solver comparisons on a single figure.

        The figure contains one row per outer PGD iteration and two
        columns: vanilla Dykstra (left) and fast-forward Dykstra (right).
        """
        if len(vanilla_results) != len(fast_forward_results):
            raise ValueError(
                "vanilla_results and fast_forward_results must have the same length."
            )

        n_outer = len(vanilla_results)
        fig, axes = plt.subplots(n_outer, 2, figsize=(12, 4 * n_outer))
        if n_outer == 1:
            axes = np.array([axes])

        for outer_idx, (vanilla_res, fast_res) in enumerate(
            zip(vanilla_results, fast_forward_results)
        ):
            vanilla_sq = vanilla_res.squared_errors
            fast_sq = fast_res.squared_errors
            if vanilla_sq is None or fast_sq is None:
                raise ValueError(
                    "All ProjectionResult entries must include squared_errors "
                    "(use track_error=True)."
                )

            self._draw_convergence_panel(
                axes[outer_idx][0],
                vanilla_res,
                np.arange(len(vanilla_sq)),
                f"Outer {outer_idx + 1} — Vanilla Dykstra",
            )
            self._draw_convergence_panel(
                axes[outer_idx][1],
                fast_res,
                np.arange(len(fast_sq)),
                f"Outer {outer_idx + 1} — Fast-Forward Dykstra",
            )

        filename = f"{filename_prefix}.png" if filename_prefix is not None else None
        return self._save_and_show(fig, filename, show)

    # Internal helpers

    @staticmethod
    def _classify_and_group(
        squared_errors: np.ndarray,
        stalled_errors: np.ndarray,
        converged_errors: np.ndarray,
    ) -> list[tuple[tuple[str, str], list[int]]]:
        """Classify each iteration and group consecutive runs of the same class.

        Each iteration is assigned a ``(colour, label)`` pair:

        * **Converged** (green) – ``converged_errors[i]`` is not NaN.
        * **Stalled** (red) – ``stalled_errors[i]`` is not NaN.
        * **Normal** (blue) – everything else.

        Consecutive iterations sharing the same colour are then collected
        into contiguous groups.
        """
        classification: list[tuple[str, str]] = []
        for i in range(len(squared_errors)):
            if not np.isnan(converged_errors[i]):
                classification.append(("tab:green", "Converged"))
            elif not np.isnan(stalled_errors[i]):
                classification.append(("tab:red", "Stalled"))
            else:
                classification.append(("tab:blue", "Normal"))

        groups: list[tuple[tuple[str, str], list[int]]] = []
        current_run: list[int] = [0]
        for idx in range(1, len(classification)):
            if classification[idx][0] == classification[current_run[0]][0]:
                current_run.append(idx)
            else:
                groups.append((classification[current_run[0]], current_run))
                current_run = [idx]
        groups.append((classification[current_run[0]], current_run))
        return groups

    def _draw_convergence_panel(
        self,
        ax: Axes,
        result: ProjectionResult,
        iters: np.ndarray,
        title: str,
    ) -> None:
        """Draw a single convergence panel on *ax*."""
        sq = result.squared_errors
        st = result.stalled_errors
        cv = result.converged_errors

        if sq is None or st is None or cv is None:
            raise ValueError(
                "ProjectionResult must have squared_errors, stalled_errors,"
                " and converged_errors set (use track_error=True)."
            )

        groups = self._classify_and_group(sq, st, cv)

        seen_labels: set[str] = set()
        for g, ((colour, lbl), indices) in enumerate(groups):
            # Extend each segment by one point so lines connect between groups.
            if g < len(groups) - 1:
                xs = indices + [groups[g + 1][1][0]]
            else:
                xs = indices
            label_arg = lbl if lbl not in seen_labels else None
            seen_labels.add(lbl)
            ax.semilogy(
                iters[xs], sq[xs], ".-",
                color=colour, markersize=3, label=label_arg,
            )

        self._style_axis(
            ax=ax,
            title=title,
            xlabel="Iteration",
            ylabel="Squared error",
        )
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, which="both", alpha=0.3)


class DistributionPlotter(_BasePlotter):
    """Plotter for sample-distribution visualisations.
    """

    def plot_kr_map_distribution_single_solver(
        self,
        normal_samples: np.ndarray,
        synthetic_samples: np.ndarray,
        mapped_samples: np.ndarray,
        solver_label: str,
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot a 3-panel comparison for one mapped KR solver output."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        panels = [
            (normal_samples, r"Reference normal $\mathcal{N}(0, I_2)$", "tab:blue"),
            (synthetic_samples, "Synthetic distribution", "tab:red"),
            (mapped_samples, f"Mapped with {solver_label}", "tab:green"),
        ]

        for ax, (samples, title, color) in zip(axes, panels):
            self._draw_distribution_panel(
                ax=ax, samples=samples, title=title,
                xlabel="$x_1$", ylabel="$x_2$", color=color,
                s=16, xlim=(-10, 10), ylim=(-10, 10), grid_alpha=0.4,
            )

        return self._save_and_show(
            fig, filename or "kr_map_distribution_single_solver.png", show
        )

    def plot_kr_map_distribution_comparison(
        self,
        normal_samples: np.ndarray,
        synthetic_samples: np.ndarray,
        vanilla_mapped_samples: np.ndarray,
        fast_mapped_samples: np.ndarray,
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot a 2x2 comparison of reference, synthetic, and mapped samples.

        Panels are arranged as:

        * top-left: reference normal samples
        * top-right: synthetic source samples
        * bottom-left: samples mapped with vanilla Dykstra coefficients
        * bottom-right: samples mapped with fast-forward Dykstra coefficients
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        panels = [
            (normal_samples, r"Reference normal $\mathcal{N}(0, I_2)$", "tab:blue"),
            (synthetic_samples, "Synthetic distribution", "tab:red"),
            (vanilla_mapped_samples, "Mapped with vanilla Dykstra", "tab:green"),
            (fast_mapped_samples, "Mapped with fast-forward Dykstra", "tab:purple"),
        ]

        for ax, (samples, title, color) in zip(axes.flatten(), panels):
            self._draw_distribution_panel(
                ax=ax, samples=samples, title=title,
                xlabel="$x_1$", ylabel="$x_2$", color=color,
                s=16, grid_alpha=0.4,
            )

        return self._save_and_show(
            fig, filename or "kr_map_distribution_comparison.png", show
        )

    def plot_distributions(
        self,
        zeta: np.ndarray,
        z: np.ndarray,
        seed: int,
        m: int,
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot and save standard-normal and crescent sample distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        self._draw_distribution_panel(
            ax=ax1, samples=zeta,
            title=r"Standard normal $\mathcal{N}(0, I_2)$",
            xlabel="$z_1$", ylabel="$z_2$", color="blue",
        )
        self._draw_distribution_panel(
            ax=ax2, samples=z,
            title="Crescent distribution",
            xlabel="$x_1$", ylabel="$x_2$", color="red",
        )

        return self._save_and_show(
            fig, filename or f"synthetic_distribution_SEED={seed}_M={m}.png", show
        )

    def _draw_distribution_panel(
        self,
        ax: Axes,
        samples: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        color: str,
        s: int = 20,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        grid_alpha: float = 0.6,
    ) -> None:
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            alpha=0.5,
            color=color,
            edgecolor="k",
            s=s,
        )
        self._style_axis(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, linestyle="--", alpha=grid_alpha)
        ax.set_aspect("equal", adjustable="box")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
