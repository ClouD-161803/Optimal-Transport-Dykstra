"""Plotting utilities for the Dykstra projection project.

Provides a central ``ProjectPlotter`` class whose methods produce the
standard figures used across benchmarks and experiments.  New plot types
should be added as methods on this class so that styling remains
consistent throughout the project.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from .projection_result import ProjectionResult


class ProjectPlotter:
    """Reusable plotter for Dykstra projection benchmarks and experiments.

    Parameters
    ----------
    output_dir : str
        Directory where figures are saved.  Created automatically if it
        does not exist.
    dpi : int, optional
        Resolution for saved figures (default 150).
    """

    def __init__(self, output_dir: str, dpi: int = 150) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)

    # Public API

    def plot_convergence_comparison(
        self,
        results: Sequence[ProjectionResult],
        labels: Sequence[str],
        max_iter: int,
        suptitle: str | None = None,
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
        suptitle : str, optional
            Overall figure title.
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

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=13)

        plt.tight_layout()

        if filename is not None:
            fig.savefig(
                os.path.join(self.output_dir, filename), dpi=self.dpi,
            )

        if show:
            plt.show()

        return fig
    
    # Internal helpers

    @staticmethod
    def _classify_iterations(
        squared_errors: np.ndarray,
        stalled_errors: np.ndarray,
        converged_errors: np.ndarray,
    ) -> list[tuple[str, str]]:
        """Return a ``(colour, label)`` pair for every iteration index.

        Categories
        ----------
        * **Converged** (green) – ``converged_errors[i]`` is not NaN.
        * **Stalled** (red) – ``stalled_errors[i]`` is not NaN.
        * **Normal** (blue) – everything else.
        """
        classification: list[tuple[str, str]] = []
        for i in range(len(squared_errors)):
            if not np.isnan(converged_errors[i]):
                classification.append(("tab:green", "Converged"))
            elif not np.isnan(stalled_errors[i]):
                classification.append(("tab:red", "Stalled"))
            else:
                classification.append(("tab:blue", "Normal"))
        return classification

    @staticmethod
    def _build_contiguous_groups(
        classification: list[tuple[str, str]],
    ) -> list[tuple[tuple[str, str], list[int]]]:
        """Group consecutive iterations that share the same colour."""
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
        
        # The ProjectionResult fields are Optional in the dataclass, but the
        # plotting routines require them.  Explicitly check for None so the
        # type checker (Pylance) knows these values are non-None for the
        # remainder of the function.
        if sq is None or st is None or cv is None:
            raise ValueError(
                "ProjectionResult must have squared_errors, stalled_errors,"
                " and converged_errors set (use track_error=True)."
            )

        classification = self._classify_iterations(sq, st, cv)
        groups = self._build_contiguous_groups(classification)

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

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Squared error")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
