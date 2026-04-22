"""Plotting utilities for the Dykstra projection project.

Provides dedicated plotter classes for disjoint plotting domains:

* ``DykstraPlotter`` for Dykstra/PGD convergence diagnostics.
* ``DistributionPlotter`` for sample-distribution visualisations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import math
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - scipy is expected in normal runs
    gaussian_filter = None
try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover - scipy is expected in normal runs
    gaussian_kde = None

from .projection_result import ProjectionResult

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16


@dataclass(frozen=True)
class DistributionStyle:
    """Style bundle for drawing a particle distribution panel."""

    point_color: str
    point_alpha: float
    point_size: int
    point_edge_color: str
    contour_cmap: Any
    contour_levels: int | Sequence[float]
    contour_alpha: float
    contour_lw: float


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

        for ax in axes[1:]:
            ax.set_ylabel("")

        return self._save_and_show(fig, filename, show)

    def plot_outer_iteration_solver_comparison(
        self,
        vanilla_results: Sequence[ProjectionResult],
        fast_forward_results: Sequence[ProjectionResult],
        outer_indices: Sequence[int] | None = None,
        filename_prefix: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot selected outer-iteration solver comparisons on one figure.

        The figure contains one row per plotted outer PGD iteration and two
        columns: vanilla Dykstra (left) and fast-forward Dykstra (right).
        """
        if len(vanilla_results) != len(fast_forward_results):
            raise ValueError(
                "vanilla_results and fast_forward_results must have the same length."
            )

        n_outer = len(vanilla_results)
        if n_outer == 0:
            raise ValueError(
                "At least one outer iteration result is required for plotting."
            )

        if outer_indices is None:
            outer_indices = list(range(n_outer))
        elif len(outer_indices) != n_outer:
            raise ValueError(
                "outer_indices must have the same length as the results sequences."
            )

        fig, axes = plt.subplots(n_outer, 2, figsize=(12, 4 * n_outer))
        if n_outer == 1:
            axes = np.array([axes])

        for row_idx, (outer_idx, vanilla_res, fast_res) in enumerate(
            zip(outer_indices, vanilla_results, fast_forward_results)
        ):
            vanilla_sq = vanilla_res.squared_errors
            fast_sq = fast_res.squared_errors
            if vanilla_sq is None or fast_sq is None:
                raise ValueError(
                    "All ProjectionResult entries must include squared_errors "
                    "(use track_error=True)."
                )

            self._draw_convergence_panel(
                axes[row_idx][0],
                vanilla_res,
                np.arange(len(vanilla_sq)),
                f"Outer {outer_idx} - Vanilla Dykstra",
            )
            self._draw_convergence_panel(
                axes[row_idx][1],
                fast_res,
                np.arange(len(fast_sq)),
                f"Outer {outer_idx} - Fast-Forward Dykstra",
            )

            axes[row_idx][1].set_ylabel("")

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

        self._style_axis(ax=ax, title=title, xlabel="Cycle", ylabel="Squared error")
        ax.grid(True, which="both", alpha=0.3)

        if not np.any(sq > 0):
            ax.text(
                0.5, 0.5,
                "Iterate strictly feasible\n(projection trivial)",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=LEGEND_FONT_SIZE,
                color="tab:green",
            )
            return

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

        ax.legend(fontsize=LEGEND_FONT_SIZE)


class DistributionPlotter(_BasePlotter):
    """Plotter for sample-distribution visualisations.
    """

    def __init__(self, output_dir: str, dpi: int = 150) -> None:
        super().__init__(output_dir=output_dir, dpi=dpi)
        self.styles = self._build_distribution_styles()

    @staticmethod
    def _build_distribution_styles() -> dict[str, DistributionStyle]:
        reference_cmap = LinearSegmentedColormap.from_list(
            "reference_gray_contours",
            ["#E5E5E5", "#8F8F8F", "#2E2E2E"],
        )
        sheared_cmap = LinearSegmentedColormap.from_list(
            "sheared_orange_yellow_contours",
            ["#F1C40F", "#E67E22", "#B22222"],
        )
        mapped_cmap = LinearSegmentedColormap.from_list(
            "mapped_blue_green_contours",
            ["#27AE60", "#16A085", "#2874A6", "#1F4EAA"],
        )
        return {
            "reference": DistributionStyle(
                point_color="black",
                point_alpha=0.33,
                point_size=16,
                point_edge_color="none",
                contour_cmap=reference_cmap,
                contour_levels=7,
                contour_alpha=0.95,
                contour_lw=1.1,
            ),
            "sheared": DistributionStyle(
                point_color="#C0392B",
                point_alpha=0.34,
                point_size=16,
                point_edge_color="none",
                contour_cmap=sheared_cmap,
                contour_levels=7,
                contour_alpha=0.95,
                contour_lw=1.15,
            ),
            "mapped": DistributionStyle(
                point_color="#2166AC",
                point_alpha=0.34,
                point_size=16,
                point_edge_color="none",
                contour_cmap=mapped_cmap,
                contour_levels=7,
                contour_alpha=0.95,
                contour_lw=1.15,
            ),
        }

    def _resolve_style(
        self,
        style: str | DistributionStyle | None,
        fallback_key: str = "reference",
    ) -> DistributionStyle:
        if isinstance(style, DistributionStyle):
            return style
        if isinstance(style, str):
            if style not in self.styles:
                raise ValueError(f"Unknown distribution style: {style}")
            return self.styles[style]
        return self.styles[fallback_key]

    @staticmethod
    def _resolve_limits(
        samples: np.ndarray,
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
        margin_ratio: float = 0.08,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if xlim is not None and ylim is not None:
            return xlim, ylim

        x = samples[:, 0]
        y = samples[:, 1]
        if xlim is None:
            xmin, xmax = float(np.min(x)), float(np.max(x))
            xspan = xmax - xmin
            if xspan <= 1e-12:
                xspan = 1.0
            xmargin = margin_ratio * xspan
            xlim_resolved = (xmin - xmargin, xmax + xmargin)
        else:
            xlim_resolved = xlim

        if ylim is None:
            ymin, ymax = float(np.min(y)), float(np.max(y))
            yspan = ymax - ymin
            if yspan <= 1e-12:
                yspan = 1.0
            ymargin = margin_ratio * yspan
            ylim_resolved = (ymin - ymargin, ymax + ymargin)
        else:
            ylim_resolved = ylim
        return xlim_resolved, ylim_resolved

    def _estimate_density_grid(
        self,
        samples: np.ndarray,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        bins: int = 80,
        smooth_sigma: float = 1.0,
        method: str = "kde",
        kde_bw_factor: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolved_xlim, resolved_ylim = self._resolve_limits(
            samples=samples, xlim=xlim, ylim=ylim
        )
        effective_method = method.strip().lower()
        if effective_method == "kde" and gaussian_kde is not None:
            nx = max(int(bins), 140)
            ny = max(int(bins), 140)
            x_grid = np.linspace(resolved_xlim[0], resolved_xlim[1], nx)
            y_grid = np.linspace(resolved_ylim[0], resolved_ylim[1], ny)
            xx, yy = np.meshgrid(x_grid, y_grid)
            coords = np.vstack([xx.ravel(), yy.ravel()])
            kde = gaussian_kde(samples[:, :2].T)
            factor_array = np.asarray(getattr(kde, "factor", 1.0), dtype=float)
            base_factor = float(factor_array.reshape(-1)[0])
            bw_factor = max(float(kde_bw_factor), 1e-3)
            kde.set_bandwidth(bw_method=base_factor * bw_factor)
            zz = kde(coords).reshape(xx.shape)
            return xx, yy, zz

        hist, x_edges, y_edges = np.histogram2d(
            samples[:, 0],
            samples[:, 1],
            bins=bins,
            range=[resolved_xlim, resolved_ylim],
            density=False,
        )
        z = hist.T
        sigma = max(float(smooth_sigma), 0.0)
        if sigma > 0.0 and gaussian_filter is not None:
            z = gaussian_filter(z, sigma=sigma)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx, yy = np.meshgrid(x_centers, y_centers)
        return xx, yy, z

    def _draw_density_contours(
        self,
        ax: Axes,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        style: DistributionStyle,
        contour_levels: int | Sequence[float] | None = None,
    ) -> Any | None:
        zmax = float(np.max(zz))
        if zmax <= 0.0:
            return None

        levels = contour_levels if contour_levels is not None else style.contour_levels
        if isinstance(levels, int):
            levels = max(2, levels)
            min_level = zmax * 0.10
            max_level = zmax * 0.95
            if max_level <= min_level:
                min_level = zmax * 0.40
            levels = np.linspace(min_level, max_level, levels)

        return ax.contour(
            xx,
            yy,
            zz,
            levels=levels,
            cmap=style.contour_cmap,
            linewidths=style.contour_lw,
            alpha=style.contour_alpha,
            zorder=2,
        )

    def plot_kr_map_distribution_single_solver(
        self,
        normal_samples: np.ndarray,
        synthetic_samples: np.ndarray,
        mapped_samples: np.ndarray,
        solver_label: str,
        panel_titles: tuple[str, str, str] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        draw_contours: bool = True,
        contour_bins: int = 80,
        contour_levels: int | Sequence[float] = 6,
        contour_smoothing_sigma: float = 1.0,
        contour_method: str = "kde",
        contour_kde_bw_factor: float = 1.35,
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot a 3-panel comparison for one mapped KR solver output."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if panel_titles is None:
            panel_titles = (
                r"Reference normal $\mathcal{N}(0, I_2)$",
                "Synthetic distribution",
                f"Mapped with {solver_label}",
            )

        panels = [
            (normal_samples, panel_titles[0], "reference"),
            (synthetic_samples, panel_titles[1], "sheared"),
            (mapped_samples, panel_titles[2], "mapped"),
        ]

        for ax, (samples, title, style_key) in zip(axes, panels):
            self._draw_distribution_panel(
                ax=ax, samples=samples, title=title,
                xlabel="$x_1$", ylabel="$x_2$", style=style_key,
                xlim=xlim, ylim=ylim, grid_alpha=0.4,
                draw_contours=draw_contours,
                contour_bins=contour_bins,
                contour_levels=contour_levels,
                contour_smoothing_sigma=contour_smoothing_sigma,
                contour_method=contour_method,
                contour_kde_bw_factor=contour_kde_bw_factor,
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
        panel_titles: tuple[str, str, str, str] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        draw_contours: bool = True,
        contour_bins: int = 80,
        contour_levels: int | Sequence[float] = 6,
        contour_smoothing_sigma: float = 1.0,
        contour_method: str = "kde",
        contour_kde_bw_factor: float = 1.35,
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

        if panel_titles is None:
            panel_titles = (
                r"Reference normal $\mathcal{N}(0, I_2)$",
                "Synthetic distribution",
                "Mapped with vanilla Dykstra",
                "Mapped with fast-forward Dykstra",
            )

        panels = [
            (normal_samples, panel_titles[0], "reference"),
            (synthetic_samples, panel_titles[1], "sheared"),
            (vanilla_mapped_samples, panel_titles[2], "mapped"),
            (fast_mapped_samples, panel_titles[3], "mapped"),
        ]

        for ax, (samples, title, style_key) in zip(axes.flatten(), panels):
            self._draw_distribution_panel(
                ax=ax, samples=samples, title=title,
                xlabel="$x_1$", ylabel="$x_2$", style=style_key,
                xlim=xlim, ylim=ylim, grid_alpha=0.4,
                draw_contours=draw_contours,
                contour_bins=contour_bins,
                contour_levels=contour_levels,
                contour_smoothing_sigma=contour_smoothing_sigma,
                contour_method=contour_method,
                contour_kde_bw_factor=contour_kde_bw_factor,
            )

        return self._save_and_show(
            fig, filename or "kr_map_distribution_comparison.png", show
        )

    def save_kr_map_distribution_shift_animation(
        self,
        normal_samples: np.ndarray,
        synthetic_samples: np.ndarray,
        mapped_samples_sequence: np.ndarray,
        solver_label: str,
        outer_indices: Sequence[int] | None = None,
        panel_titles: tuple[str, str, str] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        draw_contours: bool = True,
        contour_bins: int = 80,
        contour_levels: int | Sequence[float] = 6,
        contour_smoothing_sigma: float = 1.0,
        contour_method: str = "kde",
        contour_kde_bw_factor: float = 1.35,
        filename_prefix: str | None = None,
        fps: int = 12,
        save_mp4: bool = True,
        save_gif: bool = True,
    ) -> dict[str, str]:
        """Save mapped-distribution shift animation as MP4 and/or GIF."""
        mapped_sequence = np.asarray(mapped_samples_sequence, dtype=float)
        if mapped_sequence.ndim != 3 or mapped_sequence.shape[2] < 2:
            raise ValueError(
                "mapped_samples_sequence must have shape (num_frames, M, >=2)."
            )

        num_frames = int(mapped_sequence.shape[0])
        if num_frames < 1:
            raise ValueError("Animation requires at least one frame.")
        if not save_mp4 and not save_gif:
            raise ValueError("At least one of save_mp4 or save_gif must be True.")

        if outer_indices is None:
            resolved_outer_indices = list(range(num_frames))
        else:
            if len(outer_indices) != num_frames:
                raise ValueError(
                    "outer_indices must have the same length as mapped_samples_sequence."
                )
            resolved_outer_indices = [int(idx) for idx in outer_indices]

        if panel_titles is None:
            panel_titles = (
                r"Reference normal $\mathcal{N}(0, I_2)$",
                "Synthetic distribution",
                f"Mapped with {solver_label}",
            )

        prefix = (
            filename_prefix
            if filename_prefix is not None and filename_prefix.strip() != ""
            else "kr_map_distribution_shift"
        )
        effective_fps = max(int(fps), 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self._draw_distribution_panel(
            ax=axes[0],
            samples=np.asarray(normal_samples, dtype=float),
            title=panel_titles[0],
            xlabel="$x_1$",
            ylabel="$x_2$",
            style="reference",
            xlim=xlim,
            ylim=ylim,
            grid_alpha=0.4,
            draw_contours=draw_contours,
            contour_bins=contour_bins,
            contour_levels=contour_levels,
            contour_smoothing_sigma=contour_smoothing_sigma,
            contour_method=contour_method,
            contour_kde_bw_factor=contour_kde_bw_factor,
        )
        self._draw_distribution_panel(
            ax=axes[1],
            samples=np.asarray(synthetic_samples, dtype=float),
            title=panel_titles[1],
            xlabel="$x_1$",
            ylabel="$x_2$",
            style="sheared",
            xlim=xlim,
            ylim=ylim,
            grid_alpha=0.4,
            draw_contours=draw_contours,
            contour_bins=contour_bins,
            contour_levels=contour_levels,
            contour_smoothing_sigma=contour_smoothing_sigma,
            contour_method=contour_method,
            contour_kde_bw_factor=contour_kde_bw_factor,
        )

        mapped_ax = axes[2]
        mapped_style = self._resolve_style("mapped")
        mapped_scatter = mapped_ax.scatter(
            mapped_sequence[0, :, 0],
            mapped_sequence[0, :, 1],
            alpha=mapped_style.point_alpha,
            color=mapped_style.point_color,
            edgecolor=mapped_style.point_edge_color,
            s=mapped_style.point_size,
            zorder=3,
        )
        mapped_contours = None
        if draw_contours:
            xx, yy, zz = self._estimate_density_grid(
                samples=mapped_sequence[0, :, :2],
                xlim=xlim,
                ylim=ylim,
                bins=contour_bins,
                smooth_sigma=contour_smoothing_sigma,
                method=contour_method,
                kde_bw_factor=contour_kde_bw_factor,
            )
            mapped_contours = self._draw_density_contours(
                ax=mapped_ax,
                xx=xx,
                yy=yy,
                zz=zz,
                style=mapped_style,
                contour_levels=contour_levels,
            )

        def _title_for_frame(frame_idx: int) -> str:
            outer_idx = resolved_outer_indices[frame_idx]
            if outer_idx < 0:
                iter_label = "initial weights"
            else:
                iter_label = f"outer PGD iter {outer_idx}"
            return f"{panel_titles[2]} ({iter_label})"

        self._style_axis(
            ax=mapped_ax,
            title=_title_for_frame(0),
            xlabel="$x_1$",
            ylabel="$x_2$",
            xlim=xlim,
            ylim=ylim,
        )
        mapped_ax.grid(True, linestyle="--", alpha=0.4)
        mapped_ax.set_aspect("equal", adjustable="box")

        def _update(frame_idx: int) -> tuple[Any, ...]:
            nonlocal mapped_contours
            mapped_scatter.set_offsets(mapped_sequence[frame_idx, :, :2])
            if mapped_contours is not None:
                for collection in mapped_contours.collections:
                    collection.remove()
            if draw_contours:
                xx, yy, zz = self._estimate_density_grid(
                    samples=mapped_sequence[frame_idx, :, :2],
                    xlim=xlim,
                    ylim=ylim,
                    bins=contour_bins,
                    smooth_sigma=contour_smoothing_sigma,
                    method=contour_method,
                    kde_bw_factor=contour_kde_bw_factor,
                )
                mapped_contours = self._draw_density_contours(
                    ax=mapped_ax,
                    xx=xx,
                    yy=yy,
                    zz=zz,
                    style=mapped_style,
                    contour_levels=contour_levels,
                )
            mapped_ax.set_title(_title_for_frame(frame_idx), fontsize=TITLE_FONT_SIZE)
            return (mapped_scatter,)

        anim = animation.FuncAnimation(
            fig=fig,
            func=_update,
            frames=num_frames,
            interval=int(1000 / effective_fps),
            blit=False,
            repeat=True,
        )

        fig.tight_layout()
        saved_paths: dict[str, str] = {}
        mp4_error: str | None = None
        gif_error: str | None = None

        if save_mp4:
            if animation.writers.is_available("ffmpeg"):
                mp4_path = os.path.join(self.output_dir, f"{prefix}.mp4")
                try:
                    mp4_writer = animation.FFMpegWriter(
                        fps=effective_fps,
                        codec="h264",
                    )
                    anim.save(mp4_path, writer=mp4_writer, dpi=self.dpi)
                    saved_paths["mp4"] = mp4_path
                except Exception as exc:  # pragma: no cover - backend dependent
                    mp4_error = str(exc)
            else:
                mp4_error = "ffmpeg writer is not available in this environment."

        if save_gif:
            gif_path = os.path.join(self.output_dir, f"{prefix}.gif")
            try:
                gif_writer = animation.PillowWriter(
                    fps=effective_fps,
                    metadata={"loop": "0"},
                )
                anim.save(gif_path, writer=gif_writer, dpi=self.dpi)
                saved_paths["gif"] = gif_path
            except Exception as exc:  # pragma: no cover - backend dependent
                gif_error = str(exc)

        plt.close(fig)

        if not saved_paths:
            error_parts: list[str] = []
            if mp4_error is not None:
                error_parts.append(f"mp4: {mp4_error}")
            if gif_error is not None:
                error_parts.append(f"gif: {gif_error}")
            detail = "; ".join(error_parts) if error_parts else "unknown error"
            raise RuntimeError(f"Failed to save distribution shift animation ({detail}).")

        return saved_paths

    def plot_distributions(
        self,
        reference_samples: np.ndarray,
        sheared_samples: np.ndarray,
        seed: int,
        m: int,
        shear_label: str | None = None,
        reference_title: str | None = None,
        sheared_title: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        draw_contours: bool = True,
        contour_bins: int = 80,
        contour_levels: int | Sequence[float] = 6,
        contour_smoothing_sigma: float = 1.0,
        contour_method: str = "kde",
        contour_kde_bw_factor: float = 1.35,
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot and save standard-normal and sheared sample distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        effective_reference_title = reference_title or r"Standard normal $\mathcal{N}(0, I_2)$"
        effective_sheared_title = sheared_title or "Sheared distribution"
        filename_label = ""
        if (
            sheared_title is None
            and shear_label is not None
            and shear_label.strip() != ""
        ):
            cleaned_label = shear_label.strip()
            effective_sheared_title = f"Sheared distribution ({cleaned_label})"
            filename_label = f"_SHEAR={cleaned_label}"

        self._draw_distribution_panel(
            ax=ax1, samples=reference_samples,
            title=effective_reference_title,
            xlabel="$z_1$", ylabel="$z_2$", style="reference",
            xlim=xlim, ylim=ylim,
            draw_contours=draw_contours,
            contour_bins=contour_bins,
            contour_levels=contour_levels,
            contour_smoothing_sigma=contour_smoothing_sigma,
            contour_method=contour_method,
            contour_kde_bw_factor=contour_kde_bw_factor,
        )
        self._draw_distribution_panel(
            ax=ax2, samples=sheared_samples,
            title=effective_sheared_title,
            xlabel="$x_1$", ylabel="$x_2$", style="sheared",
            xlim=xlim, ylim=ylim,
            draw_contours=draw_contours,
            contour_bins=contour_bins,
            contour_levels=contour_levels,
            contour_smoothing_sigma=contour_smoothing_sigma,
            contour_method=contour_method,
            contour_kde_bw_factor=contour_kde_bw_factor,
        )

        default_filename = f"synthetic_distribution_SEED={seed}_M={m}{filename_label}.png"
        return self._save_and_show(
            fig, filename or default_filename, show
        )

    @staticmethod
    def _select_progress_indices(
        num_frames: int,
        num_panels: int,
        iteration_indices: Sequence[int] | None = None,
        emphasize_early: float = 2.0,
    ) -> list[int]:
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1.")

        if iteration_indices is not None:
            selected: list[int] = []
            seen: set[int] = set()
            for raw in iteration_indices:
                idx = int(raw)
                if idx < 0 or idx >= num_frames:
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                selected.append(idx)

            if 0 not in seen:
                selected.insert(0, 0)
                seen.add(0)
            final_idx = num_frames - 1
            if final_idx not in seen:
                selected.append(final_idx)
            if len(selected) < 2:
                return [0, final_idx]
            return selected

        if num_panels < 2:
            raise ValueError("num_panels must be at least 2 when auto-selecting indices.")

        warp = max(float(emphasize_early), 1e-6)
        # Blend linear and early-biased spacing with a conservative cap so
        # auto mode remains well distributed across the full trajectory.
        blend = min(max((warp - 1.0) / max(warp, 1.0), 0.0), 0.25)
        candidate_u = np.linspace(0.0, 1.0, max(num_panels * 20, num_panels))
        warped_u = candidate_u**warp
        mixed_u = (1.0 - blend) * candidate_u + blend * warped_u
        candidate_idx = np.round(mixed_u * (num_frames - 1)).astype(int)
        unique_sorted = np.unique(candidate_idx)
        selected_auto = [int(v) for v in unique_sorted.tolist()]

        if 0 not in selected_auto:
            selected_auto.insert(0, 0)
        if (num_frames - 1) not in selected_auto:
            selected_auto.append(num_frames - 1)

        if len(selected_auto) < num_panels:
            fill = np.round(np.linspace(0, num_frames - 1, num_panels)).astype(int)
            for idx in fill.tolist():
                if idx not in selected_auto:
                    selected_auto.append(int(idx))
            selected_auto = sorted(set(selected_auto))

        if len(selected_auto) > num_panels:
            trimmed = selected_auto[: max(num_panels - 1, 1)]
            if selected_auto[-1] not in trimmed:
                trimmed.append(selected_auto[-1])
            selected_auto = trimmed

        if selected_auto[0] != 0:
            selected_auto[0] = 0
        if selected_auto[-1] != num_frames - 1:
            selected_auto[-1] = num_frames - 1
        return selected_auto

    def plot_mapped_progress_grid(
        self,
        mapped_samples_sequence: np.ndarray,
        solver_label: str,
        outer_indices: Sequence[int] | None = None,
        iteration_indices: Sequence[int] | None = None,
        num_panels: int = 12,
        ncols: int = 3,
        emphasize_early: float = 2.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        draw_contours: bool = True,
        contour_bins: int = 80,
        contour_levels: int | Sequence[float] = 6,
        contour_smoothing_sigma: float = 1.0,
        contour_method: str = "kde",
        contour_kde_bw_factor: float = 1.35,
        panel_title_template: str = "PGD iteration: {iter}",
        filename: str | None = None,
        show: bool = True,
    ) -> Figure:
        """Plot mapped-distribution progress on a configurable grid.

        If ``iteration_indices`` is provided, those indices are used (with initial
        and final frames automatically enforced). Otherwise indices are sampled
        automatically with denser coverage near the beginning of the run.
        """
        mapped_sequence = np.asarray(mapped_samples_sequence, dtype=float)
        if mapped_sequence.ndim != 3 or mapped_sequence.shape[2] < 2:
            raise ValueError("mapped_samples_sequence must have shape (num_frames, M, >=2).")

        num_frames = int(mapped_sequence.shape[0])
        selected_indices = self._select_progress_indices(
            num_frames=num_frames,
            num_panels=int(num_panels),
            iteration_indices=iteration_indices,
            emphasize_early=emphasize_early,
        )

        if outer_indices is not None and len(outer_indices) != num_frames:
            raise ValueError(
                "outer_indices must have the same length as mapped_samples_sequence."
            )

        n_panels = len(selected_indices)
        ncols_eff = max(int(ncols), 1)
        nrows = int(math.ceil(n_panels / ncols_eff))
        fig, axes = plt.subplots(nrows, ncols_eff, figsize=(5.0 * ncols_eff, 4.2 * nrows))
        axes_array = np.atleast_1d(axes).reshape(nrows, ncols_eff)

        for panel_idx, frame_idx in enumerate(selected_indices):
            row = panel_idx // ncols_eff
            col = panel_idx % ncols_eff
            ax = axes_array[row, col]
            outer_iter_label = (
                int(outer_indices[frame_idx]) if outer_indices is not None else int(frame_idx)
            )
            display_iter = outer_iter_label + 1
            panel_title = panel_title_template.format(
                iter=display_iter,
                frame=frame_idx,
                solver=solver_label,
            )
            self._draw_distribution_panel(
                ax=ax,
                samples=mapped_sequence[frame_idx, :, :2],
                title=panel_title,
                xlabel="$x_1$",
                ylabel="$x_2$",
                style="mapped",
                xlim=xlim,
                ylim=ylim,
                grid_alpha=0.4,
                draw_contours=draw_contours,
                contour_bins=contour_bins,
                contour_levels=contour_levels,
                contour_smoothing_sigma=contour_smoothing_sigma,
                contour_method=contour_method,
                contour_kde_bw_factor=contour_kde_bw_factor,
            )
            if col != 0:
                ax.set_ylabel("")
            if row != (nrows - 1):
                ax.set_xlabel("")

        for panel_idx in range(n_panels, nrows * ncols_eff):
            row = panel_idx // ncols_eff
            col = panel_idx % ncols_eff
            axes_array[row, col].axis("off")

        fig.subplots_adjust(wspace=0.08, hspace=0.22)
        return self._save_and_show(
            fig,
            filename or f"kr_map_progress_{solver_label.replace(' ', '_').lower()}.png",
            show,
        )

    def _draw_distribution_panel(
        self,
        ax: Axes,
        samples: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        style: str | DistributionStyle | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        grid_alpha: float = 0.6,
        draw_contours: bool = True,
        contour_bins: int = 80,
        contour_levels: int | Sequence[float] | None = None,
        contour_smoothing_sigma: float = 1.0,
        contour_method: str = "kde",
        contour_kde_bw_factor: float = 1.35,
    ) -> None:
        resolved_style = self._resolve_style(style=style, fallback_key="reference")
        samples_array = np.asarray(samples, dtype=float)
        if draw_contours:
            xx, yy, zz = self._estimate_density_grid(
                samples=samples_array[:, :2],
                xlim=xlim,
                ylim=ylim,
                bins=contour_bins,
                smooth_sigma=contour_smoothing_sigma,
                method=contour_method,
                kde_bw_factor=contour_kde_bw_factor,
            )
            self._draw_density_contours(
                ax=ax,
                xx=xx,
                yy=yy,
                zz=zz,
                style=resolved_style,
                contour_levels=contour_levels,
            )
        ax.scatter(
            samples_array[:, 0],
            samples_array[:, 1],
            alpha=resolved_style.point_alpha,
            color=resolved_style.point_color,
            edgecolor=resolved_style.point_edge_color,
            s=resolved_style.point_size,
            zorder=3,
        )

        self._style_axis(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, linestyle="--", alpha=grid_alpha)
        ax.set_aspect("equal", adjustable="box")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)


class BenchmarkPlotter(_BasePlotter):
    """Plotter for benchmark-level summaries (e.g., runtime scaling curves)."""

    def plot_runtime_scaling(
        self,
        aggregated_rows: Sequence[dict[str, Any]],
        y_key_mean: str,
        y_key_std: str,
        title: str,
        y_label: str,
        filename: str,
        show: bool = False,
    ) -> Figure:
        """Plot runtime-vs-dimension curves with per-solver error bars.

        Uses shared global plot styling via ``_style_axis``.
        """
        by_solver: dict[str, list[dict[str, Any]]] = {}
        for row in aggregated_rows:
            solver_label = str(row["solver"])
            by_solver.setdefault(solver_label, []).append(row)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for solver_label in sorted(by_solver.keys()):
            rows = sorted(by_solver[solver_label], key=lambda r: int(r["num_dimensions"]))
            dims = [int(r["num_dimensions"]) for r in rows]
            means = [float(r[y_key_mean]) for r in rows]
            stds = [float(r[y_key_std]) for r in rows]
            ax.errorbar(
                dims,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.2,
                markersize=6,
                capsize=4,
                label=solver_label,
            )

        self._style_axis(
            ax=ax,
            title="",
            xlabel="Problem dimension",
            ylabel=y_label,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE)

        return self._save_and_show(fig, filename=filename, show=show)

