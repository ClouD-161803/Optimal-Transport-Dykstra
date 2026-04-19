"""Plotting utilities for the Dykstra projection project.

Provides dedicated plotter classes for disjoint plotting domains:

* ``DykstraPlotter`` for Dykstra/PGD convergence diagnostics.
* ``DistributionPlotter`` for sample-distribution visualisations.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from .projection_result import ProjectionResult

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16


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

        self._style_axis(ax=ax, title=title, xlabel="Iteration", ylabel="Squared error")
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

    def plot_kr_map_distribution_single_solver(
        self,
        normal_samples: np.ndarray,
        synthetic_samples: np.ndarray,
        mapped_samples: np.ndarray,
        solver_label: str,
        panel_titles: tuple[str, str, str] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
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
            (normal_samples, panel_titles[0], "tab:blue"),
            (synthetic_samples, panel_titles[1], "tab:red"),
            (mapped_samples, panel_titles[2], "tab:green"),
        ]

        for ax, (samples, title, color) in zip(axes, panels):
            self._draw_distribution_panel(
                ax=ax, samples=samples, title=title,
                xlabel="$x_1$", ylabel="$x_2$", color=color,
                s=16, xlim=xlim, ylim=ylim, grid_alpha=0.4,
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
            (normal_samples, panel_titles[0], "tab:blue"),
            (synthetic_samples, panel_titles[1], "tab:red"),
            (vanilla_mapped_samples, panel_titles[2], "tab:green"),
            (fast_mapped_samples, panel_titles[3], "tab:purple"),
        ]

        for ax, (samples, title, color) in zip(axes.flatten(), panels):
            self._draw_distribution_panel(
                ax=ax, samples=samples, title=title,
                xlabel="$x_1$", ylabel="$x_2$", color=color,
                s=16, xlim=xlim, ylim=ylim, grid_alpha=0.4,
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
            color="tab:blue",
            s=16,
            xlim=xlim,
            ylim=ylim,
            grid_alpha=0.4,
        )
        self._draw_distribution_panel(
            ax=axes[1],
            samples=np.asarray(synthetic_samples, dtype=float),
            title=panel_titles[1],
            xlabel="$x_1$",
            ylabel="$x_2$",
            color="tab:red",
            s=16,
            xlim=xlim,
            ylim=ylim,
            grid_alpha=0.4,
        )

        mapped_ax = axes[2]
        mapped_scatter = mapped_ax.scatter(
            mapped_sequence[0, :, 0],
            mapped_sequence[0, :, 1],
            alpha=0.5,
            color="tab:green",
            edgecolor="k",
            s=16,
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
            mapped_scatter.set_offsets(mapped_sequence[frame_idx, :, :2])
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
            xlabel="$z_1$", ylabel="$z_2$", color="blue",
            xlim=xlim, ylim=ylim,
        )
        self._draw_distribution_panel(
            ax=ax2, samples=sheared_samples,
            title=effective_sheared_title,
            xlabel="$x_1$", ylabel="$x_2$", color="red",
            xlim=xlim, ylim=ylim,
        )

        default_filename = f"synthetic_distribution_SEED={seed}_M={m}{filename_label}.png"
        return self._save_and_show(
            fig, filename or default_filename, show
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
            title=title,
            xlabel="Problem dimension",
            ylabel=y_label,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE)

        return self._save_and_show(fig, filename=filename, show=show)

