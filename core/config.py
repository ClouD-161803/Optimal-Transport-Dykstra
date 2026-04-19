"""Typed configuration objects for experiments."""

from dataclasses import dataclass, field
from typing import Any, Literal

SolverMode = Literal["both", "vanilla", "fast", "benchmark"]


@dataclass(frozen=True)
class RunConfig:
    """Execution-mode settings for one experiment run."""

    run_solver_mode: SolverMode = "both"
    save_full_run_iterates: bool = False
    save_distribution_shift_media: bool = False
    enforce_matching: bool = False


@dataclass(frozen=True)
class PlotConfig:
    """Plotting flags and axis settings."""

    plot_dykstra_iterates: bool = False
    plot_dykstra_outer_iterations: list[int] | None = None
    plot_distributions: bool = True
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    distribution_panel_titles_both: tuple[str, str, str, str] | None = None
    distribution_panel_titles_vanilla: tuple[str, str, str] | None = None
    distribution_panel_titles_fast: tuple[str, str, str] | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimisation and projection hyperparameters."""

    learning_rate: float
    max_outer_iter: int
    gradient_clip_value: float | None
    l1_reg: float
    lr_decay: float
    inexact_power: float
    base_inner_iter: int
    max_inner_iters: int
    batch_size: int | None
    rng_seed: int | None
    prune_threshold: float
    prune_interval: int
    dykstra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration for an experiment."""

    seed: int
    num_dimensions: int
    num_particles: int
    run: RunConfig
    plot: PlotConfig
    optimization: OptimizationConfig
