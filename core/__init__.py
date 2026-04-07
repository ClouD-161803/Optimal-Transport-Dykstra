"""Core experiment orchestration package (flat layout)."""

from .config import ExperimentConfig, OptimizationConfig, PlotConfig, RunConfig, SolverMode
from .data import DataBatch, DataSource, DatasetDataSource, SyntheticDataSource
from .io import (
    build_distribution_filename,
    plot_component_solver_comparison,
    plot_distribution_for_mode,
    save_full_run_iterates_json,
    save_full_run_iterates_npz,
    to_json_safe,
)
from .runner import (
    ExperimentRunSummary,
    ExperimentRunner,
    build_identity_initial_guesses,
    run_component_benchmark,
    run_dataset_experiment,
    run_synthetic_experiment,
)

__all__ = [
    "ExperimentConfig",
    "OptimizationConfig",
    "PlotConfig",
    "RunConfig",
    "SolverMode",
    "DataBatch",
    "DataSource",
    "DatasetDataSource",
    "SyntheticDataSource",
    "build_distribution_filename",
    "plot_component_solver_comparison",
    "plot_distribution_for_mode",
    "save_full_run_iterates_json",
    "save_full_run_iterates_npz",
    "to_json_safe",
    "ExperimentRunSummary",
    "ExperimentRunner",
    "build_identity_initial_guesses",
    "run_component_benchmark",
    "run_dataset_experiment",
    "run_synthetic_experiment",
]
