"""Experiment execution core and workflow entrypoints."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import ExperimentConfig, OptimizationConfig, SolverMode
from core.data import DatasetDataSource, SyntheticDataSource
from core.io import (
    plot_component_solver_comparison,
    plot_distribution_for_mode,
    save_full_run_iterates_json,
    save_full_run_iterates_npz,
)
from utils import DykstraProjectionSolver, DykstraStallDetectionSolver
from utils import KRMap, ProjectedGradientDescent


@dataclass(frozen=True)
class SolverVariant:
    """One configured projection-solver variant."""

    label: str
    projection_solver_class: type
    delete_spaces: bool = False


def resolve_solver_variants(run_solver_mode: SolverMode) -> list[SolverVariant]:
    """Resolve active solver variants from run mode."""
    if run_solver_mode == "both":
        return [
            SolverVariant("vanilla", DykstraProjectionSolver, delete_spaces=False),
            SolverVariant("fast", DykstraStallDetectionSolver, delete_spaces=True),
        ]
    if run_solver_mode == "vanilla":
        return [
            SolverVariant("vanilla", DykstraProjectionSolver, delete_spaces=False),
        ]
    if run_solver_mode == "fast":
        return [
            SolverVariant("fast", DykstraStallDetectionSolver, delete_spaces=True),
        ]
    raise ValueError("run_solver_mode must be one of: 'both', 'vanilla', 'fast'.")


def build_pgd_solver(
    variant: SolverVariant,
    optimization_config: OptimizationConfig,
    track_error_outer_iterations: list[int] | None = None,
    store_all_projection_results: bool = False,
    dykstra_kwargs_override: dict[str, Any] | None = None,
) -> ProjectedGradientDescent:
    """Build a configured PGD solver for one solver variant."""
    dykstra_kwargs = dict(optimization_config.dykstra_kwargs)
    if dykstra_kwargs_override is not None:
        dykstra_kwargs.update(dykstra_kwargs_override)

    solver_kwargs: dict[str, Any] = {
        "learning_rate": optimization_config.learning_rate,
        "max_outer_iter": optimization_config.max_outer_iter,
        "projection_solver_class": variant.projection_solver_class,
        "gradient_clip_value": optimization_config.gradient_clip_value,
        "l1_reg": optimization_config.l1_reg,
        "lr_decay": optimization_config.lr_decay,
        "inexact_power": optimization_config.inexact_power,
        "base_inner_iter": optimization_config.base_inner_iter,
        "batch_size": optimization_config.batch_size,
        "rng_seed": optimization_config.rng_seed,
        "prune_threshold": optimization_config.prune_threshold,
        "prune_interval": optimization_config.prune_interval,
        "track_error_outer_iterations": track_error_outer_iterations,
        "store_all_projection_results": store_all_projection_results,
        **dykstra_kwargs,
    }
    if variant.delete_spaces:
        solver_kwargs["delete_spaces"] = True
    return ProjectedGradientDescent(**solver_kwargs)


def _run_component_optimisation(
    pgd_solver: ProjectedGradientDescent,
    component_w_init: np.ndarray,
    kr_model: Any,
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Run one component PGD optimisation and return weights, history, and runtime."""
    t0 = time.perf_counter()
    weights, history = pgd_solver.optimise(
        w_init=component_w_init,
        objective_fn=kr_model.objective,
        gradient_fn=kr_model.gradient,
        A_constraint=A,
        b_constraint=b,
        gradient_batch_fn=kr_model.gradient_batch,
    )
    elapsed = time.perf_counter() - t0
    return weights, history, elapsed


def _append_solver_result(
    component_result: dict[str, Any],
    solver_label: str,
    weights: np.ndarray,
    history: dict[str, Any],
    elapsed: float,
) -> None:
    component_result[f"w_{solver_label}"] = weights
    component_result[f"time_{solver_label}"] = elapsed
    component_result[f"objective_{solver_label}"] = history["objective_value"][-1]
    component_result[f"history_{solver_label}"] = history


def _print_component_timing(
    component_dim: int,
    num_dimensions: int,
    component_result: dict[str, Any],
    run_solver_mode: SolverMode,
) -> None:
    time_vanilla = component_result.get("time_vanilla")
    time_fast = component_result.get("time_fast")
    coeff_close = component_result.get("coefficients_close")

    if run_solver_mode == "both" and time_vanilla is not None and time_fast is not None:
        print(
            f"[Component {component_dim}/{num_dimensions}] "
            f"vanilla={time_vanilla:.4f}s, fast={time_fast:.4f}s, "
            f"coeff_close={coeff_close}"
        )
    elif run_solver_mode == "vanilla" and time_vanilla is not None:
        print(f"[Component {component_dim}/{num_dimensions}] vanilla={time_vanilla:.4f}s")
    elif run_solver_mode == "fast" and time_fast is not None:
        print(f"[Component {component_dim}/{num_dimensions}] fast={time_fast:.4f}s")


def run_component_benchmark(
    z: np.ndarray,
    num_dimensions: int,
    num_particles: int,
    seed: int,
    kr_map: KRMap,
    initial_guesses_by_component: dict[int, np.ndarray],
    optimization_config: OptimizationConfig,
    run_solver_mode: SolverMode,
    plot_dykstra_iterates: bool,
    plot_outer_iterations: list[int] | None = None,
    enforce_matching: bool = False,
    store_full_projection_histories: bool = False,
    dykstra_plot_output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Benchmark each KR map component up to ``num_dimensions``."""
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must have shape (M, num_dimensions).")
    if z.shape[1] < num_dimensions:
        raise ValueError("z has fewer columns than num_dimensions.")

    solver_variants = resolve_solver_variants(run_solver_mode)
    capture_full_histories = bool(store_full_projection_histories)
    dykstra_kwargs_override = {"track_error": True} if capture_full_histories else None

    component_results: list[dict[str, Any]] = []

    for component_dim in range(1, num_dimensions + 1):
        component_data = z[:, :component_dim]
        kr_model = kr_map.make_component(component_data)
        A, b = kr_model.get_polyhedral_constraints(epsilon=1e-4)

        if component_dim not in initial_guesses_by_component:
            raise ValueError(
                f"Missing initial guess for component dimension {component_dim}."
            )

        component_w_init = np.asarray(
            initial_guesses_by_component[component_dim],
            dtype=float,
        ).reshape(-1)
        if component_w_init.size != kr_model.num_coefficients:
            raise ValueError(
                f"Initial guess size mismatch for component {component_dim}: "
                f"expected {kr_model.num_coefficients}, got {component_w_init.size}."
            )

        component_result: dict[str, Any] = {
            "component_dim": component_dim,
            "coefficients_close": None,
            "coefficients_max_abs_diff": None,
        }

        for variant in solver_variants:
            pgd_solver = build_pgd_solver(
                variant=variant,
                optimization_config=optimization_config,
                track_error_outer_iterations=(
                    plot_outer_iterations if plot_dykstra_iterates else None
                ),
                store_all_projection_results=capture_full_histories,
                dykstra_kwargs_override=dykstra_kwargs_override,
            )
            weights, history, elapsed = _run_component_optimisation(
                pgd_solver=pgd_solver,
                component_w_init=component_w_init,
                kr_model=kr_model,
                A=A,
                b=b,
            )
            _append_solver_result(
                component_result=component_result,
                solver_label=variant.label,
                weights=weights,
                history=history,
                elapsed=elapsed,
            )

        w_vanilla = component_result.get("w_vanilla")
        w_fast = component_result.get("w_fast")
        if w_vanilla is not None and w_fast is not None:
            coeff_close = bool(np.allclose(w_vanilla, w_fast, atol=1e-4))
            coeff_max_abs_diff = float(np.max(np.abs(w_vanilla - w_fast)))
            component_result["coefficients_close"] = coeff_close
            component_result["coefficients_max_abs_diff"] = coeff_max_abs_diff

            if enforce_matching:
                np.testing.assert_allclose(
                    w_vanilla,
                    w_fast,
                    atol=1e-4,
                    err_msg=(
                        "Vanilla and fast-forward Dykstra produced different coefficients "
                        f"for component dimension {component_dim}."
                    ),
                )

        if (
            plot_dykstra_iterates
            and run_solver_mode == "both"
            and "history_vanilla" in component_result
            and "history_fast" in component_result
        ):
            if dykstra_plot_output_dir is None:
                raise ValueError(
                    "dykstra_plot_output_dir must be provided when plotting iterates."
                )
            filename_prefix = (
                f"kr{num_dimensions}d_component_{component_dim}_"
                f"SEED={seed}_M={num_particles}"
            )
            plot_component_solver_comparison(
                output_dir=dykstra_plot_output_dir,
                filename_prefix=filename_prefix,
                vanilla_history=component_result["history_vanilla"],
                fast_history=component_result["history_fast"],
            )

        component_results.append(component_result)
        _print_component_timing(
            component_dim=component_dim,
            num_dimensions=num_dimensions,
            component_result=component_result,
            run_solver_mode=run_solver_mode,
        )

    return component_results


@dataclass(frozen=True)
class ExperimentRunSummary:
    """Outputs returned from a complete experiment run."""

    results: list[dict[str, Any]]
    full_run_npz_path: str | None = None
    full_run_json_path: str | None = None


class ExperimentRunner:
    """Run a configured experiment from data source to artifacts/plots."""

    def __init__(
        self,
        project_root: str,
        config: ExperimentConfig,
        data_source: Any,
        kr_map: KRMap,
        initial_guesses_by_component: dict[int, np.ndarray],
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.data_source = data_source
        self.kr_map = kr_map
        self.initial_guesses_by_component = initial_guesses_by_component

    def _resolve_results_dir(self, *segments: str) -> str:
        return os.path.join(self.project_root, "results", *segments)

    def _validate(self) -> None:
        solver_mode = self.config.run.run_solver_mode
        if solver_mode not in {"both", "vanilla", "fast"}:
            raise ValueError("RUN_SOLVER_MODE must be one of: 'both', 'vanilla', 'fast'.")
        if solver_mode != "both" and self.config.plot.plot_dykstra_iterates:
            raise ValueError(
                "PLOT_DYKSTRA_ITERATES=True is only valid when RUN_SOLVER_MODE='both'."
            )

    def run(self) -> ExperimentRunSummary:
        self._validate()

        batch = self.data_source.load(
            num_particles=self.config.num_particles,
            num_dimensions=self.config.num_dimensions,
            seed=self.config.seed,
        )

        component_results = run_component_benchmark(
            z=batch.target_samples,
            num_dimensions=self.config.num_dimensions,
            num_particles=self.config.num_particles,
            seed=self.config.seed,
            kr_map=self.kr_map,
            initial_guesses_by_component=self.initial_guesses_by_component,
            optimization_config=self.config.optimization,
            run_solver_mode=self.config.run.run_solver_mode,
            plot_dykstra_iterates=self.config.plot.plot_dykstra_iterates,
            plot_outer_iterations=self.config.plot.plot_dykstra_outer_iterations,
            enforce_matching=self.config.run.enforce_matching,
            store_full_projection_histories=self.config.run.save_full_run_iterates,
            dykstra_plot_output_dir=self._resolve_results_dir("dykstra_benchmarks"),
        )

        full_run_npz_path: str | None = None
        full_run_json_path: str | None = None
        if self.config.run.save_full_run_iterates:
            full_run_output_dir = self._resolve_results_dir(
                "full_experiment_benchmarks",
                "full_run_iterates",
            )
            full_run_npz_path, npz_component_index = save_full_run_iterates_npz(
                results=component_results,
                output_dir=full_run_output_dir,
                num_dimensions=self.config.num_dimensions,
                num_particles=self.config.num_particles,
                seed=self.config.seed,
                max_outer_iter=self.config.optimization.max_outer_iter,
                base_inner_iter=self.config.optimization.base_inner_iter,
                max_inner_iters=self.config.optimization.max_inner_iters,
                solver_mode=self.config.run.run_solver_mode,
            )
            full_run_json_path = save_full_run_iterates_json(
                results=component_results,
                solver_mode=self.config.run.run_solver_mode,
                output_dir=full_run_output_dir,
                num_dimensions=self.config.num_dimensions,
                num_particles=self.config.num_particles,
                seed=self.config.seed,
                max_outer_iter=self.config.optimization.max_outer_iter,
                base_inner_iter=self.config.optimization.base_inner_iter,
                max_inner_iters=self.config.optimization.max_inner_iters,
                batch_size=self.config.optimization.batch_size,
                learning_rate=self.config.optimization.learning_rate,
                lr_decay=self.config.optimization.lr_decay,
                l1_reg=self.config.optimization.l1_reg,
                prune_interval=self.config.optimization.prune_interval,
                prune_threshold=self.config.optimization.prune_threshold,
                dykstra_kwargs=dict(self.config.optimization.dykstra_kwargs),
                npz_path=full_run_npz_path,
                npz_component_index=npz_component_index,
            )

        if self.config.plot.plot_distributions:
            plot_distribution_for_mode(
                solver_mode=self.config.run.run_solver_mode,
                output_dir=self._resolve_results_dir("full_experiment_benchmarks"),
                normal_samples=batch.reference_samples,
                z_samples=batch.target_samples,
                results=component_results,
                kr_map=self.kr_map,
                num_dimensions=self.config.num_dimensions,
                seed=self.config.seed,
                num_particles=self.config.num_particles,
                optimization_config=self.config.optimization,
                x_lim=self.config.plot.x_lim,
                y_lim=self.config.plot.y_lim,
            )

        print(
            f"\nCompleted {self.config.num_dimensions}-dimensional KR component benchmark "
            f"with seed {self.config.seed}."
        )
        num_component_figures = (
            len(component_results)
            if (
                self.config.plot.plot_dykstra_iterates
                and self.config.run.run_solver_mode == "both"
            )
            else 0
        )
        num_distribution_figures = 1 if self.config.plot.plot_distributions else 0
        print(
            "Saved "
            f"{num_component_figures} component error figure(s) and "
            f"{num_distribution_figures} distribution comparison figure(s) "
            "in results/full_experiment_benchmarks."
        )
        if full_run_npz_path is not None:
            print(f"Saved full iterate NPZ: {full_run_npz_path}")
        if full_run_json_path is not None:
            print(f"Saved full iterate JSON: {full_run_json_path}")

        print("\nMap weights:")
        for result in component_results:
            dim = result["component_dim"]
            if "w_vanilla" in result:
                print(f"  Component {dim} (vanilla): {result['w_vanilla']}")
            if "w_fast" in result:
                print(f"  Component {dim} (fast):    {result['w_fast']}")

        return ExperimentRunSummary(
            results=component_results,
            full_run_npz_path=full_run_npz_path,
            full_run_json_path=full_run_json_path,
        )


def build_identity_initial_guesses(
    kr_map: KRMap,
    num_dimensions: int,
) -> dict[int, np.ndarray]:
    """Build identity-map initial guesses per KR component."""
    initial_guesses: dict[int, np.ndarray] = {}
    for component_dim in range(1, num_dimensions + 1):
        initial_guesses[component_dim] = kr_map.build_identity_initial_guess(
            component_dim
        )
    return initial_guesses


def run_synthetic_experiment(
    project_root: str,
    config: ExperimentConfig,
    generator: Any,
    kr_map: KRMap,
    initial_guesses_by_component: dict[int, np.ndarray],
) -> ExperimentRunSummary:
    """Run a full KR experiment using synthetic data generation."""
    data_source = SyntheticDataSource(generator=generator)
    runner = ExperimentRunner(
        project_root=project_root,
        config=config,
        data_source=data_source,
        kr_map=kr_map,
        initial_guesses_by_component=initial_guesses_by_component,
    )
    return runner.run()


def run_dataset_experiment(
    project_root: str,
    config: ExperimentConfig,
    kr_map: KRMap,
    initial_guesses_by_component: dict[int, np.ndarray],
) -> ExperimentRunSummary:
    """Run the dataset pipeline once parser support is implemented."""
    runner = ExperimentRunner(
        project_root=project_root,
        config=config,
        data_source=DatasetDataSource(),
        kr_map=kr_map,
        initial_guesses_by_component=initial_guesses_by_component,
    )
    return runner.run()

