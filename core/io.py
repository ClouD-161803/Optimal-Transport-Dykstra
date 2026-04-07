"""Artifact-writing and plotting helpers for experiment runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from core.config import OptimizationConfig, SolverMode
from utils import DistributionPlotter, DykstraPlotter, KRMap


def to_json_safe(value: Any) -> Any:
    """Convert NumPy/Python containers into JSON-safe objects."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def save_full_run_iterates_npz(
    results: list[dict[str, Any]],
    output_dir: str,
    num_dimensions: int,
    num_particles: int,
    seed: int,
    max_outer_iter: int,
    base_inner_iter: int,
    max_inner_iters: int,
    solver_mode: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Write full solver trajectories to compressed NPZ and return index metadata."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = (
        f"kr{num_dimensions}d_full_run_iterates_"
        f"SEED={seed}_M={num_particles}_PGDITERS={max_outer_iter}_"
        f"DYKSTRA_ITERS={base_inner_iter}_{max_inner_iters}_MODE={solver_mode}_TS={timestamp}.npz"
    )
    output_path = os.path.join(output_dir, filename)

    arrays_to_save: dict[str, np.ndarray] = {}
    component_index: list[dict[str, Any]] = []

    for component_result in results:
        component_dim = int(component_result["component_dim"])
        component_meta: dict[str, Any] = {"component_dim": component_dim, "solvers": {}}

        for solver_label in ("vanilla", "fast"):
            history_key = f"history_{solver_label}"
            weights_key = f"w_{solver_label}"
            if history_key not in component_result or weights_key not in component_result:
                continue

            history = component_result[history_key]
            prefix = f"comp{component_dim}_{solver_label}"

            arrays_to_save[f"{prefix}_weights"] = np.asarray(
                component_result[weights_key], dtype=float
            )
            arrays_to_save[f"{prefix}_objective_value"] = np.asarray(
                history.get("objective_value", []), dtype=float
            )
            arrays_to_save[f"{prefix}_dykstra_inner_iters"] = np.asarray(
                history.get("dykstra_inner_iters", []), dtype=int
            )

            solver_meta: dict[str, Any] = {
                "weights_key": f"{prefix}_weights",
                "objective_value_key": f"{prefix}_objective_value",
                "dykstra_inner_iters_key": f"{prefix}_dykstra_inner_iters",
            }

            for proj_key in ("projection_results", "projection_results_full"):
                if proj_key not in history:
                    continue

                proj_values = history[proj_key]
                projections = np.asarray(
                    [getattr(result, "projection", None) for result in proj_values],
                    dtype=object,
                )
                squared_errors = np.asarray(
                    [getattr(result, "squared_errors", None) for result in proj_values],
                    dtype=object,
                )
                stalled_errors = np.asarray(
                    [getattr(result, "stalled_errors", None) for result in proj_values],
                    dtype=object,
                )
                converged_errors = np.asarray(
                    [getattr(result, "converged_errors", None) for result in proj_values],
                    dtype=object,
                )
                active_half_spaces = np.asarray(
                    [getattr(result, "active_half_spaces", None) for result in proj_values],
                    dtype=object,
                )

                arrays_to_save[f"{prefix}_{proj_key}_projection"] = projections
                arrays_to_save[f"{prefix}_{proj_key}_squared_errors"] = squared_errors
                arrays_to_save[f"{prefix}_{proj_key}_stalled_errors"] = stalled_errors
                arrays_to_save[f"{prefix}_{proj_key}_converged_errors"] = converged_errors
                arrays_to_save[f"{prefix}_{proj_key}_active_half_spaces"] = active_half_spaces

                solver_meta[f"{proj_key}_keys"] = {
                    "projection": f"{prefix}_{proj_key}_projection",
                    "squared_errors": f"{prefix}_{proj_key}_squared_errors",
                    "stalled_errors": f"{prefix}_{proj_key}_stalled_errors",
                    "converged_errors": f"{prefix}_{proj_key}_converged_errors",
                    "active_half_spaces": f"{prefix}_{proj_key}_active_half_spaces",
                }

            outer_idx_key = "projection_outer_indices"
            if outer_idx_key in history:
                outer_idx_arr = np.asarray(history[outer_idx_key], dtype=int)
                arrays_to_save[f"{prefix}_{outer_idx_key}"] = outer_idx_arr
                solver_meta[f"{outer_idx_key}_key"] = f"{prefix}_{outer_idx_key}"

            outer_idx_full_key = "projection_outer_indices_full"
            if outer_idx_full_key in history:
                outer_idx_full_arr = np.asarray(history[outer_idx_full_key], dtype=int)
                arrays_to_save[f"{prefix}_{outer_idx_full_key}"] = outer_idx_full_arr
                solver_meta[f"{outer_idx_full_key}_key"] = f"{prefix}_{outer_idx_full_key}"

            component_meta["solvers"][solver_label] = solver_meta

        component_index.append(component_meta)

    np.savez_compressed(output_path, **arrays_to_save)  # type: ignore[arg-type]
    return output_path, component_index


def save_full_run_iterates_json(
    results: list[dict[str, Any]],
    solver_mode: str,
    output_dir: str,
    num_dimensions: int,
    num_particles: int,
    seed: int,
    max_outer_iter: int,
    base_inner_iter: int,
    max_inner_iters: int,
    batch_size: int | None,
    learning_rate: float,
    lr_decay: float,
    l1_reg: float,
    prune_interval: int,
    prune_threshold: float,
    dykstra_kwargs: dict[str, Any],
    npz_path: str,
    npz_component_index: list[dict[str, Any]],
) -> str:
    """Write lightweight JSON metadata and pointers to NPZ trajectory arrays."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = (
        f"kr{num_dimensions}d_full_run_iterates_"
        f"SEED={seed}_M={num_particles}_PGDITERS={max_outer_iter}_"
        f"DYKSTRA_ITERS={base_inner_iter}_{max_inner_iters}_TS={timestamp}.json"
    )
    output_path = os.path.join(output_dir, filename)

    payload: dict[str, Any] = {
        "metadata": {
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "solver_mode": solver_mode,
            "num_dimensions": num_dimensions,
            "num_particles": num_particles,
            "seed": seed,
            "max_outer_iter": max_outer_iter,
            "base_inner_iter": base_inner_iter,
            "max_inner_iters": max_inner_iters,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lr_decay": lr_decay,
            "l1_reg": l1_reg,
            "prune_interval": prune_interval,
            "prune_threshold": prune_threshold,
            "dykstra_kwargs": to_json_safe(dykstra_kwargs),
            "npz_pointer": {
                "absolute_path": npz_path,
                "relative_to_json_dir": os.path.relpath(npz_path, output_dir),
                "format": "npz",
                "compression": "zip",
                "allow_pickle_required": True,
            },
            "notes": {
                "json_payload": (
                    "Contains run metadata and compact per-component summaries only. "
                    "Large trajectory arrays are stored in the NPZ artifact."
                ),
                "npz_component_index": (
                    "Maps each component/solver to NPZ array keys for weights, "
                    "objectives, Dykstra iterates, and projection trajectories."
                ),
            },
        },
        "components": [
            {
                "component_dim": int(component_result["component_dim"]),
                "coefficients_close": to_json_safe(
                    component_result.get("coefficients_close")
                ),
                "coefficients_max_abs_diff": to_json_safe(
                    component_result.get("coefficients_max_abs_diff")
                ),
                "time_vanilla": to_json_safe(component_result.get("time_vanilla")),
                "time_fast": to_json_safe(component_result.get("time_fast")),
                "objective_vanilla": to_json_safe(
                    component_result.get("objective_vanilla")
                ),
                "objective_fast": to_json_safe(component_result.get("objective_fast")),
            }
            for component_result in results
        ],
        "npz_component_index": to_json_safe(npz_component_index),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def plot_component_solver_comparison(
    output_dir: str,
    filename_prefix: str,
    vanilla_history: dict[str, Any],
    fast_history: dict[str, Any],
) -> None:
    """Plot selected outer-iteration convergence diagnostics for one component."""
    vanilla_outer_indices = vanilla_history["projection_outer_indices"]
    fast_outer_indices = fast_history["projection_outer_indices"]
    if list(vanilla_outer_indices) != list(fast_outer_indices):
        raise ValueError(
            "Vanilla and fast solver tracked different outer iteration indices."
        )

    plotter = DykstraPlotter(output_dir=output_dir)
    plotter.plot_outer_iteration_solver_comparison(
        vanilla_results=vanilla_history["projection_results"],
        fast_forward_results=fast_history["projection_results"],
        outer_indices=vanilla_outer_indices,
        filename_prefix=filename_prefix,
        show=False,
    )


def build_distribution_filename(
    prefix: str,
    seed: int,
    num_particles: int,
    optimization_config: OptimizationConfig,
) -> str:
    """Build a distribution plot filename from explicit metadata."""
    return (
        f"{prefix}"
        f"SEED={seed}_M={num_particles}_SGD={optimization_config.batch_size}_"
        f"PGDITERS={optimization_config.max_outer_iter:,}_"
        f"DYKSTRA_ITERS={optimization_config.base_inner_iter}_"
        f"{optimization_config.max_inner_iters}_L1={optimization_config.l1_reg}_"
        f"LR={optimization_config.learning_rate:.0e}_{optimization_config.lr_decay:.0e}_"
        f"IHT={optimization_config.prune_interval}.png"
    )


def plot_distribution_for_mode(
    solver_mode: SolverMode,
    output_dir: str,
    normal_samples: np.ndarray,
    z_samples: np.ndarray,
    results: list[dict[str, Any]],
    kr_map: KRMap,
    num_dimensions: int,
    seed: int,
    num_particles: int,
    optimization_config: OptimizationConfig,
    x_lim: tuple[float, float] | None,
    y_lim: tuple[float, float] | None,
) -> None:
    """Plot mapped distributions for the selected solver mode."""
    distribution_plotter = DistributionPlotter(output_dir=output_dir)

    if solver_mode == "both":
        vanilla_weights = kr_map.assemble_component_weights(results, "w_vanilla")
        fast_weights = kr_map.assemble_component_weights(results, "w_fast")

        vanilla_mapped = kr_map.evaluate(
            z=z_samples[:, :num_dimensions],
            weights_by_component=vanilla_weights,
        )
        fast_mapped = kr_map.evaluate(
            z=z_samples[:, :num_dimensions],
            weights_by_component=fast_weights,
        )

        distribution_plotter.plot_kr_map_distribution_comparison(
            normal_samples=normal_samples[:, :2],
            synthetic_samples=z_samples[:, :2],
            vanilla_mapped_samples=vanilla_mapped[:, :2],
            fast_mapped_samples=fast_mapped[:, :2],
            xlim=x_lim,
            ylim=y_lim,
            filename=build_distribution_filename(
                prefix=f"kr{num_dimensions}d_distribution_comparison_",
                seed=seed,
                num_particles=num_particles,
                optimization_config=optimization_config,
            ),
            show=False,
        )
        return

    if solver_mode == "vanilla":
        vanilla_weights = kr_map.assemble_component_weights(results, "w_vanilla")
        vanilla_mapped = kr_map.evaluate(
            z=z_samples[:, :num_dimensions],
            weights_by_component=vanilla_weights,
        )
        distribution_plotter.plot_kr_map_distribution_single_solver(
            normal_samples=normal_samples[:, :2],
            synthetic_samples=z_samples[:, :2],
            mapped_samples=vanilla_mapped[:, :2],
            solver_label="vanilla Dykstra",
            xlim=x_lim,
            ylim=y_lim,
            filename=build_distribution_filename(
                prefix=f"kr{num_dimensions}d_distribution_vanilla_",
                seed=seed,
                num_particles=num_particles,
                optimization_config=optimization_config,
            ),
            show=False,
        )
        return

    fast_weights = kr_map.assemble_component_weights(results, "w_fast")
    fast_mapped = kr_map.evaluate(
        z=z_samples[:, :num_dimensions],
        weights_by_component=fast_weights,
    )
    distribution_plotter.plot_kr_map_distribution_single_solver(
        normal_samples=normal_samples[:, :2],
        synthetic_samples=z_samples[:, :2],
        mapped_samples=fast_mapped[:, :2],
        solver_label="fast-forward Dykstra",
        xlim=x_lim,
        ylim=y_lim,
        filename=build_distribution_filename(
            prefix=f"kr{num_dimensions}d_distribution_fast_",
            seed=seed,
            num_particles=num_particles,
            optimization_config=optimization_config,
        ),
        show=False,
    )

