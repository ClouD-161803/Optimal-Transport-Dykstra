"""Artifact-writing and plotting helpers for experiment runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable

import numpy as np

from core.config import OptimizationConfig, SolverMode
from utils.optimal_transport import KRMap
from utils.plotter import DistributionPlotter, DykstraPlotter

NPZ_REFERENCE_PLOT_SAMPLES_KEY = "run_reference_samples_plot"
NPZ_TARGET_PLOT_SAMPLES_KEY = "run_target_samples_plot"
NPZ_TARGET_EVAL_SAMPLES_KEY = "run_target_samples_eval"
NPZ_INITIAL_MAPPED_SAMPLES_VIDEO_KEY = "run_initial_mapped_samples_video"


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


def save_solver_runtime_benchmark_json(
    results: list[dict[str, Any]],
    output_dir: str,
    solver_mode: SolverMode,
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
) -> str:
    """Write benchmark-focused JSON (times, final objectives, final weights)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = (
        f"kr{num_dimensions}d_solver_runtime_benchmark_"
        f"SEED={seed}_M={num_particles}_PGDITERS={max_outer_iter}_"
        f"DYKSTRA_ITERS={base_inner_iter}_{max_inner_iters}_MODE={solver_mode}_TS={timestamp}.json"
    )
    output_path = os.path.join(output_dir, filename)

    component_payloads: list[dict[str, Any]] = []
    for component_result in results:
        component_dim = int(component_result["component_dim"])
        solver_labels = sorted(
            key.removeprefix("time_")
            for key in component_result.keys()
            if key.startswith("time_")
        )
        solver_entries: dict[str, Any] = {}
        for solver_label in solver_labels:
            time_key = f"time_{solver_label}"
            obj_key = f"objective_{solver_label}"
            w_key = f"w_{solver_label}"
            weights = component_result.get(w_key)
            solver_entries[solver_label] = {
                "runtime_seconds": to_json_safe(component_result.get(time_key)),
                "objective_final": to_json_safe(component_result.get(obj_key)),
                "weights": to_json_safe(weights),
                "weights_l2_norm": (
                    float(np.linalg.norm(np.asarray(weights, dtype=float)))
                    if weights is not None
                    else None
                ),
            }

        ranked_by_runtime = sorted(
            (
                (label, solver_entries[label].get("runtime_seconds"))
                for label in solver_entries
                if solver_entries[label].get("runtime_seconds") is not None
            ),
            key=lambda pair: float(pair[1]),
        )
        ranking_payload = [
            {"rank": idx + 1, "solver": label, "runtime_seconds": float(runtime)}
            for idx, (label, runtime) in enumerate(ranked_by_runtime)
        ]

        distance_from_fast: dict[str, float | None] = {}
        w_fast = component_result.get("w_fast")
        if w_fast is not None:
            w_fast_arr = np.asarray(w_fast, dtype=float)
            for solver_label in solver_entries:
                weights = component_result.get(f"w_{solver_label}")
                if weights is None:
                    distance_from_fast[solver_label] = None
                    continue
                w_arr = np.asarray(weights, dtype=float)
                if w_arr.shape != w_fast_arr.shape:
                    distance_from_fast[solver_label] = None
                    continue
                distance_from_fast[solver_label] = float(np.linalg.norm(w_arr - w_fast_arr))

        component_payloads.append(
            {
                "component_dim": component_dim,
                "solvers": solver_entries,
                "runtime_ranking": ranking_payload,
                "weights_l2_distance_from_fast": to_json_safe(distance_from_fast),
            }
        )

    payload: dict[str, Any] = {
        "metadata": {
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "artifact_type": "solver_runtime_benchmark",
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
        },
        "components": component_payloads,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


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
    reference_samples_for_plot: np.ndarray | None = None,
    target_samples_for_plot: np.ndarray | None = None,
    target_samples_for_eval: np.ndarray | None = None,
    initial_mapped_samples_for_video: np.ndarray | None = None,
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

    if reference_samples_for_plot is not None:
        arrays_to_save[NPZ_REFERENCE_PLOT_SAMPLES_KEY] = np.asarray(
            reference_samples_for_plot,
            dtype=float,
        )
    if target_samples_for_plot is not None:
        arrays_to_save[NPZ_TARGET_PLOT_SAMPLES_KEY] = np.asarray(
            target_samples_for_plot,
            dtype=float,
        )
    if target_samples_for_eval is not None:
        arrays_to_save[NPZ_TARGET_EVAL_SAMPLES_KEY] = np.asarray(
            target_samples_for_eval,
            dtype=float,
        )
    if initial_mapped_samples_for_video is not None:
        arrays_to_save[NPZ_INITIAL_MAPPED_SAMPLES_VIDEO_KEY] = np.asarray(
            initial_mapped_samples_for_video,
            dtype=float,
        )

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
            arrays_to_save[f"{prefix}_weight_iterates"] = np.asarray(
                history.get("weight_iterates", []), dtype=float
            )
            arrays_to_save[f"{prefix}_projection_iterations_run"] = np.asarray(
                history.get("projection_iterations_run", []), dtype=int
            )
            arrays_to_save[f"{prefix}_projection_terminated_early"] = np.asarray(
                history.get("projection_terminated_early", []), dtype=bool
            )
            arrays_to_save[f"{prefix}_projection_termination_reason"] = np.asarray(
                history.get("projection_termination_reason", []), dtype=object
            )

            solver_meta: dict[str, Any] = {
                "weights_key": f"{prefix}_weights",
                "objective_value_key": f"{prefix}_objective_value",
                "dykstra_inner_iters_key": f"{prefix}_dykstra_inner_iters",
                "weight_iterates_key": f"{prefix}_weight_iterates",
                "projection_iterations_run_key": f"{prefix}_projection_iterations_run",
                "projection_terminated_early_key": (
                    f"{prefix}_projection_terminated_early"
                ),
                "projection_termination_reason_key": (
                    f"{prefix}_projection_termination_reason"
                ),
                "projection_terminated_early_any": bool(
                    history.get("projection_terminated_early_any", False)
                ),
                "projection_terminated_early_count": int(
                    history.get("projection_terminated_early_count", 0)
                ),
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
                iterations_run = np.asarray(
                    [getattr(result, "iterations_run", None) for result in proj_values],
                    dtype=object,
                )
                terminated_early = np.asarray(
                    [getattr(result, "terminated_early", None) for result in proj_values],
                    dtype=object,
                )
                termination_reason = np.asarray(
                    [getattr(result, "termination_reason", None) for result in proj_values],
                    dtype=object,
                )

                arrays_to_save[f"{prefix}_{proj_key}_projection"] = projections
                arrays_to_save[f"{prefix}_{proj_key}_squared_errors"] = squared_errors
                arrays_to_save[f"{prefix}_{proj_key}_stalled_errors"] = stalled_errors
                arrays_to_save[f"{prefix}_{proj_key}_converged_errors"] = converged_errors
                arrays_to_save[f"{prefix}_{proj_key}_active_half_spaces"] = active_half_spaces
                arrays_to_save[f"{prefix}_{proj_key}_iterations_run"] = iterations_run
                arrays_to_save[f"{prefix}_{proj_key}_terminated_early"] = terminated_early
                arrays_to_save[f"{prefix}_{proj_key}_termination_reason"] = termination_reason

                solver_meta[f"{proj_key}_keys"] = {
                    "projection": f"{prefix}_{proj_key}_projection",
                    "squared_errors": f"{prefix}_{proj_key}_squared_errors",
                    "stalled_errors": f"{prefix}_{proj_key}_stalled_errors",
                    "converged_errors": f"{prefix}_{proj_key}_converged_errors",
                    "active_half_spaces": f"{prefix}_{proj_key}_active_half_spaces",
                    "iterations_run": f"{prefix}_{proj_key}_iterations_run",
                    "terminated_early": f"{prefix}_{proj_key}_terminated_early",
                    "termination_reason": f"{prefix}_{proj_key}_termination_reason",
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
    npz_reference_plot_samples_key: str | None = NPZ_REFERENCE_PLOT_SAMPLES_KEY,
    npz_target_plot_samples_key: str | None = NPZ_TARGET_PLOT_SAMPLES_KEY,
    npz_target_eval_samples_key: str | None = NPZ_TARGET_EVAL_SAMPLES_KEY,
    npz_initial_mapped_samples_video_key: str | None = NPZ_INITIAL_MAPPED_SAMPLES_VIDEO_KEY,
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
            "npz_data_keys": {
                "reference_samples_plot": npz_reference_plot_samples_key,
                "target_samples_plot": npz_target_plot_samples_key,
                "target_samples_eval": npz_target_eval_samples_key,
                "initial_mapped_samples_video": npz_initial_mapped_samples_video_key,
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
                "projection_terminated_early_any_vanilla": to_json_safe(
                    component_result.get("history_vanilla", {}).get(
                        "projection_terminated_early_any"
                    )
                    if isinstance(component_result.get("history_vanilla"), dict)
                    else None
                ),
                "projection_terminated_early_count_vanilla": to_json_safe(
                    component_result.get("history_vanilla", {}).get(
                        "projection_terminated_early_count"
                    )
                    if isinstance(component_result.get("history_vanilla"), dict)
                    else None
                ),
                "projection_terminated_early_any_fast": to_json_safe(
                    component_result.get("history_fast", {}).get(
                        "projection_terminated_early_any"
                    )
                    if isinstance(component_result.get("history_fast"), dict)
                    else None
                ),
                "projection_terminated_early_count_fast": to_json_safe(
                    component_result.get("history_fast", {}).get(
                        "projection_terminated_early_count"
                    )
                    if isinstance(component_result.get("history_fast"), dict)
                    else None
                ),
            }
            for component_result in results
        ],
        "npz_component_index": to_json_safe(npz_component_index),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def _resolve_npz_path_from_json_payload(
    payload: dict[str, Any],
    json_path: str,
) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("JSON payload is missing valid 'metadata'.")

    npz_pointer = metadata.get("npz_pointer", {})
    if not isinstance(npz_pointer, dict):
        raise ValueError("JSON payload is missing valid 'npz_pointer' metadata.")

    absolute_path = npz_pointer.get("absolute_path")
    if isinstance(absolute_path, str) and absolute_path.strip() != "":
        return absolute_path

    relative_path = npz_pointer.get("relative_to_json_dir")
    if isinstance(relative_path, str) and relative_path.strip() != "":
        return os.path.abspath(os.path.join(os.path.dirname(json_path), relative_path))

    raise ValueError("Could not resolve NPZ path from full-run JSON metadata.")


def _resolve_single_solver_titles(
    solver_label: str,
    panel_titles_both: tuple[str, str, str, str] | None,
    panel_titles_vanilla: tuple[str, str, str] | None,
    panel_titles_fast: tuple[str, str, str] | None,
) -> tuple[str, str, str] | None:
    if solver_label == "vanilla":
        if panel_titles_vanilla is not None:
            return panel_titles_vanilla
        if panel_titles_both is not None:
            return panel_titles_both[0], panel_titles_both[1], panel_titles_both[2]
        return None

    if panel_titles_fast is not None:
        return panel_titles_fast
    if panel_titles_both is not None:
        return panel_titles_both[0], panel_titles_both[1], panel_titles_both[3]
    return None


def _build_distribution_shift_filename_prefix(
    metadata: dict[str, Any],
    solver_label: str,
) -> str:
    num_dimensions = metadata.get("num_dimensions", "NA")
    seed = metadata.get("seed", "NA")
    num_particles = metadata.get("num_particles", "NA")
    max_outer_iter = metadata.get("max_outer_iter", "NA")
    return (
        f"kr{num_dimensions}d_distribution_shift_{solver_label}_"
        f"SEED={seed}_M={num_particles}_PGDITERS={max_outer_iter}"
    )


def save_distribution_shift_media_from_artifacts(
    full_run_json_path: str,
    output_dir: str,
    kr_map: KRMap,
    solver_mode: SolverMode,
    num_dimensions: int,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    panel_titles_both: tuple[str, str, str, str] | None = None,
    panel_titles_vanilla: tuple[str, str, str] | None = None,
    panel_titles_fast: tuple[str, str, str] | None = None,
    mapped_output_inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    fps: int = 12,
    save_mp4: bool = True,
    save_gif: bool = True,
) -> dict[str, dict[str, str]]:
    """Build and save per-outer-iteration distribution-shift animations."""
    with open(full_run_json_path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Full-run JSON must contain a 'metadata' object.")

    npz_path = _resolve_npz_path_from_json_payload(payload=payload, json_path=full_run_json_path)
    npz_component_index = payload.get("npz_component_index")
    if not isinstance(npz_component_index, list):
        raise ValueError("Full-run JSON must contain 'npz_component_index' list.")

    npz_data_keys = metadata.get("npz_data_keys", {})
    if not isinstance(npz_data_keys, dict):
        npz_data_keys = {}
    reference_key = str(
        npz_data_keys.get("reference_samples_plot", NPZ_REFERENCE_PLOT_SAMPLES_KEY)
    )
    target_plot_key = str(
        npz_data_keys.get("target_samples_plot", NPZ_TARGET_PLOT_SAMPLES_KEY)
    )
    target_eval_key = str(
        npz_data_keys.get("target_samples_eval", NPZ_TARGET_EVAL_SAMPLES_KEY)
    )
    initial_mapped_video_key_raw = npz_data_keys.get("initial_mapped_samples_video")
    initial_mapped_video_key = (
        str(initial_mapped_video_key_raw)
        if isinstance(initial_mapped_video_key_raw, str)
        else None
    )

    os.makedirs(output_dir, exist_ok=True)
    plotter = DistributionPlotter(output_dir=output_dir)

    media_paths: dict[str, dict[str, str]] = {}
    solver_labels = (
        ("vanilla", "fast")
        if solver_mode == "both"
        else (solver_mode,)
    )

    with np.load(npz_path, allow_pickle=True) as npz_data:
        if (
            reference_key not in npz_data
            or target_plot_key not in npz_data
            or target_eval_key not in npz_data
        ):
            raise KeyError(
                "Missing plotting/evaluation sample arrays in NPZ artifact. "
                "Expected keys: "
                f"{reference_key}, {target_plot_key}, {target_eval_key}."
            )

        reference_for_plot = np.asarray(npz_data[reference_key], dtype=float)
        target_for_plot = np.asarray(npz_data[target_plot_key], dtype=float)
        target_for_eval = np.asarray(npz_data[target_eval_key], dtype=float)
        initial_mapped_video_samples: np.ndarray | None = None
        if (
            initial_mapped_video_key is not None
            and initial_mapped_video_key in npz_data
        ):
            initial_mapped_video_samples = np.asarray(
                npz_data[initial_mapped_video_key],
                dtype=float,
            )

        for solver_label in solver_labels:
            weight_iterates_by_component: dict[int, np.ndarray] = {}
            for component_meta in npz_component_index:
                if not isinstance(component_meta, dict):
                    continue
                component_dim_raw = component_meta.get("component_dim")
                if not isinstance(component_dim_raw, (int, np.integer)):
                    continue
                component_dim = int(component_dim_raw)
                solvers_meta = component_meta.get("solvers", {})
                if not isinstance(solvers_meta, dict):
                    continue
                solver_meta = solvers_meta.get(solver_label)
                if not isinstance(solver_meta, dict):
                    continue
                weight_iterates_key = solver_meta.get("weight_iterates_key")
                if not isinstance(weight_iterates_key, str):
                    raise KeyError(
                        f"Missing weight_iterates_key for component {component_dim} "
                        f"solver '{solver_label}'."
                    )
                if weight_iterates_key not in npz_data:
                    raise KeyError(
                        f"NPZ key '{weight_iterates_key}' was not found for "
                        f"component {component_dim} solver '{solver_label}'."
                    )

                weight_iterates = np.asarray(npz_data[weight_iterates_key], dtype=float)
                if weight_iterates.ndim != 2:
                    raise ValueError(
                        f"Weight iterates for component {component_dim} solver "
                        f"'{solver_label}' must have shape (num_frames, n_coeffs)."
                    )
                weight_iterates_by_component[component_dim] = weight_iterates

            expected_dims = list(range(1, num_dimensions + 1))
            missing_dims = [dim for dim in expected_dims if dim not in weight_iterates_by_component]
            if missing_dims:
                raise KeyError(
                    f"Missing weight-iterate histories for solver '{solver_label}' "
                    f"component dimensions: {missing_dims}."
                )

            num_frames = min(
                int(weight_iterates_by_component[dim].shape[0])
                for dim in expected_dims
            )
            if num_frames < 1:
                raise ValueError(
                    f"Solver '{solver_label}' has no stored weight iterates to animate."
                )

            mapped_sequence: list[np.ndarray] = []
            outer_indices = [frame_idx - 1 for frame_idx in range(num_frames)]
            for frame_idx in range(num_frames):
                per_component_results = [
                    {
                        "component_dim": dim,
                        f"w_{solver_label}": weight_iterates_by_component[dim][frame_idx],
                    }
                    for dim in expected_dims
                ]
                assembled_weights = kr_map.assemble_component_weights(
                    per_component_results,
                    f"w_{solver_label}",
                )
                mapped = kr_map.evaluate(
                    z=target_for_eval[:, :num_dimensions],
                    weights_by_component=assembled_weights,
                )
                if mapped_output_inverse_transform is not None:
                    mapped = mapped_output_inverse_transform(mapped)
                mapped_sequence.append(np.asarray(mapped[:, :2], dtype=float))

            if (
                initial_mapped_video_samples is not None
                and len(mapped_sequence) > 0
                and initial_mapped_video_samples.ndim == 2
                and initial_mapped_video_samples.shape[1] >= 2
                and initial_mapped_video_samples.shape[0] == mapped_sequence[0].shape[0]
            ):
                initial_xy = np.asarray(
                    initial_mapped_video_samples[:, :2],
                    dtype=float,
                )
                if len(mapped_sequence) == 1:
                    mapped_sequence[0] = initial_xy
                else:
                    final_frame_index = float(len(mapped_sequence) - 1)
                    for frame_idx in range(len(mapped_sequence)):
                        alpha = float(frame_idx) / final_frame_index
                        mapped_sequence[frame_idx] = (
                            (1.0 - alpha) * initial_xy
                            + alpha * np.asarray(mapped_sequence[frame_idx], dtype=float)
                        )

            solver_title_label = (
                "vanilla Dykstra" if solver_label == "vanilla" else "fast-forward Dykstra"
            )
            resolved_panel_titles = _resolve_single_solver_titles(
                solver_label=solver_label,
                panel_titles_both=panel_titles_both,
                panel_titles_vanilla=panel_titles_vanilla,
                panel_titles_fast=panel_titles_fast,
            )
            if resolved_panel_titles is None:
                resolved_panel_titles = ("Reference", "Sheared", "Mapped")
            else:
                resolved_panel_titles = (
                    "Reference",
                    "Sheared",
                    "Mapped",
                )
            media_paths[solver_label] = plotter.save_kr_map_distribution_shift_animation(
                normal_samples=reference_for_plot[:, :2],
                synthetic_samples=target_for_plot[:, :2],
                mapped_samples_sequence=np.asarray(mapped_sequence, dtype=float),
                solver_label=solver_title_label,
                outer_indices=outer_indices,
                panel_titles=resolved_panel_titles,
                xlim=x_lim,
                ylim=y_lim,
                filename_prefix=_build_distribution_shift_filename_prefix(
                    metadata=metadata,
                    solver_label=solver_label,
                ),
                fps=fps,
                save_mp4=save_mp4,
                save_gif=save_gif,
                ramp_playback_speed=True,
                start_speed=1.0,
                end_speed=30.0,
                speed_ramp_mode="exp",
                target_duration_seconds=15.0,
            )

    return media_paths


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
    degree: int,
    optimization_config: OptimizationConfig,
) -> str:
    """Build a distribution plot filename from explicit metadata."""
    return (
        f"{prefix}"
        f"SEED={seed}_M={num_particles}_D={degree}_SGD={optimization_config.batch_size}_"
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
    degree: int,
    optimization_config: OptimizationConfig,
    x_lim: tuple[float, float] | None,
    y_lim: tuple[float, float] | None,
    panel_titles_both: tuple[str, str, str, str] | None = None,
    panel_titles_vanilla: tuple[str, str, str] | None = None,
    panel_titles_fast: tuple[str, str, str] | None = None,
    plotted_reference_samples: np.ndarray | None = None,
    plotted_target_samples: np.ndarray | None = None,
    eval_target_samples: np.ndarray | None = None,
    mapped_output_inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Plot mapped distributions for the selected solver mode."""
    distribution_plotter = DistributionPlotter(output_dir=output_dir)
    reference_for_plot = (
        np.asarray(plotted_reference_samples)
        if plotted_reference_samples is not None
        else normal_samples
    )
    target_for_plot = (
        np.asarray(plotted_target_samples)
        if plotted_target_samples is not None
        else z_samples
    )
    target_for_eval = (
        np.asarray(eval_target_samples)
        if eval_target_samples is not None
        else z_samples
    )

    if solver_mode == "both":
        vanilla_weights = kr_map.assemble_component_weights(results, "w_vanilla")
        fast_weights = kr_map.assemble_component_weights(results, "w_fast")

        vanilla_mapped = kr_map.evaluate(
            z=target_for_eval[:, :num_dimensions],
            weights_by_component=vanilla_weights,
        )
        fast_mapped = kr_map.evaluate(
            z=target_for_eval[:, :num_dimensions],
            weights_by_component=fast_weights,
        )
        if mapped_output_inverse_transform is not None:
            vanilla_mapped = mapped_output_inverse_transform(vanilla_mapped)
            fast_mapped = mapped_output_inverse_transform(fast_mapped)

        distribution_plotter.plot_kr_map_distribution_comparison(
            normal_samples=reference_for_plot[:, :2],
            synthetic_samples=target_for_plot[:, :2],
            vanilla_mapped_samples=vanilla_mapped[:, :2],
            fast_mapped_samples=fast_mapped[:, :2],
            panel_titles=panel_titles_both,
            xlim=x_lim,
            ylim=y_lim,
            filename=build_distribution_filename(
                prefix=f"kr{num_dimensions}d_distribution_comparison_",
                seed=seed,
                num_particles=num_particles,
                degree=degree,
                optimization_config=optimization_config,
            ),
            show=False,
        )
        return

    if solver_mode == "vanilla":
        vanilla_weights = kr_map.assemble_component_weights(results, "w_vanilla")
        vanilla_mapped = kr_map.evaluate(
            z=target_for_eval[:, :num_dimensions],
            weights_by_component=vanilla_weights,
        )
        if mapped_output_inverse_transform is not None:
            vanilla_mapped = mapped_output_inverse_transform(vanilla_mapped)
        distribution_plotter.plot_kr_map_distribution_single_solver(
            normal_samples=reference_for_plot[:, :2],
            synthetic_samples=target_for_plot[:, :2],
            mapped_samples=vanilla_mapped[:, :2],
            solver_label="vanilla Dykstra",
            panel_titles=panel_titles_vanilla,
            xlim=x_lim,
            ylim=y_lim,
            filename=build_distribution_filename(
                prefix=f"kr{num_dimensions}d_distribution_vanilla_",
                seed=seed,
                num_particles=num_particles,
                degree=degree,
                optimization_config=optimization_config,
            ),
            show=False,
        )
        return

    fast_weights = kr_map.assemble_component_weights(results, "w_fast")
    fast_mapped = kr_map.evaluate(
        z=target_for_eval[:, :num_dimensions],
        weights_by_component=fast_weights,
    )
    if mapped_output_inverse_transform is not None:
        fast_mapped = mapped_output_inverse_transform(fast_mapped)
    distribution_plotter.plot_kr_map_distribution_single_solver(
        normal_samples=reference_for_plot[:, :2],
        synthetic_samples=target_for_plot[:, :2],
        mapped_samples=fast_mapped[:, :2],
        solver_label="fast-forward Dykstra",
        panel_titles=panel_titles_fast,
        xlim=x_lim,
        ylim=y_lim,
        filename=build_distribution_filename(
            prefix=f"kr{num_dimensions}d_distribution_fast_",
            seed=seed,
            num_particles=num_particles,
            degree=degree,
            optimization_config=optimization_config,
        ),
        show=False,
    )
