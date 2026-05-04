"""Generate mapped-progress grid figures from saved full-run iterate artifacts.

This script reads saved full-run JSON/NPZ artifacts and reconstructs mapped
sample iterates for one solver. It then renders progress grids using:
1) automatic early-emphasis frame selection, and
2) explicit frame-index selection.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.optimal_transport import HermiteBasis, KRMap
from utils.plotter import DistributionPlotter


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FULL_RUN_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_experiment_benchmarks",
    "full_run_iterates",
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_experiment_benchmarks",
    "distribution_plots",
)

SOLVER_LABEL = "fast"  # "fast" or "vanilla"
NUM_DIMENSIONS = 2
KR_DEGREE = 2
PLOT_SIZE = 7.5
PLOT_XLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)
PLOT_YLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)

# Optional explicit frame indices (in mapped sequence space). Set to None to disable.
EXPLICIT_FRAME_INDICES: list[int] | None = [0, 3, 8, 10, 15, 20, 30, 80, 345, 400, 600]
# Optional substring filter for selecting a specific run JSON by filename.s
# Examples: "SEED=42", "TS=20260422-101530", "MODE=fast".
# Keep as None to default to latest run.
RUN_FILENAME_FILTER: str | None = "SEED=1234"


def _latest_full_run_json(path: str, filename_filter: str | None = None) -> str:
    candidates = sorted(glob.glob(os.path.join(path, "*.json")))
    if filename_filter is not None and filename_filter.strip() != "":
        token = filename_filter.strip()
        candidates = [p for p in candidates if token in os.path.basename(p)]
    if not candidates:
        if filename_filter is not None and filename_filter.strip() != "":
            raise FileNotFoundError(
                "No full-run JSON artifacts matched filter "
                f"'{filename_filter}' in: {path}"
            )
        raise FileNotFoundError(f"No full-run JSON artifacts found in: {path}")
    return candidates[-1]


def _resolve_npz_path(payload: dict[str, Any], json_path: str) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("JSON artifact metadata is missing or invalid.")
    pointer = metadata.get("npz_pointer", {})
    if not isinstance(pointer, dict):
        raise ValueError("JSON artifact npz_pointer is missing or invalid.")

    absolute_path = pointer.get("absolute_path")
    if isinstance(absolute_path, str) and absolute_path.strip():
        return absolute_path

    rel_path = pointer.get("relative_to_json_dir")
    if isinstance(rel_path, str) and rel_path.strip():
        return os.path.abspath(os.path.join(os.path.dirname(json_path), rel_path))

    raise ValueError("Could not resolve NPZ path from artifact JSON.")


def _build_mapped_sequence_from_artifacts(
    payload: dict[str, Any],
    npz_path: str,
    solver_label: str,
    num_dimensions: int,
) -> tuple[np.ndarray, list[int]]:
    npz_component_index = payload.get("npz_component_index")
    if not isinstance(npz_component_index, list):
        raise ValueError("npz_component_index missing from full-run JSON.")

    with np.load(npz_path, allow_pickle=True) as npz_data:
        if "run_target_samples_eval" not in npz_data:
            raise KeyError("NPZ artifact missing run_target_samples_eval.")
        target_eval = np.asarray(npz_data["run_target_samples_eval"], dtype=float)

        weight_iterates_by_component: dict[int, np.ndarray] = {}
        for component_meta in npz_component_index:
            if not isinstance(component_meta, dict):
                continue
            component_dim = component_meta.get("component_dim")
            if not isinstance(component_dim, (int, np.integer)):
                continue
            solver_meta = (
                component_meta.get("solvers", {}).get(solver_label)
                if isinstance(component_meta.get("solvers"), dict)
                else None
            )
            if not isinstance(solver_meta, dict):
                continue
            key = solver_meta.get("weight_iterates_key")
            if not isinstance(key, str):
                continue
            if key not in npz_data:
                continue
            weight_iterates_by_component[int(component_dim)] = np.asarray(
                npz_data[key],
                dtype=float,
            )

    expected_dims = list(range(1, num_dimensions + 1))
    missing = [d for d in expected_dims if d not in weight_iterates_by_component]
    if missing:
        raise KeyError(
            f"Missing weight-iterate histories for solver '{solver_label}': {missing}"
        )

    num_frames = min(weight_iterates_by_component[d].shape[0] for d in expected_dims)
    if num_frames < 1:
        raise ValueError("No frames available in saved iterate histories.")

    kr_map = KRMap(degree=KR_DEGREE, basis_1d=HermiteBasis(), log_epsilon=1e-8)
    mapped_sequence: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        component_results = [
            {
                "component_dim": dim,
                f"w_{solver_label}": weight_iterates_by_component[dim][frame_idx],
            }
            for dim in expected_dims
        ]
        weights = kr_map.assemble_component_weights(
            component_results,
            f"w_{solver_label}",
        )
        mapped = kr_map.evaluate(
            z=target_eval[:, :num_dimensions],
            weights_by_component=weights,
        )
        mapped_sequence.append(np.asarray(mapped[:, :2], dtype=float))

    outer_indices = [idx - 1 for idx in range(num_frames)]
    return np.asarray(mapped_sequence, dtype=float), outer_indices


def _build_progress_filename(
    metadata: dict[str, Any],
    num_dimensions: int,
    degree: int,
    selection_tag: str,
) -> str:
    seed = metadata.get("seed", "NA")
    num_particles = metadata.get("num_particles", "NA")
    batch_size = metadata.get("batch_size", "None")
    max_outer_iter = metadata.get("max_outer_iter", "NA")
    base_inner_iter = metadata.get("base_inner_iter", "NA")
    max_inner_iters = metadata.get("max_inner_iters", "NA")
    l1_reg = metadata.get("l1_reg", "NA")
    learning_rate = metadata.get("learning_rate", "NA")
    lr_decay = metadata.get("lr_decay", "NA")
    prune_interval = metadata.get("prune_interval", "NA")
    return (
        f"kr{num_dimensions}d_progress_"
        f"SEED={seed}_M={num_particles}_D={degree}_SGD={batch_size}_"
        f"PGDITERS={int(max_outer_iter):,}_"
        f"DYKSTRA_ITERS={base_inner_iter}_{max_inner_iters}_"
        f"L1={l1_reg}_LR={float(learning_rate):.0e}_{float(lr_decay):.0e}_"
        f"IHT={prune_interval}_{selection_tag}.png"
    )


if __name__ == "__main__":
    artifact_json = _latest_full_run_json(
        FULL_RUN_DIR,
        filename_filter=RUN_FILENAME_FILTER,
    )
    with open(artifact_json, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    artifact_npz = _resolve_npz_path(payload=payload, json_path=artifact_json)
    mapped_sequence, outer_indices = _build_mapped_sequence_from_artifacts(
        payload=payload,
        npz_path=artifact_npz,
        solver_label=SOLVER_LABEL,
        num_dimensions=NUM_DIMENSIONS,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plotter = DistributionPlotter(output_dir=OUTPUT_DIR)

    plotter.plot_mapped_progress_grid(
        mapped_samples_sequence=mapped_sequence,
        solver_label=f"{SOLVER_LABEL} mapped",
        outer_indices=outer_indices,
        iteration_indices=None,
        num_panels=12,
        ncols=3,
        emphasize_early=1.15,
        xlim=PLOT_XLIM,
        ylim=PLOT_YLIM,
        panel_title_template="PGD iteration: {iter}",
        filename=_build_progress_filename(
            metadata=metadata,
            num_dimensions=NUM_DIMENSIONS,
            degree=KR_DEGREE,
            selection_tag="SEL=auto",
        ),
        show=False,
    )

    plotter.plot_mapped_progress_grid(
        mapped_samples_sequence=mapped_sequence,
        solver_label=f"{SOLVER_LABEL} mapped",
        outer_indices=outer_indices,
        iteration_indices=EXPLICIT_FRAME_INDICES,
        num_panels=12,  # ignored when iteration_indices is provided
        ncols=3,
        emphasize_early=2.0,
        xlim=PLOT_XLIM,
        ylim=PLOT_YLIM,
        panel_title_template="PGD iteration: {iter}",
        filename=_build_progress_filename(
            metadata=metadata,
            num_dimensions=NUM_DIMENSIONS,
            degree=KR_DEGREE,
            selection_tag="SEL=custom",
        ),
        show=False,
    )

    print(f"Used artifact JSON: {artifact_json}")
    print(f"Used artifact NPZ:  {artifact_npz}")
    print(f"Saved progress grids in: {OUTPUT_DIR}")
