"""Generate runtime scaling graphs from solver_runtime_benchmark JSON files.

This test file loads existing solver_runtime_benchmark JSON files and generates
the runtime scaling graphs, without running the full experiment.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from statistics import mean, median, stdev
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.plotter import BenchmarkPlotter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_experiment_benchmarks",
    "solver_runtime_benchmarks",
)
SCALING_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "data_generation",
    "latex_visual",
)


def _load_json(path: str) -> dict[str, Any]:
    """Load JSON file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _collect_rows(json_files: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect rows from benchmark JSON files."""
    component_rows: list[dict[str, Any]] = []
    fullmap_rows: list[dict[str, Any]] = []

    for json_path in json_files:
        # Extract seed and num_dimensions from filename
        # Example: kr10d_solver_runtime_benchmark_SEED=111_M=500_...json
        filename = os.path.basename(json_path)
        parts = filename.split("_")

        # Extract num_dimensions from first part (e.g., "kr10d" -> 10)
        kr_part = parts[0]  # e.g., "kr10d"
        num_dimensions = int(kr_part[2:-1])  # Extract number from "kr10d"

        # Extract seed
        seed_part = [p for p in parts if p.startswith("SEED=")][0]
        seed = int(seed_part.split("=")[1])

        payload = _load_json(json_path)
        components = payload.get("components", [])
        if not isinstance(components, list):
            print(f"Warning: Invalid benchmark JSON components at {json_path}")
            continue

        totals_by_solver: defaultdict[str, float] = defaultdict(float)
        counts_by_solver: defaultdict[str, int] = defaultdict(int)

        for component_entry in components:
            if not isinstance(component_entry, dict):
                continue
            component_dim_raw = component_entry.get("component_dim")
            if not isinstance(component_dim_raw, (int, np.integer)):
                continue
            component_dim = int(component_dim_raw)
            solvers = component_entry.get("solvers", {})
            if not isinstance(solvers, dict):
                continue

            for solver_label, solver_entry in solvers.items():
                if not isinstance(solver_entry, dict):
                    continue
                runtime = solver_entry.get("runtime_seconds")
                if runtime is None:
                    continue
                runtime_seconds = float(runtime)
                objective_final_raw = solver_entry.get("objective_final")
                objective_final = (
                    float(objective_final_raw)
                    if objective_final_raw is not None
                    else None
                )
                weights_l2_norm_raw = solver_entry.get("weights_l2_norm")
                weights_l2_norm = (
                    float(weights_l2_norm_raw)
                    if weights_l2_norm_raw is not None
                    else None
                )

                component_rows.append(
                    {
                        "seed": seed,
                        "num_dimensions": num_dimensions,
                        "component_dim": component_dim,
                        "solver": str(solver_label),
                        "runtime_seconds": runtime_seconds,
                        "objective_final": objective_final,
                        "weights_l2_norm": weights_l2_norm,
                        "benchmark_json_path": json_path,
                    }
                )
                totals_by_solver[str(solver_label)] += runtime_seconds
                counts_by_solver[str(solver_label)] += 1

        for solver_label, total_runtime in totals_by_solver.items():
            n_components = int(counts_by_solver[solver_label])
            fullmap_rows.append(
                {
                    "seed": seed,
                    "num_dimensions": num_dimensions,
                    "solver": solver_label,
                    "num_components": n_components,
                    "total_runtime_seconds": float(total_runtime),
                    "mean_component_runtime_seconds": float(total_runtime / max(n_components, 1)),
                    "benchmark_json_path": json_path,
                }
            )

    return component_rows, fullmap_rows


def _aggregate_rows(
    rows: list[dict[str, Any]],
    value_key: str,
) -> list[dict[str, Any]]:
    """Aggregate rows by dimension and solver."""
    grouped: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        dim = int(row["num_dimensions"])
        solver = str(row["solver"])
        grouped[(dim, solver)].append(float(row[value_key]))

    aggregated: list[dict[str, Any]] = []
    for (dim, solver), values in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(values)
        aggregated.append(
            {
                "num_dimensions": dim,
                "solver": solver,
                "n_runs": n,
                "mean": float(mean(values)),
                "std": float(stdev(values)) if n > 1 else 0.0,
                "median": float(median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )
    return aggregated


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_graphs_from_run(run_timestamp: str) -> dict[str, str]:
    """Generate graphs for a specific run timestamp.

    Args:
        run_timestamp: Timestamp string (e.g., "20260419" or "20260420")

    Returns:
        Dictionary with paths to generated files
    """
    print(f"\n=== Generating graphs for run: {run_timestamp} ===")

    # Find all JSON files with matching timestamp
    json_files = []
    for filename in os.listdir(BENCHMARK_DIR):
        if filename.endswith(".json") and run_timestamp in filename:
            json_files.append(os.path.join(BENCHMARK_DIR, filename))

    if not json_files:
        raise ValueError(f"No JSON files found with timestamp {run_timestamp}")

    print(f"Found {len(json_files)} JSON files for this run")
    json_files.sort()

    # Collect and aggregate data
    component_rows, fullmap_rows = _collect_rows(json_files)
    component_agg = _aggregate_rows(component_rows, value_key="runtime_seconds")
    fullmap_agg = _aggregate_rows(fullmap_rows, value_key="total_runtime_seconds")

    print(f"Collected {len(component_rows)} component rows")
    print(f"Collected {len(fullmap_rows)} fullmap rows")

    # Generate output paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_component_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_component_raw_TS={run_timestamp}_RegenTS={timestamp}.csv",
    )
    raw_fullmap_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_fullmap_raw_TS={run_timestamp}_RegenTS={timestamp}.csv",
    )
    agg_component_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_component_agg_TS={run_timestamp}_RegenTS={timestamp}.csv",
    )
    agg_fullmap_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_fullmap_agg_TS={run_timestamp}_RegenTS={timestamp}.csv",
    )

    fullmap_plot_path = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_vs_dimension_fullmap_TS={run_timestamp}_RegenTS={timestamp}.png",
    )
    component_plot_path = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_vs_dimension_component_TS={run_timestamp}_RegenTS={timestamp}.png",
    )

    # Write CSVs
    _write_csv(raw_component_csv, component_rows)
    _write_csv(raw_fullmap_csv, fullmap_rows)
    _write_csv(agg_component_csv, component_agg)
    _write_csv(agg_fullmap_csv, fullmap_agg)

    # Generate plots
    benchmark_plotter = BenchmarkPlotter(output_dir=SCALING_OUTPUT_DIR, dpi=180)
    benchmark_plotter.plot_runtime_scaling(
        aggregated_rows=fullmap_agg,
        y_key_mean="mean",
        y_key_std="std",
        title="Solver Runtime vs Dimension (Full Map Runtime)",
        y_label="Runtime (seconds)",
        filename=os.path.basename(fullmap_plot_path),
        show=False,
    )
    benchmark_plotter.plot_runtime_scaling(
        aggregated_rows=component_agg,
        y_key_mean="mean",
        y_key_std="std",
        title="Solver Runtime vs Dimension (Per-Component Runtime)",
        y_label="Runtime (seconds)",
        filename=os.path.basename(component_plot_path),
        show=False,
    )

    print(f"\nGenerated files for run {run_timestamp}:")
    print(f"  Raw component CSV: {raw_component_csv}")
    print(f"  Raw full-map CSV: {raw_fullmap_csv}")
    print(f"  Aggregated component CSV: {agg_component_csv}")
    print(f"  Aggregated full-map CSV: {agg_fullmap_csv}")
    print(f"  Full-map plot: {fullmap_plot_path}")
    print(f"  Per-component plot: {component_plot_path}")

    return {
        "raw_component_csv": raw_component_csv,
        "raw_fullmap_csv": raw_fullmap_csv,
        "agg_component_csv": agg_component_csv,
        "agg_fullmap_csv": agg_fullmap_csv,
        "fullmap_plot_png": fullmap_plot_path,
        "component_plot_png": component_plot_path,
    }


def test_generate_graphs_run1() -> None:
    """Generate graphs for run 1 (2026-04-19)."""
    run_timestamp = "20260419"
    result = generate_graphs_from_run(run_timestamp)
    assert os.path.exists(result["fullmap_plot_png"]), "Full-map plot not created"
    assert os.path.exists(result["component_plot_png"]), "Component plot not created"
    print("\n✓ Run 1 graphs generated successfully")


def test_generate_graphs_run2() -> None:
    """Generate graphs for run 2 (2026-04-20)."""
    run_timestamp = "20260420"
    result = generate_graphs_from_run(run_timestamp)
    assert os.path.exists(result["fullmap_plot_png"]), "Full-map plot not created"
    assert os.path.exists(result["component_plot_png"]), "Component plot not created"
    print("\n✓ Run 2 graphs generated successfully")


if __name__ == "__main__":
    print("Generating runtime scaling graphs from existing benchmark JSON files...")
    test_generate_graphs_run1()
    test_generate_graphs_run2()
    print("\n=== All graphs generated successfully ===")
