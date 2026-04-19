"""Runtime-scaling benchmark across dimensions for all registered solvers.

This experiment sweeps problem dimension and seed in ``RUN_SOLVER_MODE='benchmark'``
to compare total runtime for fast Dykstra and all registered QP backends.
It saves both raw and aggregated tables plus summary plots.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median, stdev
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import ExperimentConfig, OptimizationConfig, PlotConfig, RunConfig
from core.runner import build_identity_initial_guesses, run_synthetic_experiment
from utils.data_generator import LayeredBoomerangShearFunction, DataGenerator
from utils.optimal_transport import Basis, HermiteBasis, KRMap
from utils.plotter import BenchmarkPlotter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Sweep controls
DIMENSIONS: list[int] = [2, 3, 4, 5, 6]
SEEDS: list[int] = [111, 222, 333]
NUM_PARTICLES: int = 1000

# Optimisation settings
MAX_OUTER_ITER: int = 300
DYKSTRA_KWARGS: dict[str, Any] = {"track_error": False}
GRADIENT_CLIP_VALUE: float = 10.0
L1_REG: float = 0.0

# Inexact projection: Inner iters = BASE_INNER_ITER * (outer_iter ** INEXACT_POWER)
BASE_INNER_ITER: int = 1
MAX_INNER_ITERS: int = 10
INEXACT_POWER: float = np.log(MAX_INNER_ITERS / BASE_INNER_ITER) / np.log(MAX_OUTER_ITER)

# SGD
BATCH_SIZE: int | None = 100
LEARNING_RATE: float = 0.1
LR_DECAY: float = 1e-2 if BATCH_SIZE is not None else 0.0

# IHT
PRUNE_THRESHOLD: float = 1e-2
PRUNE_INTERVAL: int = 100

# KR map / data generator
DEGREE: int = 2
BASIS: Basis = HermiteBasis()
KR_MAP: KRMap = KRMap(
    degree=DEGREE,
    basis_1d=BASIS,
    log_epsilon=1e-8,
)
DATA_GENERATOR = DataGenerator(
    shear_function=LayeredBoomerangShearFunction(
        linear_scale=0.30,
        quadratic_scale=0.55,
        strength_decay=0.80,
    ),
)

# Output folder
SCALING_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_experiment_benchmarks",
    "solver_runtime_scaling",
)


@dataclass(frozen=True)
class RunArtifact:
    seed: int
    num_dimensions: int
    benchmark_json_path: str


def _build_experiment_config(seed: int, num_dimensions: int) -> ExperimentConfig:
    rng_seed = seed + 1 if BATCH_SIZE is not None else None
    run_config = RunConfig(
        run_solver_mode="benchmark",
        save_full_run_iterates=False,
        save_distribution_shift_media=False,
        enforce_matching=False,
    )
    plot_config = PlotConfig(
        plot_dykstra_iterates=False,
        plot_dykstra_outer_iterations=None,
        plot_distributions=False,
        x_lim=None,
        y_lim=None,
    )
    optimization_config = OptimizationConfig(
        learning_rate=LEARNING_RATE,
        max_outer_iter=MAX_OUTER_ITER,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        l1_reg=L1_REG,
        lr_decay=LR_DECAY,
        inexact_power=INEXACT_POWER,
        base_inner_iter=BASE_INNER_ITER,
        max_inner_iters=MAX_INNER_ITERS,
        batch_size=BATCH_SIZE,
        rng_seed=rng_seed,
        prune_threshold=PRUNE_THRESHOLD,
        prune_interval=PRUNE_INTERVAL,
        dykstra_kwargs=dict(DYKSTRA_KWARGS),
    )
    return ExperimentConfig(
        seed=seed,
        num_dimensions=num_dimensions,
        num_particles=NUM_PARTICLES,
        run=run_config,
        plot=plot_config,
        optimization=optimization_config,
    )


def _run_single(seed: int, num_dimensions: int) -> RunArtifact:
    config = _build_experiment_config(seed=seed, num_dimensions=num_dimensions)
    initial_guesses = build_identity_initial_guesses(
        kr_map=KR_MAP,
        num_dimensions=num_dimensions,
    )
    summary = run_synthetic_experiment(
        project_root=PROJECT_ROOT,
        config=config,
        generator=DATA_GENERATOR,
        kr_map=KR_MAP,
        initial_guesses_by_component=initial_guesses,
    )
    if summary.benchmark_json_path is None:
        raise RuntimeError("Expected benchmark_json_path in benchmark mode, got None.")
    return RunArtifact(
        seed=seed,
        num_dimensions=num_dimensions,
        benchmark_json_path=summary.benchmark_json_path,
    )


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _collect_rows(
    artifacts: list[RunArtifact],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_rows: list[dict[str, Any]] = []
    fullmap_rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        payload = _load_json(artifact.benchmark_json_path)
        components = payload.get("components", [])
        if not isinstance(components, list):
            raise ValueError(
                f"Invalid benchmark JSON components at {artifact.benchmark_json_path}"
            )

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
                        "seed": artifact.seed,
                        "num_dimensions": artifact.num_dimensions,
                        "component_dim": component_dim,
                        "solver": str(solver_label),
                        "runtime_seconds": runtime_seconds,
                        "objective_final": objective_final,
                        "weights_l2_norm": weights_l2_norm,
                        "benchmark_json_path": artifact.benchmark_json_path,
                    }
                )
                totals_by_solver[str(solver_label)] += runtime_seconds
                counts_by_solver[str(solver_label)] += 1

        for solver_label, total_runtime in totals_by_solver.items():
            n_components = int(counts_by_solver[solver_label])
            fullmap_rows.append(
                {
                    "seed": artifact.seed,
                    "num_dimensions": artifact.num_dimensions,
                    "solver": solver_label,
                    "num_components": n_components,
                    "total_runtime_seconds": float(total_runtime),
                    "mean_component_runtime_seconds": float(total_runtime / max(n_components, 1)),
                    "benchmark_json_path": artifact.benchmark_json_path,
                }
            )

    return component_rows, fullmap_rows


def _aggregate_rows(
    rows: list[dict[str, Any]],
    value_key: str,
) -> list[dict[str, Any]]:
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


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_runtime_scaling_experiment() -> dict[str, str]:
    os.makedirs(SCALING_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(
        "Running solver runtime scaling sweep with "
        f"dimensions={DIMENSIONS}, seeds={SEEDS}, particles={NUM_PARTICLES}"
    )

    artifacts: list[RunArtifact] = []
    for num_dimensions in DIMENSIONS:
        for seed in SEEDS:
            print(f"\n=== Scaling run: dim={num_dimensions}, seed={seed} ===")
            artifact = _run_single(seed=seed, num_dimensions=num_dimensions)
            print(f"Saved per-run benchmark JSON: {artifact.benchmark_json_path}")
            artifacts.append(artifact)

    component_rows, fullmap_rows = _collect_rows(artifacts=artifacts)
    component_agg = _aggregate_rows(component_rows, value_key="runtime_seconds")
    fullmap_agg = _aggregate_rows(fullmap_rows, value_key="total_runtime_seconds")

    raw_component_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_component_raw_TS={timestamp}.csv",
    )
    raw_fullmap_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_fullmap_raw_TS={timestamp}.csv",
    )
    agg_component_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_component_agg_TS={timestamp}.csv",
    )
    agg_fullmap_csv = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_fullmap_agg_TS={timestamp}.csv",
    )
    summary_json = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_scaling_summary_TS={timestamp}.json",
    )

    _write_csv(raw_component_csv, component_rows)
    _write_csv(raw_fullmap_csv, fullmap_rows)
    _write_csv(agg_component_csv, component_agg)
    _write_csv(agg_fullmap_csv, fullmap_agg)

    fullmap_plot_path = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_vs_dimension_fullmap_TS={timestamp}.png",
    )
    component_plot_path = os.path.join(
        SCALING_OUTPUT_DIR,
        f"solver_runtime_vs_dimension_component_TS={timestamp}.png",
    )
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

    summary_payload = {
        "metadata": {
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "dimensions": DIMENSIONS,
            "seeds": SEEDS,
            "num_particles": NUM_PARTICLES,
            "max_outer_iter": MAX_OUTER_ITER,
            "base_inner_iter": BASE_INNER_ITER,
            "max_inner_iters": MAX_INNER_ITERS,
            "run_solver_mode": "benchmark",
            "raw_run_count": len(artifacts),
        },
        "artifacts": {
            "raw_component_csv": raw_component_csv,
            "raw_fullmap_csv": raw_fullmap_csv,
            "agg_component_csv": agg_component_csv,
            "agg_fullmap_csv": agg_fullmap_csv,
            "fullmap_plot_png": fullmap_plot_path,
            "component_plot_png": component_plot_path,
            "per_run_benchmark_jsons": [a.benchmark_json_path for a in artifacts],
        },
    }
    _write_json(summary_json, summary_payload)

    print("\n=== Runtime scaling complete ===")
    print(f"Raw component CSV: {raw_component_csv}")
    print(f"Raw full-map CSV: {raw_fullmap_csv}")
    print(f"Aggregated component CSV: {agg_component_csv}")
    print(f"Aggregated full-map CSV: {agg_fullmap_csv}")
    print(f"Full-map plot: {fullmap_plot_path}")
    print(f"Per-component plot: {component_plot_path}")
    print(f"Summary JSON: {summary_json}")

    return {
        "raw_component_csv": raw_component_csv,
        "raw_fullmap_csv": raw_fullmap_csv,
        "agg_component_csv": agg_component_csv,
        "agg_fullmap_csv": agg_fullmap_csv,
        "fullmap_plot_png": fullmap_plot_path,
        "component_plot_png": component_plot_path,
        "summary_json": summary_json,
    }


if __name__ == "__main__":
    run_runtime_scaling_experiment()
