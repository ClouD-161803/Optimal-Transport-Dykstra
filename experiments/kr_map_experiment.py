"""End-to-end benchmark entrypoint for n-dimensional KR map components.

This module is intentionally thin: orchestration now lives under ``core/``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import cast
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import ExperimentConfig, OptimizationConfig, PlotConfig, RunConfig, SolverMode
from core.runner import (
    build_identity_initial_guesses,
    run_component_benchmark,
    run_synthetic_experiment,
)
from utils.data_generator import (
    BoomerangShearFunction,
    CubicShearFunction,
    DataGenerator,
    GVMDataGenerator,
    RoughLineShearFunction,
    XShapedShearFunction,
)
from utils.optimal_transport import Basis, HermiteBasis, KRMap


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Run-mode settings
RUN_SOLVER_MODE: str = "fast"  # options: "both", "vanilla", "fast", "benchmark"
SAVE_FULL_RUN_ITERATES: bool = False
SAVE_DISTRIBUTION_SHIFT_MEDIA: bool = False
ENFORCE_MATCHING: bool = False

# Plot settings
PLOT_DYKSTRA_ITERATES: bool = False
PLOT_DYKSTRA_OUTER_ITERATIONS: list[int] | None = (
    [0, -2, -1] if PLOT_DYKSTRA_ITERATES else None
)
PLOT_DISTRIBUTIONS: bool = True
PLOT_SIZE: float = 7.0 if PLOT_DISTRIBUTIONS else 0.0
X_LIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE) if PLOT_DISTRIBUTIONS else None
Y_LIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE) if PLOT_DISTRIBUTIONS else None

# SEED = int(time.time() * 1000) % 1000000
SEED: int = 0
NUM_DIMENSIONS: int = 2
NUM_PARTICLES: int = 500

# Optimisation settings
MAX_OUTER_ITER: int = 1000
DYKSTRA_KWARGS: dict[str, Any] = {"track_error": False}
GRADIENT_CLIP_VALUE: float = 10.0
L1_REG: float = 0.0

# Inexact projection: Inner iters = BASE_INNER_ITER * (outer_iter ** INEXACT_POWER)
BASE_INNER_ITER: int = 1
MAX_INNER_ITERS: int = 10
INEXACT_POWER: float = np.log(MAX_INNER_ITERS / BASE_INNER_ITER) / np.log(MAX_OUTER_ITER)

# SGD
BATCH_SIZE: int | None = 100
RNG_SEED: int | None = SEED + 1 if BATCH_SIZE is not None else None
LEARNING_RATE: float = 0.1
LR_DECAY: float = 1e-2 if BATCH_SIZE is not None else 0.0

# IHT
PRUNE_THRESHOLD: float = 1e-2
PRUNE_INTERVAL: int = 100

# Data generation (uncomment one)
GVM_ALPHA: float = -3.2
GVM_BETA: np.ndarray = np.array([0.0, 0.0], dtype=float)
GVM_GAMMA: np.ndarray = np.array([[4.2, 0.0], [0.0, 0.0]], dtype=float)
GVM_KAPPA: float = 7.0
LINE_SIGMA: float = 0.15

# Distribution constraints
# DATA_HALFSPACE_A: np.ndarray = np.array([[0.0, 1.0]], dtype=float)
# DATA_HALFSPACE_B: np.ndarray = np.array([5.0], dtype=float)
DATA_HALFSPACE_A: np.ndarray | None = None
DATA_HALFSPACE_B: np.ndarray | None = None

# DATA_GENERATOR = GVMDataGenerator(
#     alpha=GVM_ALPHA,
#     beta=GVM_BETA,
#     gamma=GVM_GAMMA,
#     kappa=GVM_KAPPA,
#     halfspace_A=DATA_HALFSPACE_A if "DATA_HALFSPACE_A" in globals() else None,
#     halfspace_b=DATA_HALFSPACE_B if "DATA_HALFSPACE_B" in globals() else None,
# )
DATA_GENERATOR = DataGenerator(
    shear_function=BoomerangShearFunction(),
)
# DATA_GENERATOR = DataGenerator(
#     shear_function=RoughLineShearFunction(sigma=LINE_SIGMA),
# )
# DATA_GENERATOR = DataGenerator(
#     shear_function=CubicShearFunction(strength=0.7),  # Suggested DEGREE: 3
# )
# DATA_GENERATOR = DataGenerator(
#     shear_function=XShapedShearFunction(strength=1.2),  # Suggested DEGREE: 2
# )

DEGREE: int = 2
BASIS: Basis = HermiteBasis()
KR_MAP: KRMap = KRMap(
    degree=DEGREE,
    basis_1d=BASIS,
    log_epsilon=1e-8,
)
W_INIT: dict[int, np.ndarray] = build_identity_initial_guesses(
    kr_map=KR_MAP,
    num_dimensions=NUM_DIMENSIONS,
)


def _build_experiment_config() -> ExperimentConfig:
    """Build a typed config from module-level constants."""
    if RUN_SOLVER_MODE not in {"both", "vanilla", "fast", "benchmark"}:
        raise ValueError(
            "RUN_SOLVER_MODE must be one of: 'both', 'vanilla', 'fast', 'benchmark'."
        )
    validated_run_solver_mode = cast(SolverMode, RUN_SOLVER_MODE)

    run_config = RunConfig(
        run_solver_mode=validated_run_solver_mode,
        save_full_run_iterates=(
            SAVE_FULL_RUN_ITERATES or SAVE_DISTRIBUTION_SHIFT_MEDIA
        ),
        save_distribution_shift_media=SAVE_DISTRIBUTION_SHIFT_MEDIA,
        enforce_matching=ENFORCE_MATCHING,
    )
    plot_config = PlotConfig(
        plot_dykstra_iterates=PLOT_DYKSTRA_ITERATES,
        plot_dykstra_outer_iterations=PLOT_DYKSTRA_OUTER_ITERATIONS,
        plot_distributions=PLOT_DISTRIBUTIONS,
        x_lim=X_LIM,
        y_lim=Y_LIM,
    )
    optimization_kwargs: dict[str, Any] = {
        "learning_rate": LEARNING_RATE,
        "max_outer_iter": MAX_OUTER_ITER,
        "gradient_clip_value": GRADIENT_CLIP_VALUE,
        "l1_reg": L1_REG,
        "lr_decay": LR_DECAY,
        "inexact_power": INEXACT_POWER,
        "base_inner_iter": BASE_INNER_ITER,
        "max_inner_iters": MAX_INNER_ITERS,
        "batch_size": BATCH_SIZE,
        "rng_seed": RNG_SEED,
        "prune_threshold": PRUNE_THRESHOLD,
        "prune_interval": PRUNE_INTERVAL,
        "dykstra_kwargs": dict(DYKSTRA_KWARGS),
    }
    optimization_config = OptimizationConfig(
        **optimization_kwargs,
    )
    return ExperimentConfig(
        seed=SEED,
        num_dimensions=NUM_DIMENSIONS,
        num_particles=NUM_PARTICLES,
        run=run_config,
        plot=plot_config,
        optimization=optimization_config,
    )


def benchmark_kr_map_components_nd(
    z: np.ndarray,
    num_dimensions: int,
    num_particles: int,
    seed: int,
    kr_map: KRMap,
    initial_guesses_by_component: dict[int, np.ndarray],
    learning_rate: float,
    max_outer_iter: int,
    dykstra_kwargs: dict[str, Any],
    run_solver_mode: str,
    gradient_clip_value: float | None,
    l1_reg: float,
    lr_decay: float,
    inexact_power: float,
    base_inner_iter: int,
    plot_dykstra_iterates: bool,
    plot_outer_iterations: list[int] | None = None,
    batch_size: int | None = None,
    rng_seed: int | None = None,
    prune_threshold: float = 0.0,
    prune_interval: int = 50,
    enforce_matching: bool = False,
    store_full_projection_histories: bool = False,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around the refactored core benchmark loop."""
    if run_solver_mode not in {"both", "vanilla", "fast", "benchmark"}:
        raise ValueError(
            "run_solver_mode must be one of: 'both', 'vanilla', 'fast', 'benchmark'."
        )
    validated_run_solver_mode = cast(SolverMode, run_solver_mode)

    optimization_kwargs: dict[str, Any] = {
        "learning_rate": learning_rate,
        "max_outer_iter": max_outer_iter,
        "gradient_clip_value": gradient_clip_value,
        "l1_reg": l1_reg,
        "lr_decay": lr_decay,
        "inexact_power": inexact_power,
        "base_inner_iter": base_inner_iter,
        "max_inner_iters": int(base_inner_iter * (max_outer_iter**max(inexact_power, 0.0))),
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "prune_threshold": prune_threshold,
        "prune_interval": prune_interval,
        "dykstra_kwargs": dict(dykstra_kwargs),
    }
    optimization_config = OptimizationConfig(**optimization_kwargs)
    return run_component_benchmark(
        z=z,
        num_dimensions=num_dimensions,
        num_particles=num_particles,
        seed=seed,
        kr_map=kr_map,
        initial_guesses_by_component=initial_guesses_by_component,
        optimization_config=optimization_config,
        run_solver_mode=validated_run_solver_mode,
        plot_dykstra_iterates=plot_dykstra_iterates,
        plot_outer_iterations=plot_outer_iterations,
        enforce_matching=enforce_matching,
        store_full_projection_histories=store_full_projection_histories,
        dykstra_plot_output_dir=os.path.join(PROJECT_ROOT, "results", "dykstra_benchmarks"),
    )


def run_benchmark() -> list[dict[str, Any]]:
    """Run the KR benchmark using module-level configuration."""
    config = _build_experiment_config()
    summary = run_synthetic_experiment(
        project_root=PROJECT_ROOT,
        config=config,
        generator=DATA_GENERATOR,
        kr_map=KR_MAP,
        initial_guesses_by_component=W_INIT,
    )
    return summary.results


if __name__ == "__main__":
    run_benchmark()

