"""End-to-end benchmark for n-dimensional KR map components.

Runs PGD + Dykstra benchmarks component-by-component for a Knothe-Rosenblatt
map in arbitrary dimension. For each component, this script compares vanilla
Dykstra against stall-detection Dykstra and saves one figure containing all
outer-iteration error traces.
"""

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DykstraPlotter
from utils import DykstraProjectionSolver, DykstraStallDetectionSolver
from utils import DistributionPlotter
from utils import HermiteBasis, TensorHermiteBasis
from utils import KRMapComponent
from utils import assemble_component_weights, evaluate_kr_map
from utils import build_identity_initial_guess
from utils import ProjectedGradientDescent
from utils import generate_crescent_data_2d

def benchmark_kr_map_components_nd(
    z: np.ndarray,
    num_dimensions: int,
    num_particles: int,
    seed: int,
    initial_guesses_by_component: dict[int, np.ndarray],
    learning_rate: float,
    max_outer_iter: int,
    dykstra_kwargs: dict[str, Any],
    degree: int,
    basis: Any,
    gradient_clip_value: float | None,
    plot_dykstra_iterates: bool,
    enforce_matching: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark each KR map component up to ``num_dimensions``.

    Parameters
    ----------
    z : np.ndarray
        Sample matrix with shape ``(M, num_dimensions)``.
    num_dimensions : int, optional
        Number of KR map components to benchmark.
    num_particles : int, optional
        Number of particles (used for summary labelling).
    seed : int, optional
        Random seed (used for summary labelling).
    initial_guesses_by_component : dict[int, np.ndarray]
        Initial coefficient vectors keyed by component dimension.
    learning_rate : float, optional
        PGD learning rate.
    max_outer_iter : int, optional
        Number of outer PGD iterations.
    dykstra_kwargs : dict, optional
        Keyword arguments forwarded to Dykstra solvers.
    degree : int, optional
        Maximum Hermite degree.
    basis : Any, optional
        Basis object for the first component. Higher components use
        ``TensorHermiteBasis``.
    gradient_clip_value : float | None
        Elementwise gradient clipping bound for PGD. If ``None``, clipping is
        disabled.
    plot_dykstra_iterates : bool
        Whether to plot and save per-component Dykstra iterate figures.
    enforce_matching : bool, optional
        If ``True``, raises when vanilla and fast-forward coefficients differ
        beyond tolerance.

    Returns
    -------
    list of dict
        Per-component benchmark outputs.
    """
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must have shape (M, num_dimensions).")
    if z.shape[1] < num_dimensions:
        raise ValueError("z has fewer columns than num_dimensions.")

    plot_output_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "dykstra_benchmarks"
    )
    plotter = DykstraPlotter(output_dir=plot_output_dir)

    component_results: list[dict[str, Any]] = []

    for component_dim in range(1, num_dimensions + 1):
        component_data = z[:, :component_dim]
        component_basis = basis if component_dim == 1 else TensorHermiteBasis()

        kr_model = KRMapComponent(
            data=component_data,
            basis=component_basis,
            degree=degree,
        )

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

        pgd_vanilla = ProjectedGradientDescent(
            learning_rate=learning_rate,
            max_outer_iter=max_outer_iter,
            projection_solver_class=DykstraProjectionSolver,
            gradient_clip_value=gradient_clip_value,
            **dykstra_kwargs,
        )
        t0 = time.perf_counter()
        w_vanilla, history_vanilla = pgd_vanilla.optimise(
            w_init=component_w_init,
            objective_fn=kr_model.objective,
            gradient_fn=kr_model.gradient,
            A_constraint=A,
            b_constraint=b,
        )
        time_vanilla = time.perf_counter() - t0

        pgd_fast = ProjectedGradientDescent(
            learning_rate=learning_rate,
            max_outer_iter=max_outer_iter,
            projection_solver_class=DykstraStallDetectionSolver,
            gradient_clip_value=gradient_clip_value,
            delete_spaces=True,
            **dykstra_kwargs,
        )
        t0 = time.perf_counter()
        w_fast, history_fast = pgd_fast.optimise(
            w_init=component_w_init,
            objective_fn=kr_model.objective,
            gradient_fn=kr_model.gradient,
            A_constraint=A,
            b_constraint=b,
        )
        time_fast = time.perf_counter() - t0

        coeff_close = bool(np.allclose(w_vanilla, w_fast, atol=1e-4))
        coeff_max_abs_diff = float(np.max(np.abs(w_vanilla - w_fast)))

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

        prefix = (
            f"kr{num_dimensions}d_component_{component_dim}_"
            f"SEED={seed}_M={num_particles}"
        )
        if plot_dykstra_iterates:
            plotter.plot_outer_iteration_solver_comparison(
                vanilla_results=history_vanilla["projection_results"],
                fast_forward_results=history_fast["projection_results"],
                filename_prefix=prefix,
                show=False,
            )

        component_results.append(
            {
                "component_dim": component_dim,
                "w_vanilla": w_vanilla,
                "w_fast": w_fast,
                "time_vanilla": time_vanilla,
                "time_fast": time_fast,
                "objective_vanilla": history_vanilla["objective_value"][-1],
                "objective_fast": history_fast["objective_value"][-1],
                "history_vanilla": history_vanilla,
                "history_fast": history_fast,
                "coefficients_close": coeff_close,
                "coefficients_max_abs_diff": coeff_max_abs_diff,
            }
        )

        print(
            f"[Component {component_dim}/{num_dimensions}] "
            f"vanilla={time_vanilla:.4f}s, fast={time_fast:.4f}s, "
            f"coeff_close={coeff_close}"
        )

    return component_results


def run_benchmark() -> list[dict[str, Any]]:
    """Run the n-dimensional KR benchmark using module-level configuration."""
    if NUM_DIMENSIONS == 2:
        normal_samples, z_samples = generate_crescent_data_2d(NUM_PARTICLES, seed=SEED)
    else:
        rng = np.random.default_rng(SEED)
        normal_samples = rng.normal(size=(NUM_PARTICLES, NUM_DIMENSIONS))
        z_samples = rng.normal(size=(NUM_PARTICLES, NUM_DIMENSIONS))

    results = benchmark_kr_map_components_nd(
        z=z_samples,
        num_dimensions=NUM_DIMENSIONS,
        num_particles=NUM_PARTICLES,
        seed=SEED,
        initial_guesses_by_component=W_INIT,
        learning_rate=LEARNING_RATE,
        max_outer_iter=MAX_OUTER_ITER,
        dykstra_kwargs=DYKSTRA_KWARGS,
        degree=DEGREE,
        basis=BASIS,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        plot_dykstra_iterates=PLOT_DYKSTRA_ITERATES,
        enforce_matching=ENFORCE_MATCHING,
    )

    if PLOT_DISTRIBUTION_COMPARISON:
        vanilla_weights = assemble_component_weights(results, "w_vanilla")
        fast_weights = assemble_component_weights(results, "w_fast")

        vanilla_mapped = evaluate_kr_map(
            z=z_samples[:, :NUM_DIMENSIONS],
            degree=DEGREE,
            weights_by_component=vanilla_weights,
            basis_1d=BASIS,
            tensor_basis=TensorHermiteBasis(),
        )
        fast_mapped = evaluate_kr_map(
            z=z_samples[:, :NUM_DIMENSIONS],
            degree=DEGREE,
            weights_by_component=fast_weights,
            basis_1d=BASIS,
            tensor_basis=TensorHermiteBasis(),
        )

        plot_output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "full_experiment_benchmarks"
        )
        distribution_plotter = DistributionPlotter(output_dir=plot_output_dir)
        distribution_plotter.plot_kr_map_distribution_comparison(
            normal_samples=normal_samples[:, :2],
            synthetic_samples=z_samples[:, :2],
            vanilla_mapped_samples=vanilla_mapped[:, :2],
            fast_mapped_samples=fast_mapped[:, :2],
            filename=(
                f"kr{NUM_DIMENSIONS}d_distribution_comparison_"
                f"SEED={SEED}_M={NUM_PARTICLES}.png"
            ),
            show=False,
        )

    print("\nCompleted n-dimensional KR component benchmark.")
    num_component_figures = len(results) if PLOT_DYKSTRA_ITERATES else 0
    num_distribution_figures = 1 if PLOT_DISTRIBUTION_COMPARISON else 0
    print(
        "Saved "
        f"{num_component_figures} component error figure(s) and "
        f"{num_distribution_figures} distribution comparison figure(s) "
        "in results/dykstra_benchmarks."
    )
    return results


if __name__ == "__main__":
    NUM_DIMENSIONS = 2
    NUM_PARTICLES = 1000
    SEED = 42

    LEARNING_RATE = 0.001
    MAX_OUTER_ITER = 10
    DYKSTRA_KWARGS = {"max_iter": 1000, "track_error": True}
    DEGREE = 3
    BASIS = HermiteBasis()
    GRADIENT_CLIP_VALUE = 10.0
    PLOT_DYKSTRA_ITERATES = False
    PLOT_DISTRIBUTION_COMPARISON = True

    W_INIT: dict[int, np.ndarray] = {}

    for component_dim in range(1, NUM_DIMENSIONS + 1):
        W_INIT[component_dim] = build_identity_initial_guess(
            component_dim=component_dim,
            degree=DEGREE,
        )

    ENFORCE_MATCHING = False

    run_benchmark()