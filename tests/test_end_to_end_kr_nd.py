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
from utils import HermiteBasis
from utils import KRMap
from utils import ProjectedGradientDescent
from utils import generate_crescent_data_2d

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
    kr_map : KRMap
        KR map orchestrator used to create component models.
    initial_guesses_by_component : dict[int, np.ndarray]
        Initial coefficient vectors keyed by component dimension.
    learning_rate : float, optional
        PGD learning rate.
    max_outer_iter : int, optional
        Number of outer PGD iterations.
    dykstra_kwargs : dict, optional
        Keyword arguments forwarded to Dykstra solvers.
    run_solver_mode : str
        Solver execution mode: ``"both"``, ``"vanilla"``, or ``"fast"``.
    gradient_clip_value : float | None
        Elementwise gradient clipping bound for PGD. If ``None``, clipping is
        disabled.
    l1_reg : float
        L1 regularisation strength passed to ``ProjectedGradientDescent``.
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

    run_solver_mode = run_solver_mode.lower()
    if run_solver_mode not in {"both", "vanilla", "fast"}:
        raise ValueError("run_solver_mode must be one of: 'both', 'vanilla', 'fast'.")

    run_vanilla = run_solver_mode in {"both", "vanilla"}
    run_fast = run_solver_mode in {"both", "fast"}

    plot_output_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "dykstra_benchmarks"
    )
    plotter = DykstraPlotter(output_dir=plot_output_dir)

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

        w_vanilla = None
        history_vanilla = None
        time_vanilla = None
        if run_vanilla:
            pgd_vanilla = ProjectedGradientDescent(
                learning_rate=learning_rate,
                max_outer_iter=max_outer_iter,
                projection_solver_class=DykstraProjectionSolver,
                gradient_clip_value=gradient_clip_value,
                l1_reg=l1_reg,
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

        w_fast = None
        history_fast = None
        time_fast = None
        if run_fast:
            pgd_fast = ProjectedGradientDescent(
                learning_rate=learning_rate,
                max_outer_iter=max_outer_iter,
                projection_solver_class=DykstraStallDetectionSolver,
                gradient_clip_value=gradient_clip_value,
                l1_reg=l1_reg,
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

        coeff_close = None
        coeff_max_abs_diff = None
        if run_vanilla and run_fast and w_vanilla is not None and w_fast is not None:
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
        if plot_dykstra_iterates and run_vanilla and run_fast:
            plotter.plot_outer_iteration_solver_comparison(
                vanilla_results=history_vanilla["projection_results"],  # type: ignore[index]
                fast_forward_results=history_fast["projection_results"],  # type: ignore[index]
                filename_prefix=prefix,
                show=False,
            )

        component_result: dict[str, Any] = {
            "component_dim": component_dim,
            "coefficients_close": coeff_close,
            "coefficients_max_abs_diff": coeff_max_abs_diff,
        }

        if run_vanilla and w_vanilla is not None and history_vanilla is not None:
            component_result["w_vanilla"] = w_vanilla
            component_result["time_vanilla"] = time_vanilla
            component_result["objective_vanilla"] = history_vanilla["objective_value"][-1]
            component_result["history_vanilla"] = history_vanilla
        if run_fast and w_fast is not None and history_fast is not None:
            component_result["w_fast"] = w_fast
            component_result["time_fast"] = time_fast
            component_result["objective_fast"] = history_fast["objective_value"][-1]
            component_result["history_fast"] = history_fast

        component_results.append(component_result)

        if run_vanilla and run_fast:
            print(
                f"[Component {component_dim}/{num_dimensions}] "
                f"vanilla={time_vanilla:.4f}s, fast={time_fast:.4f}s, "
                f"coeff_close={coeff_close}"
            )
        elif run_vanilla:
            print(
                f"[Component {component_dim}/{num_dimensions}] "
                f"vanilla={time_vanilla:.4f}s"
            )
        else:
            print(
                f"[Component {component_dim}/{num_dimensions}] "
                f"fast={time_fast:.4f}s"
            )

    return component_results


def run_benchmark() -> list[dict[str, Any]]:
    """Run the n-dimensional KR benchmark using module-level configuration."""
    solver_mode = RUN_SOLVER_MODE.lower()
    if solver_mode not in {"both", "vanilla", "fast"}:
        raise ValueError("RUN_SOLVER_MODE must be one of: 'both', 'vanilla', 'fast'.")

    if solver_mode != "both" and PLOT_DYKSTRA_ITERATES:
        raise ValueError(
            "PLOT_DYKSTRA_ITERATES=True is only valid when RUN_SOLVER_MODE='both'."
        )

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
        kr_map=KR_MAP,
        initial_guesses_by_component=W_INIT,
        learning_rate=LEARNING_RATE,
        max_outer_iter=MAX_OUTER_ITER,
        dykstra_kwargs=DYKSTRA_KWARGS,
        run_solver_mode=solver_mode,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        l1_reg=L1_REG,
        plot_dykstra_iterates=PLOT_DYKSTRA_ITERATES,
        enforce_matching=ENFORCE_MATCHING,
    )

    if PLOT_DISTRIBUTIONS:
        plot_output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "full_experiment_benchmarks"
        )
        distribution_plotter = DistributionPlotter(output_dir=plot_output_dir)
        if solver_mode == "both":
            vanilla_weights = KR_MAP.assemble_component_weights(results, "w_vanilla")
            fast_weights = KR_MAP.assemble_component_weights(results, "w_fast")

            vanilla_mapped = KR_MAP.evaluate(
                z=z_samples[:, :NUM_DIMENSIONS],
                weights_by_component=vanilla_weights,
            )
            fast_mapped = KR_MAP.evaluate(
                z=z_samples[:, :NUM_DIMENSIONS],
                weights_by_component=fast_weights,
            )

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
        elif solver_mode == "vanilla":
            vanilla_weights = KR_MAP.assemble_component_weights(results, "w_vanilla")
            vanilla_mapped = KR_MAP.evaluate(
                z=z_samples[:, :NUM_DIMENSIONS],
                weights_by_component=vanilla_weights,
            )
            distribution_plotter.plot_kr_map_distribution_single_solver(
                normal_samples=normal_samples[:, :2],
                synthetic_samples=z_samples[:, :2],
                mapped_samples=vanilla_mapped[:, :2],
                solver_label="vanilla Dykstra",
                filename=(
                    f"kr{NUM_DIMENSIONS}d_distribution_vanilla_"
                    f"SEED={SEED}_M={NUM_PARTICLES}.png"
                ),
                show=False,
            )
        else:
            fast_weights = KR_MAP.assemble_component_weights(results, "w_fast")
            fast_mapped = KR_MAP.evaluate(
                z=z_samples[:, :NUM_DIMENSIONS],
                weights_by_component=fast_weights,
            )
            distribution_plotter.plot_kr_map_distribution_single_solver(
                normal_samples=normal_samples[:, :2],
                synthetic_samples=z_samples[:, :2],
                mapped_samples=fast_mapped[:, :2],
                solver_label="fast-forward Dykstra",
                filename=(
                    f"kr{NUM_DIMENSIONS}d_distribution_fast_"
                    f"SEED={SEED}_M={NUM_PARTICLES}.png"
                ),
                show=False,
            )

    print(f"\nCompleted {NUM_DIMENSIONS}-dimensional KR component benchmark with seed {SEED}.")
    num_component_figures = (
        len(results)
        if (PLOT_DYKSTRA_ITERATES and solver_mode == "both")
        else 0
    )
    num_distribution_figures = 1 if PLOT_DISTRIBUTIONS else 0
    print(
        "Saved "
        f"{num_component_figures} component error figure(s) and "
        f"{num_distribution_figures} distribution comparison figure(s) "
        "in results/full_experiment_benchmarks."
    )

    print("\nMap weights:")
    for result in results:
        dim = result["component_dim"]
        if "w_vanilla" in result:
            print(f"  Component {dim} (vanilla): {result['w_vanilla']}")
        if "w_fast" in result:
            print(f"  Component {dim} (fast):    {result['w_fast']}")

    return results


if __name__ == "__main__":
    RUN_SOLVER_MODE = "fast"  # options: "both", "vanilla", "fast"
    ENFORCE_MATCHING = False
    PLOT_DYKSTRA_ITERATES = False
    PLOT_DISTRIBUTIONS = True

    SEED = int(time.time() * 1000) % 1000000
    # SEED = 42

    NUM_DIMENSIONS = 2
    NUM_PARTICLES = 500

    LEARNING_RATE = 0.1
    MAX_OUTER_ITER = 1000
    DYKSTRA_KWARGS = {"max_iter": 1000, "track_error": False}
    DEGREE = 3
    BASIS = HermiteBasis()
    KR_MAP = KRMap(
        degree=DEGREE,
        basis_1d=BASIS,
        log_epsilon=1e-8,
    )
    GRADIENT_CLIP_VALUE = 15.0
    L1_REG = 0.5

    W_INIT: dict[int, np.ndarray] = {}
    for component_dim in range(1, NUM_DIMENSIONS + 1):
        W_INIT[component_dim] = KR_MAP.build_identity_initial_guess(component_dim)

    run_benchmark()
