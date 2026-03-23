"""End-to-end benchmark for n-dimensional KR map components.
"""

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DykstraProjectionSolver, DykstraStallDetectionSolver
from utils import DykstraPlotter, DistributionPlotter
from utils import DataGenerator
from utils import HermiteBasis, KRMap
from utils import ProjectedGradientDescent


def _build_pgd_solver(
    projection_solver_class: type,
    learning_rate: float,
    max_outer_iter: int,
    gradient_clip_value: float | None,
    l1_reg: float,
    lr_decay: float,
    inexact_power: float,
    base_inner_iter: int,
    batch_size: int | None,
    rng_seed: int | None,
    prune_threshold: float,
    prune_interval: int,
    dykstra_kwargs: dict[str, Any],
    delete_spaces: bool = False,
) -> ProjectedGradientDescent:
    """Build a configured PGD solver instance for one component run."""
    solver_kwargs: dict[str, Any] = {
        "learning_rate": learning_rate,
        "max_outer_iter": max_outer_iter,
        "projection_solver_class": projection_solver_class,
        "gradient_clip_value": gradient_clip_value,
        "l1_reg": l1_reg,
        "lr_decay": lr_decay,
        "inexact_power": inexact_power,
        "base_inner_iter": base_inner_iter,
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "prune_threshold": prune_threshold,
        "prune_interval": prune_interval,
        **dykstra_kwargs,
    }
    if delete_spaces:
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
    """Append one solver's outputs to a component benchmark result dictionary."""
    component_result[f"w_{solver_label}"] = weights
    component_result[f"time_{solver_label}"] = elapsed
    component_result[f"objective_{solver_label}"] = history["objective_value"][-1]
    component_result[f"history_{solver_label}"] = history


def _print_component_timing(
    component_dim: int,
    num_dimensions: int,
    run_vanilla: bool,
    run_fast: bool,
    time_vanilla: float | None,
    time_fast: float | None,
    coeff_close: bool | None,
) -> None:
    """Print one-line timing summary for the active solver mode(s)."""
    if run_vanilla and run_fast and time_vanilla is not None and time_fast is not None:
        print(
            f"[Component {component_dim}/{num_dimensions}] "
            f"vanilla={time_vanilla:.4f}s, fast={time_fast:.4f}s, "
            f"coeff_close={coeff_close}"
        )
    elif run_vanilla and time_vanilla is not None:
        print(
            f"[Component {component_dim}/{num_dimensions}] "
            f"vanilla={time_vanilla:.4f}s"
        )
    elif run_fast and time_fast is not None:
        print(
            f"[Component {component_dim}/{num_dimensions}] "
            f"fast={time_fast:.4f}s"
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
    batch_size: int | None = None,
    rng_seed: int | None = None,
    prune_threshold: float = 0.0,
    prune_interval: int = 50,
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
    dykstra_kwargs : dict
        Extra keyword arguments forwarded to Dykstra solvers (e.g.
        ``track_error``, ``delete_spaces``).  Do not include ``max_iter``;
        it is set dynamically by the schedule.
    run_solver_mode : str
        Solver execution mode: ``"both"``, ``"vanilla"``, or ``"fast"``.
    gradient_clip_value : float | None
        Elementwise gradient clipping bound for PGD. If ``None``, clipping is
        disabled.
    l1_reg : float
        L1 regularisation strength passed to ``ProjectedGradientDescent``.
    lr_decay : float
        Learning-rate decay coefficient used in the schedule
        ``η_t = η_0 / (1 + lr_decay * t)``.
    inexact_power : float
        Exponent controlling how fast the inner Dykstra budget grows; see
        ``ProjectedGradientDescent`` for details.
    base_inner_iter : int
        Base number of inner Dykstra iterations; scaled by
        ``t**inexact_power`` at each outer step.
    plot_dykstra_iterates : bool
        Whether to plot and save per-component Dykstra iterate figures.
    batch_size : int or None, optional
        Mini-batch size for the stochastic gradient step.  When ``None``
        (default) the full gradient is used.
    rng_seed : int or None, optional
        Seed for the PGD mini-batch sampling RNG.  ``None`` gives a random
        seed.
    prune_threshold : float, optional
        Threshold for Iterative Hard Thresholding. Coefficients below this
        threshold are forcibly zeroed and locked. Default ``0.0`` (IHT disabled).
    prune_interval : int, optional
        Number of outer iterations between IHT checks. Default ``50``.
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
            pgd_vanilla = _build_pgd_solver(
                projection_solver_class=DykstraProjectionSolver,
                learning_rate=learning_rate,
                max_outer_iter=max_outer_iter,
                gradient_clip_value=gradient_clip_value,
                l1_reg=l1_reg,
                lr_decay=lr_decay,
                inexact_power=inexact_power,
                base_inner_iter=base_inner_iter,
                batch_size=batch_size,
                rng_seed=rng_seed,
                prune_threshold=prune_threshold,
                prune_interval=prune_interval,
                dykstra_kwargs=dykstra_kwargs,
            )
            w_vanilla, history_vanilla, time_vanilla = _run_component_optimisation(
                pgd_solver=pgd_vanilla,
                component_w_init=component_w_init,
                kr_model=kr_model,
                A=A,
                b=b,
            )

        w_fast = None
        history_fast = None
        time_fast = None
        if run_fast:
            pgd_fast = _build_pgd_solver(
                projection_solver_class=DykstraStallDetectionSolver,
                learning_rate=learning_rate,
                max_outer_iter=max_outer_iter,
                gradient_clip_value=gradient_clip_value,
                l1_reg=l1_reg,
                lr_decay=lr_decay,
                inexact_power=inexact_power,
                base_inner_iter=base_inner_iter,
                batch_size=batch_size,
                rng_seed=rng_seed,
                prune_threshold=prune_threshold,
                prune_interval=prune_interval,
                dykstra_kwargs=dykstra_kwargs,
                delete_spaces=True,
            )
            w_fast, history_fast, time_fast = _run_component_optimisation(
                pgd_solver=pgd_fast,
                component_w_init=component_w_init,
                kr_model=kr_model,
                A=A,
                b=b,
            )

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

        if (
            run_vanilla
            and w_vanilla is not None
            and history_vanilla is not None
            and time_vanilla is not None
        ):
            _append_solver_result(
                component_result=component_result,
                solver_label="vanilla",
                weights=w_vanilla,
                history=history_vanilla,
                elapsed=time_vanilla,
            )
        if (
            run_fast
            and w_fast is not None
            and history_fast is not None
            and time_fast is not None
        ):
            _append_solver_result(
                component_result=component_result,
                solver_label="fast",
                weights=w_fast,
                history=history_fast,
                elapsed=time_fast,
            )

        component_results.append(component_result)

        _print_component_timing(
            component_dim=component_dim,
            num_dimensions=num_dimensions,
            run_vanilla=run_vanilla,
            run_fast=run_fast,
            time_vanilla=time_vanilla,
            time_fast=time_fast,
            coeff_close=coeff_close,
        )

    return component_results


def build_distribution_filename(prefix: str) -> str:
    """Build a distribution-plot filename using shared experiment metadata."""
    return (
        f"{prefix}"
        f"SEED={SEED}_M={NUM_PARTICLES}_SGD={BATCH_SIZE}_PGDITERS={MAX_OUTER_ITER:,}_"
        f"DYKSTRA_ITERS={BASE_INNER_ITER}_{MAX_INNER_ITERS}_L1={L1_REG}_"
        f"LR={LEARNING_RATE:.0e}_{LR_DECAY:.0e}_IHT={PRUNE_INTERVAL}.png"
    )


def _plot_distribution_for_mode(
    solver_mode: str,
    distribution_plotter: DistributionPlotter,
    normal_samples: np.ndarray,
    z_samples: np.ndarray,
    results: list[dict[str, Any]],
    kr_map: KRMap,
    num_dimensions: int,
) -> None:
    """Plot mapped distributions for the selected solver mode."""
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
            filename=build_distribution_filename(
                f"kr{num_dimensions}d_distribution_comparison_"
            ),
            show=False,
        )
    elif solver_mode == "vanilla":
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
            filename=build_distribution_filename(
                f"kr{num_dimensions}d_distribution_vanilla_"
            ),
            show=False,
        )
    else:
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
            filename=build_distribution_filename(
                f"kr{num_dimensions}d_distribution_fast_"
            ),
            show=False,
        )

def run_benchmark() -> list[dict[str, Any]]:
    """Run the n-dimensional KR benchmark using module-level configuration."""
    solver_mode = RUN_SOLVER_MODE.lower()
    if solver_mode not in {"both", "vanilla", "fast"}:
        raise ValueError("RUN_SOLVER_MODE must be one of: 'both', 'vanilla', 'fast'.")

    if solver_mode != "both" and PLOT_DYKSTRA_ITERATES:
        raise ValueError(
            "PLOT_DYKSTRA_ITERATES=True is only valid when RUN_SOLVER_MODE='both'."
        )

    normal_samples, z_samples = DATA_GENERATOR.generate(
        num_particles=NUM_PARTICLES,
        num_dimensions=NUM_DIMENSIONS,
        seed=SEED,
    )

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
        lr_decay=LR_DECAY,
        inexact_power=INEXACT_POWER,
        base_inner_iter=BASE_INNER_ITER,
        plot_dykstra_iterates=PLOT_DYKSTRA_ITERATES,
        batch_size=BATCH_SIZE,
        rng_seed=RNG_SEED,
        prune_threshold=PRUNE_THRESHOLD,
        prune_interval=PRUNE_INTERVAL,
        enforce_matching=ENFORCE_MATCHING,
    )

    if PLOT_DISTRIBUTIONS:
        plot_output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "full_experiment_benchmarks"
        )
        distribution_plotter = DistributionPlotter(output_dir=plot_output_dir)
        _plot_distribution_for_mode(
            solver_mode=solver_mode,
            distribution_plotter=distribution_plotter,
            normal_samples=normal_samples,
            z_samples=z_samples,
            results=results,
            kr_map=KR_MAP,
            num_dimensions=NUM_DIMENSIONS,
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

    # SEED = int(time.time() * 1000) % 1000000
    SEED = 69420

    NUM_DIMENSIONS = 2
    NUM_PARTICLES = 300

    MAX_OUTER_ITER = 10
    DYKSTRA_KWARGS = {"track_error": False}
    GRADIENT_CLIP_VALUE = 10.0
    L1_REG = 0.0

    # Inexact projection: Inner iters = BASE_INNER_ITER * (outer_iter ** INEXACT_POWER)
    BASE_INNER_ITER = 10
    MAX_INNER_ITERS = 1000 # int(BASE_INNER_ITER * (MAX_OUTER_ITER ** INEXACT_POWER))
    INEXACT_POWER = np.log(MAX_INNER_ITERS - BASE_INNER_ITER) / np.log(MAX_OUTER_ITER) # 0 for fixed dykstra budget
    
    # SGD
    # BATCH_SIZE: int | None = None
    BATCH_SIZE: int | None = 100
    RNG_SEED: int | None = SEED + 1 if BATCH_SIZE is not None else None  # different seed
    LEARNING_RATE = 0.00001
    LR_DECAY = 1e-4 if BATCH_SIZE is not None else 0.0

    # IHT
    PRUNE_THRESHOLD = 1e-3
    PRUNE_INTERVAL = 50

    def SHEAR_FUNCTION(zeta: np.ndarray) -> np.ndarray:
        return zeta[:, 0] ** 2

    DATA_GENERATOR = DataGenerator(shear_function=SHEAR_FUNCTION)
    DEGREE = 2
    BASIS = HermiteBasis()
    KR_MAP = KRMap(
        degree=DEGREE,
        basis_1d=BASIS,
        log_epsilon=1e-8,
    )
    W_INIT: dict[int, np.ndarray] = {}
    for component_dim in range(1, NUM_DIMENSIONS + 1):
        W_INIT[component_dim] = KR_MAP.build_identity_initial_guess(component_dim)

    run_benchmark()
