"""End-to-end benchmark for n-dimensional KR map components.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DykstraProjectionSolver, DykstraStallDetectionSolver
from utils import DykstraPlotter, DistributionPlotter
from utils import DataGenerator
from utils import Basis, HermiteBasis, KRMap
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
    track_error_outer_iterations: list[int] | None = None,
    store_all_projection_results: bool = False,
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
        "track_error_outer_iterations": track_error_outer_iterations,
        "store_all_projection_results": store_all_projection_results,
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

def _to_json_safe(value: Any) -> Any:
    """Convert NumPy/Python containers into JSON-safe objects."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _serialise_projection_result(result: Any) -> dict[str, Any]:
    """Serialise one projection result object for JSON export."""
    return {
        "projection": _to_json_safe(getattr(result, "projection", None)),
        "squared_errors": _to_json_safe(getattr(result, "squared_errors", None)),
        "stalled_errors": _to_json_safe(getattr(result, "stalled_errors", None)),
        "converged_errors": _to_json_safe(getattr(result, "converged_errors", None)),
        "active_half_spaces": _to_json_safe(
            getattr(result, "active_half_spaces", None)
        ),
    }


def _serialise_history(history: dict[str, Any]) -> dict[str, Any]:
    """Serialise one PGD history dictionary for JSON export."""
    serialised: dict[str, Any] = {}
    for key, value in history.items():
        if key in {"projection_results", "projection_results_full"}:
            serialised[key] = [
                _serialise_projection_result(result)
                for result in value
            ]
        else:
            serialised[key] = _to_json_safe(value)
    return serialised


def _serialise_component_result(component_result: dict[str, Any]) -> dict[str, Any]:
    """Serialise one component benchmark result dictionary."""
    serialised: dict[str, Any] = {}
    for key, value in component_result.items():
        if key.startswith("history_") and isinstance(value, dict):
            serialised[key] = _serialise_history(value)
        else:
            serialised[key] = _to_json_safe(value)
    return serialised


def _save_full_run_iterates_json(
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
) -> str | None:
    """Write a full solver-trajectory JSON payload."""

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
            "dykstra_kwargs": _to_json_safe(dykstra_kwargs),
            "notes": {
                "weight_iterates": (
                    "Index 0 is the initial coefficient vector; index t is "
                    "the coefficients after outer iteration t."
                ),
                "projection_results_full": (
                    "Contains one Dykstra run per outer PGD iteration."
                ),
                "projection_result.squared_errors": (
                    "Per-inner-cycle Dykstra squared-error trajectory for "
                    "each outer PGD iteration."
                ),
            },
        },
        "components": [
            _serialise_component_result(component_result)
            for component_result in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


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
    plot_outer_iterations : list[int] or None, optional
        Optional list of outer PGD iteration indices for which Dykstra
        residual histories should be tracked and plotted. Negative indices
        follow Python conventions (e.g. ``-1`` is the final outer iteration).
        Out-of-range values are ignored. If ``None``, all outer iterations are
        tracked in plotting mode.
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
    store_full_projection_histories : bool, optional
        If ``True``, stores projection solver outputs for every outer PGD
        iteration in each solver history under
        ``projection_results_full``/``projection_outer_indices_full``.

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
    capture_full_histories = store_full_projection_histories
    dykstra_kwargs_effective = dict(dykstra_kwargs)
    if capture_full_histories:
        dykstra_kwargs_effective["track_error"] = True

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
                dykstra_kwargs=dykstra_kwargs_effective,
                track_error_outer_iterations=(
                    plot_outer_iterations if plot_dykstra_iterates else None
                ),
                store_all_projection_results=capture_full_histories,
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
                dykstra_kwargs=dykstra_kwargs_effective,
                track_error_outer_iterations=(
                    plot_outer_iterations if plot_dykstra_iterates else None
                ),
                store_all_projection_results=capture_full_histories,
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
            vanilla_outer_indices = history_vanilla["projection_outer_indices"]  # type: ignore[index]
            fast_outer_indices = history_fast["projection_outer_indices"]  # type: ignore[index]
            if list(vanilla_outer_indices) != list(fast_outer_indices):
                raise ValueError(
                    "Vanilla and fast solver tracked different outer iteration indices."
                )
            plotter.plot_outer_iteration_solver_comparison(
                vanilla_results=history_vanilla["projection_results"],  # type: ignore[index]
                fast_forward_results=history_fast["projection_results"],  # type: ignore[index]
                outer_indices=vanilla_outer_indices,
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

    dykstra_kwargs = dict(DYKSTRA_KWARGS)
    if SAVE_FULL_RUN_ITERATES_JSON:
        dykstra_kwargs["track_error"] = True

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
        dykstra_kwargs=dykstra_kwargs,
        run_solver_mode=solver_mode,
        gradient_clip_value=GRADIENT_CLIP_VALUE,
        l1_reg=L1_REG,
        lr_decay=LR_DECAY,
        inexact_power=INEXACT_POWER,
        base_inner_iter=BASE_INNER_ITER,
        plot_dykstra_iterates=PLOT_DYKSTRA_ITERATES,
        plot_outer_iterations=PLOT_DYKSTRA_OUTER_ITERATIONS,
        batch_size=BATCH_SIZE,
        rng_seed=RNG_SEED,
        prune_threshold=PRUNE_THRESHOLD,
        prune_interval=PRUNE_INTERVAL,
        enforce_matching=ENFORCE_MATCHING,
        store_full_projection_histories=SAVE_FULL_RUN_ITERATES_JSON,
    )

    full_run_json_path = None
    if SAVE_FULL_RUN_ITERATES_JSON:
        full_run_json_path = _save_full_run_iterates_json(
            results=results,
            solver_mode=solver_mode,
            output_dir=os.path.join(
                os.path.dirname(__file__),
                "..",
                "results",
                "full_experiment_benchmarks",
                "full_run_iterates",
            ),
            num_dimensions=NUM_DIMENSIONS,
            num_particles=NUM_PARTICLES,
            seed=SEED,
            max_outer_iter=MAX_OUTER_ITER,
            base_inner_iter=BASE_INNER_ITER,
            max_inner_iters=MAX_INNER_ITERS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            lr_decay=LR_DECAY,
            l1_reg=L1_REG,
            prune_interval=PRUNE_INTERVAL,
            prune_threshold=PRUNE_THRESHOLD,
            dykstra_kwargs=dykstra_kwargs,
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
    if full_run_json_path is not None:
        print(f"Saved full iterate JSON: {full_run_json_path}")

    print("\nMap weights:")
    for result in results:
        dim = result["component_dim"]
        if "w_vanilla" in result:
            print(f"  Component {dim} (vanilla): {result['w_vanilla']}")
        if "w_fast" in result:
            print(f"  Component {dim} (fast):    {result['w_fast']}")

    return results


if __name__ == "__main__":

    RUN_SOLVER_MODE: str = "fast"  # options: "both", "vanilla", "fast"
    SAVE_FULL_RUN_ITERATES_JSON: bool = True
    ENFORCE_MATCHING: bool = False
    PLOT_DYKSTRA_ITERATES: bool = False
    PLOT_DYKSTRA_OUTER_ITERATIONS: list[int] | None = [0, -2, -1] \
        if PLOT_DYKSTRA_ITERATES else None
    PLOT_DISTRIBUTIONS: bool = True

    # SEED = int(time.time() * 1000) % 1000000
    SEED: int = 69

    NUM_DIMENSIONS: int = 6
    NUM_PARTICLES: int = 20

    MAX_OUTER_ITER: int = 4
    DYKSTRA_KWARGS: dict = {"track_error": False}
    GRADIENT_CLIP_VALUE: float = 10.0
    L1_REG: float = 0.05

    # Inexact projection: Inner iters = BASE_INNER_ITER * (outer_iter ** INEXACT_POWER)
    BASE_INNER_ITER: int = 10
    MAX_INNER_ITERS: int = 10 # int(BASE_INNER_ITER * (MAX_OUTER_ITER ** INEXACT_POWER))
    INEXACT_POWER: float = np.log(MAX_INNER_ITERS / BASE_INNER_ITER) / np.log(MAX_OUTER_ITER) # 0 for fixed dykstra budget
    
    # SGD
    # BATCH_SIZE: int | None = None
    BATCH_SIZE: int | None = None
    RNG_SEED: int | None = SEED + 1 \
        if BATCH_SIZE is not None else None # different seed
    LEARNING_RATE: float = 1
    LR_DECAY: float = 1e-4 \
        if BATCH_SIZE is not None else 0.0

    # IHT
    PRUNE_THRESHOLD: float = 1e-4
    PRUNE_INTERVAL: int = 100

    def SHEAR_FUNCTION(zeta: np.ndarray) -> np.ndarray:
        return zeta[:, 0] ** 2

    DATA_GENERATOR = DataGenerator(shear_function=SHEAR_FUNCTION)
    DEGREE: int = 2
    BASIS: Basis = HermiteBasis()
    KR_MAP: KRMap = KRMap(
        degree=DEGREE,
        basis_1d=BASIS,
        log_epsilon=1e-8,
    )
    W_INIT: dict[int, np.ndarray] = {}
    for component_dim in range(1, NUM_DIMENSIONS + 1):
        W_INIT[component_dim] = KR_MAP.build_identity_initial_guess(component_dim)

    run_benchmark()
