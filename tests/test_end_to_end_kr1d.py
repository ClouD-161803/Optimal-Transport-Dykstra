"""End-to-end test for the 1D Knothe-Rosenblatt map optimisation pipeline.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import generate_crescent_data_2d
from utils import HermiteBasis, KRMap1D
from utils import ProjectedGradientDescent
from utils import DykstraPlotter
from utils import (
    DykstraProjectionSolver,
    DykstraStallDetectionSolver,
)


def test_dykstra_fast_forward_advantage() -> None:
    """Compare vanilla Dykstra and stall-detection Dykstra inside PGD.

    The test generates synthetic crescent data, builds a degree-3 Hermite
    KR map, and runs Projected Gradient Descent twice — once with the
    standard ``DykstraProjectionSolver`` and once with the
    ``DykstraStallDetectionSolver`` (with inactive half-space removal
    enabled). It then asserts that both solvers converge to the same
    coefficient vector, and saves one figure with selected per-outer-iteration
    squared-error comparisons.
    """
    # Step 1 – Data
    num_particles: int = 500
    seed: int = 42
    _, z = generate_crescent_data_2d(num_particles, seed=seed)
    z1: np.ndarray = z[:, 0]

    # Step 2 – Model
    degree: int = 3
    basis = HermiteBasis()
    kr_model = KRMap1D(data=z1, basis=basis, degree=degree)

    # Step 3 – Constraints
    A, b = kr_model.get_polyhedral_constraints(epsilon=1e-4)

    # Step 4 – Initial guess (identity map: S(z) = z)
    w_init: np.ndarray = np.array([-10., 10., 10., -10.5])

    learning_rate: float = 0.01
    max_outer_iter: int = 20
    dykstra_kwargs: dict = {"track_error": False}
    plot_outer_iterations: list[int] | None = [0, 1, 2, 4]

    # Step 5a – Vanilla Dykstra
    pgd_vanilla = ProjectedGradientDescent(
        learning_rate=learning_rate,
        max_outer_iter=max_outer_iter,
        projection_solver_class=DykstraProjectionSolver,
        track_error_outer_iterations=plot_outer_iterations,
        **dykstra_kwargs,
    )

    t0 = time.perf_counter()
    w_vanilla, history_vanilla = pgd_vanilla.optimise(
        w_init=w_init,
        objective_fn=kr_model.objective,
        gradient_fn=kr_model.gradient,
        A_constraint=A,
        b_constraint=b,
    )
    time_vanilla: float = time.perf_counter() - t0

    # Step 5b – Fast-forward (stall-detection) Dykstra
    pgd_fast = ProjectedGradientDescent(
        learning_rate=learning_rate,
        max_outer_iter=max_outer_iter,
        projection_solver_class=DykstraStallDetectionSolver,
        track_error_outer_iterations=plot_outer_iterations,
        delete_spaces=True,
        **dykstra_kwargs,
    )

    t0 = time.perf_counter()
    w_fast, history_fast = pgd_fast.optimise(
        w_init=w_init,
        objective_fn=kr_model.objective,
        gradient_fn=kr_model.gradient,
        A_constraint=A,
        b_constraint=b,
    )
    time_fast: float = time.perf_counter() - t0

    # Step 6 – Assertions and summary
    np.testing.assert_allclose(
        w_vanilla,
        w_fast,
        atol=1e-4,
        err_msg="Vanilla and fast-forward Dykstra produced different coefficients.",
    )

    obj_vanilla: float = history_vanilla["objective_value"][-1]
    obj_fast: float = history_fast["objective_value"][-1]
    expected_outer_indices: list[int] = []
    if plot_outer_iterations is None:
        expected_outer_indices = list(range(max_outer_iter))
    else:
        for raw_idx in plot_outer_iterations:
            idx = raw_idx + max_outer_iter if raw_idx < 0 else raw_idx
            if 0 <= idx < max_outer_iter and idx not in expected_outer_indices:
                expected_outer_indices.append(idx)

    assert history_vanilla["projection_outer_indices"] == expected_outer_indices
    assert history_fast["projection_outer_indices"] == expected_outer_indices
    assert len(history_vanilla["projection_results"]) == len(expected_outer_indices)
    assert len(history_fast["projection_results"]) == len(expected_outer_indices)

    plot_output_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "dykstra_benchmarks"
    )
    plotter = DykstraPlotter(output_dir=plot_output_dir)
    plotter.plot_outer_iteration_solver_comparison(
        vanilla_results=history_vanilla["projection_results"],
        fast_forward_results=history_fast["projection_results"],
        outer_indices=history_vanilla["projection_outer_indices"],
        filename_prefix=(
            f"kr1d_outer_iter_comparison_SEED={seed}_M={num_particles}"
        ),
        show=False,
    )

    print("\n")
    print("  End-to-End KR1D Test — Dykstra Solver Comparison")
    print("")
    print(f"  Particles  : {num_particles}")
    print(f"  Degree     : {degree}")
    print(f"  Outer iters: {max_outer_iter}")
    print(f"  Inner iters: inexact schedule (base_tol=1e-3, power=1.1)")
    print("")
    print(f"  {'Solver':<28} {'Time (s)':>10} {'Final obj':>12}")
    print("")
    print(f"  {'Vanilla Dykstra':<28} {time_vanilla:>10.4f} {obj_vanilla:>12.6f}")
    print(f"  {'Stall-Detection Dykstra':<28} {time_fast:>10.4f} {obj_fast:>12.6f}")
    print("")

    if time_vanilla > 0:
        speedup: float = time_vanilla / time_fast
        print(f"  Speedup: {speedup:.2f}x")
    print("")

    print(f"\n  w_vanilla = {np.array2string(w_vanilla, precision=6)}")
    print(f"  w_fast    = {np.array2string(w_fast, precision=6)}")
    print("\n  Assertion passed: both solvers converged to the same solution.")


if __name__ == "__main__":
    test_dykstra_fast_forward_advantage()
