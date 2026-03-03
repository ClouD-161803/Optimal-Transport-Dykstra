"""End-to-end test for the 1D Knothe-Rosenblatt map optimisation pipeline.

Compares the standard Dykstra projection solver against the stall-detection
(fast-forward) variant within a Projected Gradient Descent loop, verifying
that both reach the same optimised coefficients while highlighting the
wall-clock speedup from avoiding inner-loop stalling.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import generate_crescent_data_2d
from utils import HermiteBasis, KRMap1D
from utils import ProjectedGradientDescent
from utils import ProjectPlotter
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
    coefficient vector, and saves one figure with all per-outer-iteration
    squared-error comparisons.
    """
    # Step 1 – Data
    num_particles: int = 500
    _, z = generate_crescent_data_2d(num_particles, seed=43)
    z1: np.ndarray = z[:, 0]

    # Step 2 – Model
    degree: int = 3
    basis = HermiteBasis()
    kr_model = KRMap1D(data=z1, basis=basis, degree=degree)

    # Step 3 – Constraints
    A, b = kr_model.get_polyhedral_constraints(epsilon=5e-1)

    # Step 4 – Initial guess (identity map: S(z) = z)
    w_init: np.ndarray = np.array([3.5, -4.5, 3.0, -3.0])

    learning_rate: float = 0.5
    max_outer_iter: int = 3
    dykstra_kwargs: dict = {"max_iter": 1000, "track_error": True}

    # Step 5a – Vanilla Dykstra
    pgd_vanilla = ProjectedGradientDescent(
        learning_rate=learning_rate,
        max_outer_iter=max_outer_iter,
        projection_solver_class=DykstraProjectionSolver,
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

    plot_output_dir = os.path.join(
        os.path.dirname(__file__), "..", "results", "dykstra_benchmarks"
    )
    plotter = ProjectPlotter(output_dir=plot_output_dir)
    plotter.plot_outer_iteration_solver_comparison(
        vanilla_results=history_vanilla["projection_results"],
        fast_forward_results=history_fast["projection_results"],
        suptitle="KR1D PGD Inner Dykstra Error",
        filename_prefix="kr1d_outer_iter_comparison",
        show=False,
    )

    print("\n" + "=" * 65)
    print("  End-to-End KR1D Test — Dykstra Solver Comparison")
    print("=" * 65)
    print(f"  Particles  : {num_particles}")
    print(f"  Degree     : {degree}")
    print(f"  Outer iters: {max_outer_iter}")
    print(f"  Inner iters: {dykstra_kwargs['max_iter']}  (per outer step)")
    print("-" * 65)
    print(f"  {'Solver':<28} {'Time (s)':>10} {'Final obj':>12}")
    print("-" * 65)
    print(f"  {'Vanilla Dykstra':<28} {time_vanilla:>10.4f} {obj_vanilla:>12.6f}")
    print(f"  {'Stall-Detection Dykstra':<28} {time_fast:>10.4f} {obj_fast:>12.6f}")
    print("-" * 65)

    if time_vanilla > 0:
        speedup: float = time_vanilla / time_fast
        print(f"  Speedup: {speedup:.2f}x")
    print("=" * 65)

    print(f"\n  w_vanilla = {np.array2string(w_vanilla, precision=6)}")
    print(f"  w_fast    = {np.array2string(w_fast, precision=6)}")
    print("\n  Assertion passed: both solvers converged to the same solution.")


if __name__ == "__main__":
    test_dykstra_fast_forward_advantage()
