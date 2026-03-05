"""Benchmark: Standard Dykstra vs Stall-Detection (Fast-Forward) Dykstra."""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DykstraProjectionSolver, DykstraStallDetectionSolver
from utils import DykstraPlotter

def run_benchmark():

    rng = np.random.default_rng(SEED)

    x_star = rng.standard_normal(DIM)

    N = rng.standard_normal((NUM_HALFSPACES, DIM))
    N = N / np.linalg.norm(N, axis=1, keepdims=True)

    c = N @ x_star + MARGIN

    z = x_star + 10.0 * rng.standard_normal(DIM)

    print(f"Problem: dim={DIM}, halfspaces={NUM_HALFSPACES}, max_iter={MAX_ITER}")
    print(f"Starting distance from x_star: {np.linalg.norm(z - x_star):.4f}")
    print()

    t0 = time.perf_counter()
    solver_std = DykstraProjectionSolver(
        z, N, c, max_iter=MAX_ITER, track_error=True, min_error=MIN_ERROR,
    )
    result_std = solver_std.solve()
    time_std = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver_ff = DykstraStallDetectionSolver(
        z, N, c, max_iter=MAX_ITER, track_error=True, min_error=MIN_ERROR, delete_spaces=True,
    )
    result_ff = solver_ff.solve()
    time_ff = time.perf_counter() - t0

    print("")
    print(f"{'Solver':<30} {'Time (s)':>10} {'Final error':>15}")
    print("")
    print(f"{'Standard Dykstra':<30} {time_std:>10.4f} {result_std.squared_errors[-1]:>15.6e}")  # type: ignore
    print(f"{'Stall-Detection (FF)':<30} {time_ff:>10.4f} {result_ff.squared_errors[-1]:>15.6e}")  # type: ignore
    print("")
    print()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "dykstra_benchmarks")
    plotter = DykstraPlotter(output_dir=out_dir)

    plotter.plot_convergence_comparison(
        results=[result_std, result_ff],
        labels=["Standard Dykstra", "Stall-Detection (FF)"],
        max_iter=MAX_ITER,
        filename=f"benchmark_convergence_SEED={SEED}_DIM={DIM}_HS={NUM_HALFSPACES}.png",
        show=True,
    )

if __name__ == "__main__":
    SEED = int(time.time() * 1000) % 1000000
    # SEED = 42
    DIM = 2
    NUM_HALFSPACES = 20
    MARGIN = 0.01
    MAX_ITER = 200
    MIN_ERROR = 1e-5

    run_benchmark()
