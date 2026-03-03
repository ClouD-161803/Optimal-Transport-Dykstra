"""Benchmark: Standard Dykstra vs Stall-Detection (Fast-Forward) Dykstra."""

import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.projection_solver import DykstraProjectionSolver, DykstraStallDetectionSolver

SEED = 42
DIM = 2
NUM_HALFSPACES = 20
MARGIN = 0.01
MAX_ITER = 200
MIN_ERROR = 1e-5

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

print("=" * 60)
print(f"{'Solver':<30} {'Time (s)':>10} {'Final error':>15}")
print("-" * 60)
print(f"{'Standard Dykstra':<30} {time_std:>10.4f} {result_std.squared_errors[-1]:>15.6e}")  # type: ignore
print(f"{'Stall-Detection (FF)':<30} {time_ff:>10.4f} {result_ff.squared_errors[-1]:>15.6e}")  # type: ignore
print("=" * 60)
print()

iters = np.arange(MAX_ITER + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

for ax, label, result in [
    (axes[0], "Standard Dykstra", result_std),
    (axes[1], "Stall-Detection (FF)", result_ff),
]:
    sq = result.squared_errors  # type: ignore
    st = result.stalled_errors  # type: ignore
    cv = result.converged_errors  # type: ignore

    normal_mask = np.isnan(st) & np.isnan(cv)  # type: ignore
    ax.semilogy(iters[normal_mask], sq[normal_mask], '.-', color='tab:blue', # type: ignore
                markersize=3, label='Normal') # type: ignore

    stall_mask = ~np.isnan(st)  # type: ignore
    if np.any(stall_mask):
        ax.semilogy(iters[stall_mask], st[stall_mask], '.-', color='tab:red', # type: ignore
                    markersize=3, label='Stalled') # type: ignore

    conv_mask = ~np.isnan(cv)  # type: ignore
    if np.any(conv_mask):
        ax.semilogy(iters[conv_mask], cv[conv_mask], '.-', color='tab:green', # type: ignore
                    markersize=3, label='Converged') # type: ignore

    ax.set_title(label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Squared error")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(
    f"Convergence Comparison  (dim={DIM}, halfspaces={NUM_HALFSPACES})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "benchmark_convergence.png"),
            dpi=150)
plt.show()
