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
    sq: np.ndarray = result.squared_errors  # type: ignore
    st: np.ndarray = result.stalled_errors  # type: ignore
    cv: np.ndarray = result.converged_errors  # type: ignore

    color_seq = []
    for i in range(len(iters)):
        if not np.isnan(cv[i]):
            color_seq.append(('tab:green', 'Converged'))
        elif not np.isnan(st[i]):
            color_seq.append(('tab:red', 'Stalled'))
        else:
            color_seq.append(('tab:blue', 'Normal'))

    groups: list = []
    current_run = [0]
    for idx in range(1, len(iters)):
        if color_seq[idx][0] == color_seq[current_run[0]][0]:
            current_run.append(idx)
        else:
            groups.append((color_seq[current_run[0]], current_run))
            current_run = [idx]
    groups.append((color_seq[current_run[0]], current_run))

    seen_labels: set = set()
    for g, ((color, lbl), indices) in enumerate(groups):
        xs = indices + [groups[g + 1][1][0]] if g < len(groups) - 1 else indices
        lbl_arg = lbl if lbl not in seen_labels else None
        seen_labels.add(lbl)
        ax.semilogy(iters[xs], sq[xs], '.-', color=color, markersize=3, label=lbl_arg)  # type: ignore

    ax.set_title(label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Squared error")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(
    f"Convergence Comparison  (dim={DIM}, halfspaces={NUM_HALFSPACES})",
    fontsize=13,
)
out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "dykstra_benchmarks")
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"benchmark_convergence_SEED={SEED}_DIM={DIM}_HS={NUM_HALFSPACES}.png"), dpi=150)
plt.show()
