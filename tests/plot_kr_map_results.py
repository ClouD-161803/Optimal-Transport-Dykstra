"""Plot 3-panel KR map experiment results from saved NPZ/JSON data.

This test loads pre-computed KR map experiment results (reference samples,
target samples, and the final mapping) and creates a 3-panel visualization
showing the distribution transformation.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.plotter import DistributionPlotter
from utils.optimal_transport import Basis, HermiteBasis, KRMap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default run: RoughLine shear with 10000 PGD iterations
SEED: int = 2222
M: int = 500
PGDITERS: int = 10000
DYKSTRA_ITERS: str = "1_10"
MODE: str = "fast"
TIMESTAMP: str = "20260412-001825"

PLOT_SIZE: float = 5.0
X_LIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)
Y_LIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "full_experiment_benchmarks",
    "full_run_iterates",
)
NPZ_FILENAME = f"kr2d_full_run_iterates_SEED={SEED}_M={M}_PGDITERS={PGDITERS}_DYKSTRA_ITERS={DYKSTRA_ITERS}_MODE={MODE}_TS={TIMESTAMP}.npz"
JSON_FILENAME = f"kr2d_full_run_iterates_SEED={SEED}_M={M}_PGDITERS={PGDITERS}_DYKSTRA_ITERS={DYKSTRA_ITERS}_TS={TIMESTAMP[:-6]}.json"

NPZ_PATH = os.path.join(RESULTS_DIR, NPZ_FILENAME)
JSON_PATH = os.path.join(RESULTS_DIR, JSON_FILENAME)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "full_experiment_benchmarks", "distribution_plots")

# KR map settings (must match the experimental config)
DEGREE: int = 2
BASIS: Basis = HermiteBasis()
KR_MAP: KRMap = KRMap(
    degree=DEGREE,
    basis_1d=BASIS,
    log_epsilon=1e-8,
)


def load_kr_map_results(
    npz_path: str,
    kr_map: KRMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load reference, target, and mapped samples from NPZ file.

    Applies the learned KR map weights to the target samples to get mapped samples.

    Returns:
        Tuple of (reference_samples, target_samples, mapped_samples)
    """
    data = np.load(npz_path, allow_pickle=True)

    reference_samples = np.asarray(data["run_reference_samples_plot"], dtype=float)
    target_samples = np.asarray(data["run_target_samples_plot"], dtype=float)

    weights_by_component = {
        1: np.asarray(data["comp1_fast_weights"], dtype=float),
        2: np.asarray(data["comp2_fast_weights"], dtype=float),
    }

    mapped_samples = kr_map.evaluate(target_samples, weights_by_component)

    return reference_samples, target_samples, mapped_samples


def plot_kr_map_results(
    npz_path: str,
    json_path: str,
    output_dir: str,
    kr_map: KRMap,
    seed: int,
    m: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> str:
    """Plot 3-panel KR map results and save figure.

    Parameters
    ----------
    npz_path : str
        Path to NPZ file containing the run data
    json_path : str
        Path to JSON file (for metadata)
    output_dir : str
        Directory to save the output plot
    kr_map : KRMap
        The KR map object with configured degree and basis
    seed : int
        Random seed used in the experiment
    m : int
        Number of particles
    xlim, ylim : optional
        Axis limits for the plots

    Returns
    -------
    str
        Path to the saved figure
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    reference_samples, target_samples, mapped_samples = load_kr_map_results(npz_path, kr_map)

    plotter = DistributionPlotter(output_dir=output_dir)

    shear_label = "RoughLine"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)
                metadata = json_data.get("metadata", {})
                shear_label = metadata.get("shear_label", "RoughLine")
        except Exception:
            pass

    fig = plotter.plot_kr_map_distribution_single_solver(
        normal_samples=reference_samples,
        synthetic_samples=target_samples,
        mapped_samples=mapped_samples,
        solver_label="Fast Dykstra",
        panel_titles=(
            "Reference standard normal",
            f"Synthetic distribution ({shear_label})",
            "Mapped with KR map",
        ),
        xlim=xlim,
        ylim=ylim,
        filename=f"kr_map_results_SEED={seed}_M={m}_{shear_label}.png",
        show=True,
    )

    output_path = os.path.join(
        output_dir, f"kr_map_results_SEED={seed}_M={m}_{shear_label}.png"
    )
    return output_path


if __name__ == "__main__":
    print(f"Loading KR map results from: {NPZ_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    output_path = plot_kr_map_results(
        npz_path=NPZ_PATH,
        json_path=JSON_PATH,
        output_dir=OUTPUT_DIR,
        kr_map=KR_MAP,
        seed=SEED,
        m=M,
        xlim=X_LIM,
        ylim=Y_LIM,
    )

    print(f"Plot saved to: {output_path}")
