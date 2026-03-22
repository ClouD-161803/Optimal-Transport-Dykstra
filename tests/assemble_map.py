import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DataGenerator
from utils import DistributionPlotter
from utils import HermiteBasis
from utils import KRMap


def run_assembly_map_test() -> dict[str, np.ndarray]:
	"""Run the KR map assembly test using module-level configuration."""
	data_generator = DataGenerator(shear_function=SHEAR_FUNCTION)
	normal_samples, synthetic_samples = data_generator.generate(
		num_particles=NUM_PARTICLES,
		num_dimensions=NUM_DIMENSIONS,
		seed=SEED,
	)

	basis = HermiteBasis()
	kr_map = KRMap(
		degree=DEGREE,
		basis_1d=basis,
		log_epsilon=1e-8,
	)

	t0 = time.perf_counter()
	mapped_samples = kr_map.evaluate(
		z=synthetic_samples,
		weights_by_component=WEIGHTS,
	)
	elapsed = time.perf_counter() - t0

	plotter = DistributionPlotter(output_dir=PLOT_OUTPUT_DIR)
	plotter.plot_kr_map_distribution_single_solver(
		normal_samples=normal_samples[:, :2],
		synthetic_samples=synthetic_samples[:, :2],
		mapped_samples=mapped_samples[:, :2],
		solver_label="assembled KR map",
		filename=PLOT_FILENAME,
		show=SHOW_PLOT,
	)

	print(f"Map evaluation runtime: {elapsed:.4f}s")
	print(f"Component 1 weights: {WEIGHTS[1]}")
	print(f"Component 2 weights: {WEIGHTS[2]}")
	print(f"Saved plot: {os.path.join(PLOT_OUTPUT_DIR, PLOT_FILENAME)}")

	return {
		"normal_samples": normal_samples,
		"synthetic_samples": synthetic_samples,
		"mapped_samples": mapped_samples,
	}


if __name__ == "__main__":
	def SHEAR_FUNCTION(zeta: np.ndarray) -> np.ndarray:
		"""Experiment shear function applied internally by DataGenerator."""
		return zeta[:, 0] ** 2

	SEED = 42
	NUM_PARTICLES = 1000
	NUM_DIMENSIONS = 2
	DEGREE = 2
	
	WEIGHTS = {
		1: np.array([0.0, 1.0, 0.0]),
		2: np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
	}

	PLOT_OUTPUT_DIR = os.path.join(
		os.path.dirname(__file__), "..", "results", "map_reconstruction"
	)
	PLOT_FILENAME = (
		f"assemble_map_reconstruction_SEED={SEED}_M={NUM_PARTICLES}.png"
	)
	SHOW_PLOT = False

	run_assembly_map_test()
