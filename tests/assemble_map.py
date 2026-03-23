import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DataGenerator
from utils import DistributionPlotter
from utils import HermiteBasis
from utils import KRMap


def _run_single_reconstruction(
	normal_samples: np.ndarray,
	synthetic_samples: np.ndarray,
	kr_map: KRMap,
	weights: dict[int, np.ndarray],
	benchmark_label: str,
	plot_filename: str,
) -> dict[str, np.ndarray]:
	"""Evaluate and plot one map reconstruction benchmark."""
	t0 = time.perf_counter()
	mapped_samples = kr_map.evaluate(
		z=synthetic_samples,
		weights_by_component=weights,
	)
	elapsed = time.perf_counter() - t0

	plotter = DistributionPlotter(output_dir=PLOT_OUTPUT_DIR)
	plotter.plot_kr_map_distribution_single_solver(
		normal_samples=normal_samples[:, :2],
		synthetic_samples=synthetic_samples[:, :2],
		mapped_samples=mapped_samples[:, :2],
		solver_label=f"assembled KR map ({benchmark_label})",
		filename=plot_filename,
		show=SHOW_PLOT,
	)

	print(f"\n[{benchmark_label}] Map evaluation runtime: {elapsed:.4f}s")
	print(f"[{benchmark_label}] Component 1 weights: {weights[1]}")
	print(f"[{benchmark_label}] Component 2 weights: {weights[2]}")
	print(f"[{benchmark_label}] Saved plot: {os.path.join(PLOT_OUTPUT_DIR, plot_filename)}")

	return {
		"mapped_samples": mapped_samples,
	}


def run_assembly_map_test() -> dict[str, dict[str, np.ndarray]]:
	"""Run map reconstruction benchmarks using module-level configuration."""
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

	baseline_result = _run_single_reconstruction(
		normal_samples=normal_samples,
		synthetic_samples=synthetic_samples,
		kr_map=kr_map,
		weights=BASELINE_WEIGHTS,
		benchmark_label="baseline",
		plot_filename=BASELINE_PLOT_FILENAME,
	)

	truncated_result = _run_single_reconstruction(
		normal_samples=normal_samples,
		synthetic_samples=synthetic_samples,
		kr_map=kr_map,
		weights=TRUNCATED_WEIGHTS,
		benchmark_label="truncated",
		plot_filename=TRUNCATED_PLOT_FILENAME,
	)

	return {
		"baseline": {
			"normal_samples": normal_samples,
			"synthetic_samples": synthetic_samples,
			"mapped_samples": baseline_result["mapped_samples"],
		},
		"truncated": {
			"normal_samples": normal_samples,
			"synthetic_samples": synthetic_samples,
			"mapped_samples": truncated_result["mapped_samples"],
		},
	}


if __name__ == "__main__":
	def SHEAR_FUNCTION(zeta: np.ndarray) -> np.ndarray:
		"""Experiment shear function applied internally by DataGenerator."""
		return zeta[:, 0] ** 2

	SEED = 42
	NUM_PARTICLES = 1000
	NUM_DIMENSIONS = 2
	DEGREE = 2
	
	BASELINE_WEIGHTS = {
		1: np.array([0.0, 1.0, 0.0]),
		2: np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
	}

	# TRUNCATED_WEIGHTS = {
	# 	1: np.array([3.56913064e-06, 7.14601383e-01, -2.66425150e-06]),
	# 	2: np.array([
	# 		2.11429379e-06,
	# 		4.55947604e-01,
	# 		-6.72377179e-03,
	# 		-1.27633283e-06,
	# 		4.39760107e-06,
	# 		-3.42189765e-02,
	# 	]),
	# }
	TRUNCATED_WEIGHTS = {
		1: np.array([3.56913064e-06, 9.14601383e-01, -2.66425150e-06]),
		2: np.array([
			2.11429379e-06,
			9.55947604e-01,
			-6.72377179e-03,
			-1.27633283e-06,
			4.39760107e-06,
			-9.42189765e-01,
		]),
	}
	

	PLOT_OUTPUT_DIR = os.path.join(
		os.path.dirname(__file__), "..", "results", "map_reconstruction"
	)
	BASELINE_PLOT_FILENAME = (
		f"assemble_map_reconstruction_baseline_SEED={SEED}_M={NUM_PARTICLES}.png"
	)
	TRUNCATED_PLOT_FILENAME = (
		f"assemble_map_reconstruction_truncated_SEED={SEED}_M={NUM_PARTICLES}.png"
	)
	SHOW_PLOT = False

	run_assembly_map_test()
