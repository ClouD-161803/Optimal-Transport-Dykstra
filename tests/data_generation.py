import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DistributionPlotter
from utils import (
    BoomerangShearFunction,
    DataGenerator,
    GVMShearFunction,
    RoughLineShearFunction,
)

if __name__ == "__main__":
    M = 2500
    SEED = 42
    NUM_DIMENSIONS = 2
    PLOT_SIZE = 20.0
    DISTRIBUTION_XLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)
    DISTRIBUTION_YLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)

    # Line
    SIGMA = 0.15

    # GVM tuned to resemble the paper's last-row thin, stretched U-shape (panels e/f)
    GVM_ALPHA = -3.2
    GVM_BETA = np.array([0.0, 0.0], dtype=float)
    GVM_GAMMA = np.array([[4.2, 0.0], [0.0, 0.0]], dtype=float)

    # SHEAR_FUNCTION_MODEL = BoomerangShearFunction()  # boomerang shear

    # SHEAR_FUNCTION_MODEL = RoughLineShearFunction(sigma=SIGMA)  # rough line near y=x shear

    SHEAR_FUNCTION_MODEL = GVMShearFunction(  # quadratic GVM shear
        alpha=GVM_ALPHA,
        beta=GVM_BETA,
        gamma=GVM_GAMMA,
    )

    generator = DataGenerator(shear_function=SHEAR_FUNCTION_MODEL)
    reference_particles, sheared_particles = generator.generate(
        num_particles=M,
        num_dimensions=NUM_DIMENSIONS,
        seed=SEED,
    )
    shear_label = type(SHEAR_FUNCTION_MODEL).__name__.replace("ShearFunction", "")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "data_generation")
    plotter = DistributionPlotter(output_dir=out_dir)
    plotter.plot_distributions(
        reference_samples=reference_particles,
        sheared_samples=sheared_particles,
        seed=SEED,
        m=M,
        shear_label=shear_label,
        xlim=DISTRIBUTION_XLIM,
        ylim=DISTRIBUTION_YLIM,
        show=True,
    )
