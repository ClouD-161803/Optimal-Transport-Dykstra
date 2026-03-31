import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DistributionPlotter
from utils import (
    AxialGaussianVonMisesShearFunction,
    BoomerangShearFunction,
    DataGenerator,
    GaussianVonMisesShearFunction,
    RoughLineShearFunction,
)

if __name__ == "__main__":
    M = 1000
    SEED = 42
    NUM_DIMENSIONS = 2
    PLOT_SIZE = 5.0
    DISTRIBUTION_XLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)
    DISTRIBUTION_YLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)

    # Line
    SIGMA = 0.15

    # GVM
    VM_AMPLITUDE = 4.0
    VM_KAPPA = 0.1
    VM_RADIUS_MEAN = 1.3
    VM_RADIUS_STD = 0.5
    VM_MEAN_DIRECTION = np.array([1.0, 0.0], dtype=float)

    # SHEAR_FUNCTION_MODEL = BoomerangShearFunction()  # boomerang shear

    # SHEAR_FUNCTION_MODEL = RoughLineShearFunction(sigma=SIGMA)  # rough line near y=x shear

    # SHEAR_FUNCTION_MODEL = GaussianVonMisesShearFunction(  # directional nD Gaussian-von-Mises shear
    #     amplitude=VM_AMPLITUDE,
    #     kappa=VM_KAPPA,
    #     radius_mean=VM_RADIUS_MEAN,
    #     radius_std=VM_RADIUS_STD,
    #     mean_direction=VM_MEAN_DIRECTION,
    # )

    SHEAR_FUNCTION_MODEL = AxialGaussianVonMisesShearFunction(  # axial nD Gaussian-von-Mises shear
        amplitude=VM_AMPLITUDE,
        kappa=VM_KAPPA,
        radius_mean=VM_RADIUS_MEAN,
        radius_std=VM_RADIUS_STD,
        mean_direction=VM_MEAN_DIRECTION,
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