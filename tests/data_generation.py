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

    # Rough-line parameters.
    SIGMA = 0.15

    # (Axial/directional) GVM parameters.
    VM_AMPLITUDE = 2.0
    VM_KAPPA = 8.0
    VM_RADIUS_MEAN = 2.0
    VM_RADIUS_STD = 0.30
    VM_MEAN_DIRECTION = np.array([1.0, 0.0], dtype=float)

    SHEAR_FUNCTION_MODEL = BoomerangShearFunction()  # boomerang shear
    # SHEAR_FUNCTION_MODEL = RoughLineShearFunction(sigma=SIGMA)  # rough line near y=x shear
    # SHEAR_FUNCTION_MODEL = GaussianVonMisesShearFunction(  # directional nD Gaussian-von-Mises shear
    #     amplitude=VM_AMPLITUDE,
    #     kappa=VM_KAPPA,
    #     radius_mean=VM_RADIUS_MEAN,
    #     radius_std=VM_RADIUS_STD,
    #     mean_direction=VM_MEAN_DIRECTION,
    # )
    # SHEAR_FUNCTION_MODEL = AxialGaussianVonMisesShearFunction(  # axial nD Gaussian-von-Mises shear
    #     amplitude=VM_AMPLITUDE,
    #     kappa=VM_KAPPA,
    #     radius_mean=VM_RADIUS_MEAN,
    #     radius_std=VM_RADIUS_STD,
    #     mean_direction=VM_MEAN_DIRECTION,
    # )

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
        show=True,
    )