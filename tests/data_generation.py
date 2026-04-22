import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_generator import (
    BoomerangShearFunction,
    CubicShearFunction,
    DataGenerator,
    GVMDataGenerator,
    RoughLineShearFunction,
    XShapedShearFunction,
)
from utils.plotter import DistributionPlotter

if __name__ == "__main__":
    M = 1000
    SEED = 0
    NUM_DIMENSIONS = 2
    PLOT_SIZE = 8.0
    DISTRIBUTION_XLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)
    DISTRIBUTION_YLIM: tuple[float, float] | None = (-PLOT_SIZE, PLOT_SIZE)

    # GVM tuned for a milder U-shape so contour behaviour is easier to read.
    GVM_ALPHA = -3.2
    GVM_BETA = np.array([0.0, 0.0], dtype=float)
    GVM_GAMMA = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=float)
    GVM_KAPPA = 3.0
    LINE_SIGMA = 0.15

    # generator = GVMDataGenerator(
    #     alpha=GVM_ALPHA,
    #     beta=GVM_BETA,
    #     gamma=GVM_GAMMA,
    #     kappa=GVM_KAPPA,
    # )
    # generator = DataGenerator(
    #     shear_function=BoomerangShearFunction(),  # classic boomerang: x_1^2 added to x_2
    # )
    # generator = DataGenerator(
    #     shear_function=RoughLineShearFunction(sigma=LINE_SIGMA),  # line-like shear near y = x
    # )
    # generator = DataGenerator(
    #     shear_function=CubicShearFunction(strength=0.7),  # cubic S-curve wave pattern
    # )
    generator = DataGenerator(
        shear_function=XShapedShearFunction(strength=1.2),  # X-shaped: two crossing diagonals
    )
    reference_particles, sheared_particles = generator.generate(
        num_particles=M,
        num_dimensions=NUM_DIMENSIONS,
        seed=SEED,
    )
    shear_label = type(generator.shear_function).__name__.replace("ShearFunction", "")

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
