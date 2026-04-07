import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_generator import GVMDataGenerator
from utils.plotter import DistributionPlotter


def test_gvm_halfspace_constraint_x2_leq_5() -> None:
    gvm_alpha: float = -3.2
    gvm_beta: np.ndarray = np.array([0.0, 0.0], dtype=float)
    gvm_gamma: np.ndarray = np.array([[4.2, 0.0], [0.0, 0.0]], dtype=float)
    gvm_kappa: float = 12.0
    halfspace_A: np.ndarray = np.array([[0.0, 1.0]], dtype=float)
    halfspace_b: np.ndarray = np.array([5.0], dtype=float)

    generator = GVMDataGenerator(
        alpha=gvm_alpha,
        beta=gvm_beta,
        gamma=gvm_gamma,
        kappa=gvm_kappa,
        halfspace_A=halfspace_A,
        halfspace_b=halfspace_b,
    )

    num_particles = 2500
    num_dimensions = 2
    seed = 1234
    plot_size = 20.0
    distribution_xlim: tuple[float, float] | None = (-plot_size, plot_size)
    distribution_ylim: tuple[float, float] | None = (-plot_size, plot_size)
    shear_label = "GVM_x2_leq_5"
    zeta, z = generator.generate(
        num_particles=num_particles,
        num_dimensions=num_dimensions,
        seed=seed,
    )

    assert z.shape == (num_particles, num_dimensions)
    assert np.all((halfspace_A @ z.T).T <= halfspace_b + 1e-12)
    assert np.max(z[:, 1]) <= 5.0 + 1e-12

    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "data_generation",
    )
    filename = (
        f"synthetic_distribution_SEED={seed}_M={num_particles}_"
        f"SHEAR={shear_label}.png"
    )
    plotter = DistributionPlotter(output_dir=output_dir)
    plotter.plot_distributions(
        reference_samples=zeta[:, :2],
        sheared_samples=z[:, :2],
        seed=seed,
        m=num_particles,
        shear_label=shear_label,
        xlim=distribution_xlim,
        ylim=distribution_ylim,
        filename=filename,
        show=False,
    )
    assert os.path.exists(os.path.join(output_dir, filename))
