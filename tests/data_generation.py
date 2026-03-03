import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import DistributionPlotter
from utils import generate_crescent_data_2d

if __name__ == "__main__":
    M = 1000
    SEED = 42

    normal_particles, crescent_particles = generate_crescent_data_2d(num_particles=M, seed=SEED)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "data_generation")
    plotter = DistributionPlotter(output_dir=out_dir)
    plotter.plot_distributions(
        zeta=normal_particles,
        z=crescent_particles,
        seed=SEED,
        m=M,
        show=True,
    )