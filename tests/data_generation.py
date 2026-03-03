import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_generator import generate_crescent_data_2d


def plot_distributions(zeta: np.ndarray, z: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(zeta[:, 0], zeta[:, 1], alpha=0.5, color='blue', edgecolor='k', s=20)
    ax1.set_title(r"Standard normal $\mathcal{N}(0, I_2)$")
    ax1.set_xlabel("$z_1$")
    ax1.set_ylabel("$z_2$")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal')

    ax2.scatter(z[:, 0], z[:, 1], alpha=0.5, color='red', edgecolor='k', s=20)
    ax2.set_title("Crescent distribution")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axis('equal')

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "data_generation")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"synthetic_distribution_SEED={SEED}_M={M}.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    M = 1000
    SEED = 42

    normal_particles, crescent_particles = generate_crescent_data_2d(num_particles=M, seed=SEED)
    plot_distributions(normal_particles, crescent_particles)