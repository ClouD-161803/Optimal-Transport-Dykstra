import numpy as np
from typing import Callable


class DataGenerator:
    """Configurable synthetic data generator for KR-map experiments."""

    def __init__(
        self,
        shear_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.shear_function = shear_function or self._default_shear_function

    def generate(
        self,
        num_particles: int,
        num_dimensions: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate n-dimensional samples and apply configured shear."""
        if num_dimensions < 2:
            raise ValueError("num_dimensions must be >= 2 for crescent data generation.")

        rng = np.random.default_rng(seed)
        zeta = rng.standard_normal((num_particles, num_dimensions))
        z = self._apply_shear(zeta)
        return zeta, z

    def _apply_shear(self, zeta: np.ndarray) -> np.ndarray:
        """Apply the configured shear to the second coordinate."""
        z = zeta.copy()
        shear_values = np.asarray(self.shear_function(zeta), dtype=float).reshape(-1)
        if shear_values.shape[0] != zeta.shape[0]:
            raise ValueError(
                "shear_function must return one scalar shear value per particle."
            )
        z[:, 1] = zeta[:, 1] + shear_values
        return z

    @staticmethod
    def _default_shear_function(zeta: np.ndarray) -> np.ndarray:
        """Default crescent shear: x₁² added to x₂."""
        return zeta[:, 0] ** 2

def generate_crescent_data_nd(
    num_particles: int,
    num_dimensions: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an n-dimensional crescent-like dataset.

    The base samples are standard normal in ``num_dimensions`` dimensions.
    A nonlinear shear is then applied to the second coordinate using the first
    coordinate:

    ``z[:, 1] = zeta[:, 1] + zeta[:, 0]**2``.

    All remaining coordinates are left unchanged.

    Args:
        num_particles (int): The number of particles (M) to generate.
        num_dimensions (int): Ambient data dimension.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - zeta (M x num_dimensions): Standard normal particles.
            - z (M x num_dimensions): Crescent-transformed particles.
    """
    return DataGenerator().generate(
        num_particles=num_particles,
        num_dimensions=num_dimensions,
        seed=seed,
    )


def generate_crescent_data_2d(
    num_particles: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible 2D crescent data generator wrapper."""
    return generate_crescent_data_nd(
        num_particles=num_particles,
        num_dimensions=2,
        seed=seed,
    )