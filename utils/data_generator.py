from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class ShearFunction(ABC):
    """Base interface for shear-function families used by DataGenerator."""

    @abstractmethod
    def shear(self, zeta: np.ndarray) -> np.ndarray:
        """Compute one scalar shear value per particle."""

    def __call__(self, zeta: np.ndarray) -> np.ndarray:
        return self.shear(zeta)


class BoomerangShearFunction(ShearFunction):
    """Classic crescent/boomerang shear: x1^2 added to x2."""

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        return zeta[:, 0] ** 2


class RoughLineShearFunction(ShearFunction):
    """Approximately linear shear around y = x with tunable spread."""

    def __init__(self, sigma: float = 0.15) -> None:
        self.sigma = float(sigma)

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        return zeta[:, 0] - (1.0 - self.sigma) * zeta[:, 1]


class GaussianVonMisesShearFunction(ShearFunction):
    """Directional nD Gaussian-von-Mises shear."""

    def __init__(
        self,
        amplitude: float = 0.35,
        kappa: float = 6.0,
        radius_mean: float = 1.2,
        radius_std: float = 0.45,
        mean_direction: np.ndarray | None = None,
    ) -> None:
        if radius_std <= 0.0:
            raise ValueError("radius_std must be positive.")

        self.amplitude = float(amplitude)
        self.kappa = float(kappa)
        self.radius_mean = float(radius_mean)
        self.radius_std = float(radius_std)
        self.mean_direction = None if mean_direction is None else np.asarray(mean_direction, dtype=float).reshape(-1)

    def _resolved_unit_direction(self, num_dimensions: int) -> np.ndarray:
        """Build and normalise a direction vector in the active ambient dimension."""
        if self.mean_direction is None:
            direction = np.pad(
                np.array([1.0, 1.0], dtype=float),
                (0, max(0, num_dimensions - 2)),
                mode="constant",
            )[:num_dimensions]
        elif self.mean_direction.size < num_dimensions:
            direction = np.pad(
                self.mean_direction,
                (0, num_dimensions - self.mean_direction.size),
                mode="constant",
            )
        else:
            direction = self.mean_direction[:num_dimensions].copy()

        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 0.0:
            raise ValueError("mean_direction must have non-zero norm.")
        return direction / direction_norm

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        num_dimensions = int(zeta.shape[1])
        direction = self._resolved_unit_direction(num_dimensions)

        radius = np.linalg.norm(zeta, axis=1)
        radius_safe = np.maximum(radius, 1e-12)
        unit_vectors = zeta / radius_safe[:, None]
        cos_theta = np.clip(unit_vectors @ direction, -1.0, 1.0)

        gaussian_term = np.exp(-0.5 * ((radius - self.radius_mean) / self.radius_std) ** 2)
        von_mises_term = np.exp(self.kappa * (cos_theta - 1.0))
        return self.amplitude * gaussian_term * von_mises_term


class AxialGaussianVonMisesShearFunction(GaussianVonMisesShearFunction):
    """Axial nD Gaussian-von-Mises shear with symmetric angular peaks at +/-direction."""

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        num_dimensions = int(zeta.shape[1])
        direction = self._resolved_unit_direction(num_dimensions)

        radius = np.linalg.norm(zeta, axis=1)
        radius_safe = np.maximum(radius, 1e-12)
        unit_vectors = zeta / radius_safe[:, None]
        cos_theta = np.clip(unit_vectors @ direction, -1.0, 1.0)

        gaussian_term = np.exp(-0.5 * ((radius - self.radius_mean) / self.radius_std) ** 2)
        axial_von_mises_term = np.exp(self.kappa * ((cos_theta ** 2) - 1.0))
        return self.amplitude * gaussian_term * axial_von_mises_term


class DataGenerator:
    """Configurable synthetic data generator for KR-map experiments."""

    def __init__(
        self,
        shear_function: Callable[[np.ndarray], np.ndarray] | ShearFunction | None = None,
    ) -> None:
        self.shear_function = shear_function or BoomerangShearFunction()

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