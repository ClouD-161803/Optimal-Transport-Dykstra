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


class GVMShearFunction(ShearFunction):
    """Quadratic GVM shear, parameterised for robust n-dimensional use."""

    def __init__(
        self,
        alpha: float,
        beta: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = np.asarray(beta, dtype=float).reshape(-1)
        gamma_array = np.asarray(gamma, dtype=float)
        if gamma_array.ndim != 2:
            raise ValueError("gamma must be a 2D array-like object.")
        self.gamma = gamma_array

    @staticmethod
    def _resolved_beta(beta: np.ndarray, num_dimensions: int) -> np.ndarray:
        """Normalise beta shape against ambient dimension and disable self-shear on index 1."""
        resolved = np.zeros(num_dimensions, dtype=float)
        upper = min(beta.size, num_dimensions)
        if upper > 0:
            resolved[:upper] = beta[:upper]
        if num_dimensions > 1:
            resolved[1] = 0.0
        return resolved

    @staticmethod
    def _resolved_gamma(gamma: np.ndarray, num_dimensions: int) -> np.ndarray:
        """Normalise gamma shape against ambient dimension and disable index-1 self-coupling."""
        resolved = np.zeros((num_dimensions, num_dimensions), dtype=float)
        rows = min(gamma.shape[0], num_dimensions)
        cols = min(gamma.shape[1], num_dimensions)
        if rows > 0 and cols > 0:
            resolved[:rows, :cols] = gamma[:rows, :cols]
        if num_dimensions > 1:
            resolved[1, :] = 0.0
            resolved[:, 1] = 0.0
        return resolved

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        """Compute per-particle shear from alpha + beta^T z + 0.5 * z^T gamma z."""
        num_dimensions = int(zeta.shape[1])
        beta = self._resolved_beta(self.beta, num_dimensions)
        gamma = self._resolved_gamma(self.gamma, num_dimensions)

        linear_term = zeta @ beta
        quadratic_term = np.einsum("bi,ij,bj->b", zeta, gamma, zeta)
        return self.alpha + linear_term + 0.5 * quadratic_term


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
