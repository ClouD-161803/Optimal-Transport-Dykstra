from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import scipy.stats


class ShearFunction(ABC):
    """Base interface for shear-function families used by DataGenerator."""

    @abstractmethod
    def shear(self, zeta: np.ndarray) -> np.ndarray:
        """Compute one scalar shear value per particle."""

    def __call__(self, zeta: np.ndarray) -> np.ndarray:
        return self.shear(zeta)


class BoomerangShearFunction(ShearFunction):
    """Quadratic linking shear with robust n-dimensional padding."""

    def __init__(
        self,
        alpha: float = 0.0,
        beta: np.ndarray | None = None,
        gamma: np.ndarray | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = (
            np.asarray(beta, dtype=float).reshape(-1)
            if beta is not None
            else np.zeros(2, dtype=float)
        )
        if gamma is None:
            self.gamma = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=float)
        else:
            gamma_array = np.asarray(gamma, dtype=float)
            if gamma_array.ndim != 2:
                raise ValueError("gamma must be a 2D array-like object.")
            self.gamma = gamma_array

    @staticmethod
    def _resolved_beta(beta: np.ndarray, num_dimensions: int) -> np.ndarray:
        """Normalise beta to ambient dimension and disable index-1 self-shear."""
        resolved = np.zeros(num_dimensions, dtype=float)
        upper = min(beta.size, num_dimensions)
        if upper > 0:
            resolved[:upper] = beta[:upper]
        if num_dimensions > 1:
            resolved[1] = 0.0
        return resolved

    @staticmethod
    def _resolved_gamma(gamma: np.ndarray, num_dimensions: int) -> np.ndarray:
        """Normalise gamma to ambient dimension and disable index-1 self-coupling."""
        resolved = np.zeros((num_dimensions, num_dimensions), dtype=float)
        rows = min(gamma.shape[0], num_dimensions)
        cols = min(gamma.shape[1], num_dimensions)
        if rows > 0 and cols > 0:
            resolved[:rows, :cols] = gamma[:rows, :cols]
        if num_dimensions > 1:
            resolved[1, :] = 0.0
            resolved[:, 1] = 0.0
        return resolved

    def resolved_parameters(self, num_dimensions: int) -> tuple[np.ndarray, np.ndarray]:
        """Return beta and gamma padded/truncated to the current dimension."""
        return (
            self._resolved_beta(self.beta, num_dimensions),
            self._resolved_gamma(self.gamma, num_dimensions),
        )

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        """Compute alpha + zeta beta + 0.5 * diag(zeta gamma zeta^T)."""
        num_dimensions = int(zeta.shape[1])
        beta, gamma = self.resolved_parameters(num_dimensions)
        linear_term = zeta @ beta
        quadratic_term = np.einsum("bi,ij,bj->b", zeta, gamma, zeta)
        return self.alpha + linear_term + 0.5 * quadratic_term


class RoughLineShearFunction(ShearFunction):
    """Approximately linear shear around y = x with tunable spread."""

    def __init__(self, sigma: float = 0.15) -> None:
        self.sigma = float(sigma)

    def shear(self, zeta: np.ndarray) -> np.ndarray:
        return zeta[:, 0] - (1.0 - self.sigma) * zeta[:, 1]


class DataGenerator:
    """Configurable synthetic data generator for KR-map experiments."""

    def __init__(
        self,
        shear_function: Callable[[np.ndarray], np.ndarray] | ShearFunction | None = None,
        halfspace_A: np.ndarray | None = None,
        halfspace_b: np.ndarray | None = None,
        max_rejection_rounds: int = 1_000,
    ) -> None:
        self.shear_function = shear_function or BoomerangShearFunction()
        self.max_rejection_rounds = int(max_rejection_rounds)
        self.halfspace_A, self.halfspace_b = self._normalise_halfspace_constraints(
            halfspace_A=halfspace_A,
            halfspace_b=halfspace_b,
        )

    @staticmethod
    def _normalise_halfspace_constraints(
        halfspace_A: np.ndarray | None,
        halfspace_b: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Validate and store polyhedral constraints of form A @ z <= b."""
        if (halfspace_A is None) != (halfspace_b is None):
            raise ValueError(
                "halfspace_A and halfspace_b must both be provided, or both be None."
            )
        if halfspace_A is None or halfspace_b is None:
            return None, None

        A = np.asarray(halfspace_A, dtype=float)
        b = np.asarray(halfspace_b, dtype=float).reshape(-1)
        if A.ndim != 2:
            raise ValueError("halfspace_A must be a 2D array-like object.")
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                "halfspace_A and halfspace_b size mismatch: rows(A) must equal len(b)."
            )
        return A, b

    def _resolved_halfspace_matrix(self, num_dimensions: int) -> np.ndarray | None:
        """Pad/truncate configured half-space normals to ambient dimension."""
        if self.halfspace_A is None:
            return None
        resolved = np.zeros((self.halfspace_A.shape[0], num_dimensions), dtype=float)
        cols = min(self.halfspace_A.shape[1], num_dimensions)
        if cols > 0:
            resolved[:, :cols] = self.halfspace_A[:, :cols]
        return resolved

    def _inside_halfspace_constraints(
        self,
        z: np.ndarray,
        num_dimensions: int,
    ) -> np.ndarray:
        """Return mask of particles satisfying all configured half-space constraints."""
        if self.halfspace_A is None or self.halfspace_b is None:
            return np.ones(z.shape[0], dtype=bool)
        A_resolved = self._resolved_halfspace_matrix(num_dimensions)
        if A_resolved is None or A_resolved.shape[0] == 0:
            return np.ones(z.shape[0], dtype=bool)
        return np.all((A_resolved @ z.T).T <= self.halfspace_b + 1e-12, axis=1)

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

        if self.halfspace_A is None:
            zeta = rng.standard_normal((num_particles, num_dimensions))
            z = self._apply_shear(zeta)
            return zeta, z

        zeta_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        accepted = 0
        rounds = 0

        while accepted < num_particles:
            rounds += 1
            if rounds > self.max_rejection_rounds:
                raise RuntimeError(
                    "Unable to generate enough particles satisfying half-space "
                    "constraints. Relax constraints or increase max_rejection_rounds."
                )

            remaining = num_particles - accepted
            batch_size = max(remaining, 2 * remaining)
            zeta_candidate = rng.standard_normal((batch_size, num_dimensions))
            z_candidate = self._apply_shear(zeta_candidate)
            keep_mask = self._inside_halfspace_constraints(
                z=z_candidate,
                num_dimensions=num_dimensions,
            )

            if not np.any(keep_mask):
                continue

            zeta_kept = zeta_candidate[keep_mask]
            z_kept = z_candidate[keep_mask]
            take = min(remaining, zeta_kept.shape[0])
            zeta_chunks.append(zeta_kept[:take])
            z_chunks.append(z_kept[:take])
            accepted += take

        return np.vstack(zeta_chunks), np.vstack(z_chunks)

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


class GVMDataGenerator(DataGenerator):
    """Generate true GVM samples by Gaussian-to-von-Mises transport."""

    def __init__(
        self,
        alpha: float,
        beta: np.ndarray,
        gamma: np.ndarray,
        kappa: float,
        halfspace_A: np.ndarray | None = None,
        halfspace_b: np.ndarray | None = None,
        max_rejection_rounds: int = 1_000,
    ) -> None:
        self.linking_function = BoomerangShearFunction(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.kappa = float(kappa)
        super().__init__(
            shear_function=self.linking_function,
            halfspace_A=halfspace_A,
            halfspace_b=halfspace_b,
            max_rejection_rounds=max_rejection_rounds,
        )

    def _apply_shear(self, zeta: np.ndarray) -> np.ndarray:
        """Apply the exact PIT map from Gaussian phase to von Mises phase."""
        z = zeta.copy()
        theta_loc = self.linking_function.shear(zeta)
        u = scipy.stats.norm.cdf(zeta[:, 1])
        z[:, 1] = scipy.stats.vonmises.ppf(u, self.kappa, loc=theta_loc)
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
