import abc
from itertools import product

import numpy as np

from .hermite import hermite_polynomial


class Basis(abc.ABC):
    """Abstract base class for polynomial basis families.

    Subclasses must implement methods to evaluate the basis functions and
    their first derivatives at a set of sample points.
    """

    @abc.abstractmethod
    def evaluate(self, z: np.ndarray, max_degree: int) -> np.ndarray:
        """Evaluate basis functions at the given points.

        Parameters
        ----------
        z : np.ndarray
            Array of evaluation points, shape ``(M, k)``.
        max_degree : int
            Maximum polynomial degree to include per dimension.

        Returns
        -------
        np.ndarray
            Basis matrix of shape ``(M, n_terms)``.
        """

    @abc.abstractmethod
    def evaluate_derivative(self, z: np.ndarray, max_degree: int) -> np.ndarray:
        """Evaluate the first derivatives of the basis functions.

        Parameters
        ----------
        z : np.ndarray
            Array of evaluation points, shape ``(M, k)``.
        max_degree : int
            Maximum polynomial degree to include per dimension.

        Returns
        -------
        np.ndarray
            Derivative basis matrix of shape ``(M, n_terms)``.
        """


class HermiteBasis(Basis):
    """Probabilist's Hermite polynomial basis.

        d/dx He_j(x) = j * He_{j-1}(x)

    for first derivatives.
    """

    def evaluate(self, z: np.ndarray, max_degree: int) -> np.ndarray:
        """Evaluate probabilist's Hermite polynomials He_0 .. He_degree.

        Parameters
        ----------
        z : np.ndarray
            1D array of evaluation points, shape ``(M,)``.
        max_degree : int
            Maximum polynomial degree.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(M, max_degree + 1)``.
        """
        z = np.asarray(z)
        if z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]
        return hermite_polynomial(z, max_degree)

    def evaluate_derivative(self, z: np.ndarray, max_degree: int) -> np.ndarray:
        """Evaluate derivatives of probabilist's Hermite polynomials.

        Uses the identity ``d/dx He_j(x) = j * He_{j-1}(x)``.  The 0-th
        column is all zeros because He_0 is constant.

        Parameters
        ----------
        z : np.ndarray
            1D array of evaluation points, shape ``(M,)``.
        max_degree : int
            Maximum polynomial degree.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(M, max_degree + 1)``.
        """
        z = np.asarray(z)
        if z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]

        M = len(z)
        dHe = np.zeros((M, max_degree + 1))

        if max_degree >= 1:
            # He_{j-1} values are needed for columns j = 1 .. degree.
            # hermite_polynomial(x, degree - 1) gives columns 0 .. degree-1.
            He_prev = hermite_polynomial(z, max_degree - 1)

            # j runs from 1 to degree (inclusive)
            j = np.arange(1, max_degree + 1)  # shape (max_degree,)
            dHe[:, 1:] = He_prev * j  # broadcast (M, degree) * (degree,)

        return dHe


class TensorHermiteBasis(Basis):
    """Tensor-product probabilist's Hermite basis in arbitrary dimension.

    For each particle ``z_i = (z_{i,1}, ..., z_{i,k})`` and multi-index
    ``j = (j_1, ..., j_k)``, the basis term is

    ``He_{j_1}(z_{i,1}) * ... * He_{j_k}(z_{i,k})``.

    The derivative matrix is the partial derivative with respect to the last
    coordinate only:

    ``∂/∂z_k [Π_{r=1}^k He_{j_r}(z_r)]``.
    """

    @staticmethod
    def _validate_input(z: np.ndarray) -> np.ndarray:
        """Validate and return a 2D input array.

        Parameters
        ----------
        z : np.ndarray
            Input particle array with shape ``(M, k)``.

        Returns
        -------
        np.ndarray
            Validated particle array with shape ``(M, k)``.
        """
        z = np.asarray(z)
        if z.ndim != 2:
            raise ValueError("Input z must have shape (M, k).")
        return z

    @staticmethod
    def _multi_indices(k: int, max_degree: int) -> np.ndarray:
        """Build all tensor-product degree combinations.

        Parameters
        ----------
        k : int
            Number of dimensions.
        max_degree : int
            Maximum degree in each dimension.

        Returns
        -------
        np.ndarray
            Integer matrix of shape ``(n_terms, k)`` containing all
            combinations of degrees from 0 to ``max_degree``.
        """
        return np.asarray(
            list(product(range(max_degree + 1), repeat=k)),
            dtype=int,
        )

    def evaluate(self, z: np.ndarray, max_degree: int) -> np.ndarray:
        """Evaluate the tensor Hermite basis.

        Parameters
        ----------
        z : np.ndarray
            Particle matrix with shape ``(M, k)``.
        max_degree : int
            Maximum degree in each dimension.

        Returns
        -------
        np.ndarray
            Basis matrix with shape ``(M, (max_degree + 1)^k)``.
        """
        z = self._validate_input(z)
        M, k = z.shape

        multi_idx = self._multi_indices(k, max_degree)  # (n_terms, k)
        hermite_vals = np.stack(
            [hermite_polynomial(z[:, dim], max_degree) for dim in range(k)],
            axis=1,
        )  # (M, k, max_degree + 1)

        gather_idx = np.broadcast_to(multi_idx.T[None, :, :], (M, k, multi_idx.shape[0]))
        selected = np.take_along_axis(hermite_vals, gather_idx, axis=2)  # (M, k, n_terms)
        return np.prod(selected, axis=1)  # (M, n_terms)

    def evaluate_derivative(self, z: np.ndarray, max_degree: int) -> np.ndarray:
        """Evaluate the tensor basis derivative in the last variable only.

        Parameters
        ----------
        z : np.ndarray
            Particle matrix with shape ``(M, k)``.
        max_degree : int
            Maximum degree in each dimension.

        Returns
        -------
        np.ndarray
            Derivative basis matrix with shape ``(M, (max_degree + 1)^k)``.
        """
        z = self._validate_input(z)
        M, k = z.shape

        multi_idx = self._multi_indices(k, max_degree)  # (n_terms, k)

        last_vals = hermite_polynomial(z[:, -1], max_degree)  # (M, max_degree + 1)
        d_last_vals = np.zeros_like(last_vals)
        if max_degree >= 1:
            j = np.arange(1, max_degree + 1)
            d_last_vals[:, 1:] = last_vals[:, :-1] * j

        d_last_selected = d_last_vals[:, multi_idx[:, -1]]  # (M, n_terms)

        if k == 1:
            return d_last_selected

        hermite_other = np.stack(
            [hermite_polynomial(z[:, dim], max_degree) for dim in range(k - 1)],
            axis=1,
        )  # (M, k - 1, max_degree + 1)

        gather_idx = np.broadcast_to(
            multi_idx[:, :-1].T[None, :, :],
            (M, k - 1, multi_idx.shape[0]),
        )
        selected_other = np.take_along_axis(hermite_other, gather_idx, axis=2)  # (M, k - 1, n_terms)
        return np.prod(selected_other, axis=1) * d_last_selected


class KRMapComponent:
    """Dimension-agnostic Knothe-Rosenblatt map component.

    Given *M* source particles and a polynomial basis of given degree, this
    class pre-computes the basis matrix and its derivative matrix, then
    exposes the negative log-likelihood objective, its gradient, and the
    polyhedral monotonicity constraints needed by Dykstra's algorithm.

    Parameters
    ----------
    data : np.ndarray
        Source particles, shape ``(M, k)``.
    basis : Basis
        A ``Basis`` instance used to build the design matrices.
    degree : int
        Maximum polynomial degree for the map parameterisation.

    Attributes
    ----------
    M : int
        Number of source particles.
    Psi : np.ndarray
        Basis matrix of shape ``(M, n_terms)``.
    dPsi : np.ndarray
        Derivative basis matrix of shape ``(M, n_terms)``.
    num_coefficients : int
        Number of coefficients in the component parameterisation.
    """

    def __init__(self, data: np.ndarray, basis: Basis, degree: int) -> None:
        self.data = np.asarray(data)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        elif self.data.ndim != 2:
            raise ValueError("data must have shape (M, k).")

        self.M = self.data.shape[0]
        self.degree = degree

        # Pre-compute and cache the design matrices.
        self.Psi: np.ndarray = basis.evaluate(self.data, degree)
        self.dPsi: np.ndarray = basis.evaluate_derivative(self.data, degree)
        self.num_coefficients: int = self.Psi.shape[1]

    def objective(self, w: np.ndarray) -> float:
        """Compute the negative log-likelihood objective.

        .. math::

                 f(w) = \\frac{1}{M} \\sum_{i=1}^{M}
                   \\left[\\frac{1}{2}(\\Psi_i w)^2
                   - \\ln(\\nabla\\Psi_i w)\\right]

        Parameters
        ----------
        w : np.ndarray
            Coefficient vector, shape ``(n_terms,)``.

        Returns
        -------
        float
            Scalar objective value.
        """
        Psi_w = self.Psi @ w        # (M,)
        dPsi_w = self.dPsi @ w      # (M,)
        return (0.5 * np.dot(Psi_w, Psi_w) - np.sum(np.log(dPsi_w))) / self.M

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Compute the gradient of the negative log-likelihood.

        .. math::

            \\nabla f(w) = \\frac{1}{M}
                \\left[\\Psi^T (\\Psi w)
                - (\\nabla\\Psi)^T \\frac{1}{\\nabla\\Psi\\, w}\\right]

        Parameters
        ----------
        w : np.ndarray
            Coefficient vector, shape ``(n_terms,)``.

        Returns
        -------
        np.ndarray
            Gradient vector, shape ``(degree + 1,)``.
        """
        Psi_w = self.Psi @ w        # (M,)
        dPsi_w = self.dPsi @ w      # (M,)
        return (self.Psi.T @ Psi_w - self.dPsi.T @ (1.0 / dPsi_w)) / self.M

    def get_polyhedral_constraints(
        self, epsilon: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the polyhedral monotonicity constraints for Dykstra's algorithm.

        The monotonicity requirement is ``dPsi @ w >= epsilon``.  Because the
        Dykstra solver expects the form ``A @ w <= b``, this method returns the
        negated version:

            A = -dPsi,   b = -epsilon * ones(M)

        Parameters
        ----------
        epsilon : float, optional
            Strict-monotonicity margin (default ``1e-4``).

        Returns
        -------
        A : np.ndarray
            Constraint matrix, shape ``(M, n_terms)``.
        b : np.ndarray
            Right-hand-side vector, shape ``(M,)``.
        """
        A = -self.dPsi
        b = -epsilon * np.ones(self.M)
        return A, b


KRMap1D = KRMapComponent # backwards compatibility alias
