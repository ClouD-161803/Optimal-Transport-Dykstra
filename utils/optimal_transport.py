"""Classes for 1D Knothe-Rosenblatt map estimation via Maximum Likelihood."""

import abc

import numpy as np

from .hermite import hermite_polynomial


class Basis(abc.ABC):
    """Abstract base class for polynomial basis families.

    Subclasses must implement methods to evaluate the basis functions and
    their first derivatives at a set of sample points.
    """

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Evaluate basis functions at the given points.

        Parameters
        ----------
        x : np.ndarray
            1D array of evaluation points, shape ``(M,)``.
        degree : int
            Maximum polynomial degree to include.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(M, degree + 1)`` where column *j* contains
            the *j*-th basis function evaluated at every point in *x*.
        """

    @abc.abstractmethod
    def evaluate_derivative(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Evaluate the first derivatives of the basis functions.

        Parameters
        ----------
        x : np.ndarray
            1D array of evaluation points, shape ``(M,)``.
        degree : int
            Maximum polynomial degree to include.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(M, degree + 1)`` where column *j* contains
            d/dx of the *j*-th basis function evaluated at every point in *x*.
        """


class HermiteBasis(Basis):
    """Probabilist's Hermite polynomial basis.

        d/dx He_j(x) = j * He_{j-1}(x)

    for first derivatives.
    """

    def evaluate(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Evaluate probabilist's Hermite polynomials He_0 .. He_degree.

        Parameters
        ----------
        x : np.ndarray
            1D array of evaluation points, shape ``(M,)``.
        degree : int
            Maximum polynomial degree.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(M, degree + 1)``.
        """
        return hermite_polynomial(x, degree)

    def evaluate_derivative(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Evaluate derivatives of probabilist's Hermite polynomials.

        Uses the identity ``d/dx He_j(x) = j * He_{j-1}(x)``.  The 0-th
        column is all zeros because He_0 is constant.

        Parameters
        ----------
        x : np.ndarray
            1D array of evaluation points, shape ``(M,)``.
        degree : int
            Maximum polynomial degree.

        Returns
        -------
        np.ndarray
            Matrix of shape ``(M, degree + 1)``.
        """
        M = len(x)
        dHe = np.zeros((M, degree + 1))

        if degree >= 1:
            # He_{j-1} values are needed for columns j = 1 .. degree.
            # hermite_polynomial(x, degree - 1) gives columns 0 .. degree-1.
            He_prev = hermite_polynomial(x, degree - 1)

            # j runs from 1 to degree (inclusive)
            j = np.arange(1, degree + 1)  # shape (degree,)
            dHe[:, 1:] = He_prev * j  # broadcast (M, degree) * (degree,)

        return dHe


class KRMap1D:
    """One-dimensional Knothe-Rosenblatt map estimated by Maximum Likelihood.

    Given *M* source particles and a polynomial basis of given degree, this
    class pre-computes the basis matrix and its derivative matrix, then
    exposes the negative log-likelihood objective, its gradient, and the
    polyhedral monotonicity constraints needed by Dykstra's algorithm.

    Parameters
    ----------
    data : np.ndarray
        1D array of source particles (z_1 coordinates), shape ``(M,)``.
    basis : Basis
        A ``Basis`` instance used to build the design matrices.
    degree : int
        Maximum polynomial degree for the map parameterisation.

    Attributes
    ----------
    M : int
        Number of source particles.
    Psi : np.ndarray
        Basis matrix of shape ``(M, degree + 1)``.
    dPsi : np.ndarray
        Derivative basis matrix of shape ``(M, degree + 1)``.
    """

    def __init__(self, data: np.ndarray, basis: Basis, degree: int) -> None:
        self.data = data
        self.M = len(data)
        self.degree = degree

        # Pre-compute and cache the design matrices.
        self.Psi: np.ndarray = basis.evaluate(data, degree)
        self.dPsi: np.ndarray = basis.evaluate_derivative(data, degree)

    def objective(self, w: np.ndarray) -> float:
        """Compute the negative log-likelihood objective.

        .. math::

            f(w) = \\frac{1}{M} \\sum_{i=1}^{M}
                   \\left[\\frac{1}{2}(\\Psi_i w)^2
                   - \\ln(\\nabla\\Psi_i w)\\right]

        Parameters
        ----------
        w : np.ndarray
            Coefficient vector, shape ``(degree + 1,)``.

        Returns
        -------
        float
            Scalar objective value.
        """
        Psi_w = self.Psi @ w        # (M,)
        dPsi_w = self.dPsi @ w       # (M,)
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
            Coefficient vector, shape ``(degree + 1,)``.

        Returns
        -------
        np.ndarray
            Gradient vector, shape ``(degree + 1,)``.
        """
        Psi_w = self.Psi @ w        # (M,)
        dPsi_w = self.dPsi @ w       # (M,)
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
            Constraint matrix, shape ``(M, degree + 1)``.
        b : np.ndarray
            Right-hand-side vector, shape ``(M,)``.
        """
        A = -self.dPsi
        b = -epsilon * np.ones(self.M)
        return A, b
