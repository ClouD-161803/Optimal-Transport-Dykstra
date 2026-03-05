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

    def __init__(
        self,
        data: np.ndarray,
        basis: Basis,
        degree: int,
        log_epsilon: float = 1e-8,
    ) -> None:
        self.data = np.asarray(data)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        elif self.data.ndim != 2:
            raise ValueError("data must have shape (M, k).")

        self.M = self.data.shape[0]
        self.degree = degree
        self.log_epsilon = log_epsilon

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
        safe_dPsi_w = np.maximum(dPsi_w + self.log_epsilon, self.log_epsilon)
        return (0.5 * np.dot(Psi_w, Psi_w) - np.sum(np.log(safe_dPsi_w))) / self.M

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
        safe_dPsi_w = np.maximum(dPsi_w + self.log_epsilon, self.log_epsilon)
        return (self.Psi.T @ Psi_w - self.dPsi.T @ (1.0 / safe_dPsi_w)) / self.M

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


def assemble_component_weights(
    component_results: list[dict],
    weight_key: str,
) -> dict[int, np.ndarray]:
    """Assemble KR component weights from benchmark outputs.

    Parameters
    ----------
    component_results : list of dict
        Per-component benchmark outputs containing ``component_dim`` and
        the requested weight vector key.
    weight_key : str
        Dictionary key for the weight vector, for example ``"w_vanilla"``
        or ``"w_fast"``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping ``component_dimension -> coefficient_vector``.
    """
    weights_by_component: dict[int, np.ndarray] = {}

    for result in component_results:
        if "component_dim" not in result:
            raise ValueError("Each component result must include 'component_dim'.")
        if weight_key not in result:
            raise ValueError(f"Missing weight key '{weight_key}' in component result.")

        component_dim = int(result["component_dim"])
        weights_by_component[component_dim] = np.asarray(result[weight_key]).reshape(-1)

    return weights_by_component


def get_tensor_identity_term_index(component_dim: int, degree: int) -> int:
    """Return the coefficient index for ``He_0...He_0 He_1``.

    For a ``component_dim``-dimensional KR component with tensor Hermite basis,
    this identifies the term corresponding to the identity initial map in the
    last variable:

    ``He_0(z_1) * ... * He_0(z_{k-1}) * He_1(z_k)``.
    """
    if component_dim < 1:
        raise ValueError("component_dim must be >= 1.")
    if degree < 1:
        raise ValueError("degree must be >= 1 to represent the identity term.")

    multi_idx = np.asarray(
        list(product(range(degree + 1), repeat=component_dim)),
        dtype=int,
    )
    target = np.zeros(component_dim, dtype=int)
    target[-1] = 1

    matches = np.where(np.all(multi_idx == target, axis=1))[0]
    if matches.size != 1:
        raise RuntimeError("Unable to uniquely identify tensor identity term index.")

    return int(matches[0])


def build_identity_initial_guess(component_dim: int, degree: int) -> np.ndarray:
    """Build an identity-map initial guess for a KR component.

    Returns a zero vector with exactly one coefficient set to ``1.0`` at the
    tensor term index representing ``He_0...He_0 He_1``.
    """
    num_coefficients = (degree + 1) ** component_dim
    w_init = np.zeros(num_coefficients, dtype=float)
    identity_idx = get_tensor_identity_term_index(component_dim, degree)
    w_init[identity_idx] = 1.0
    return w_init


def evaluate_kr_map(
    z: np.ndarray,
    degree: int,
    weights_by_component: dict[int, np.ndarray],
    basis_1d: Basis | None = None,
    tensor_basis: Basis | None = None,
) -> np.ndarray:
    """Evaluate an assembled n-dimensional KR map on input particles.

    Parameters
    ----------
    z : np.ndarray
        Input particle matrix with shape ``(M, d)``.
    degree : int
        Maximum polynomial degree for all components.
    weights_by_component : dict[int, np.ndarray]
        Mapping from component dimension ``k`` to coefficient vector for the
        ``k``-th KR component.
    basis_1d : Basis, optional
        Basis to use for the first component. Defaults to ``HermiteBasis``.
    tensor_basis : Basis, optional
        Basis to use for components ``k >= 2``. Defaults to
        ``TensorHermiteBasis``.

    Returns
    -------
    np.ndarray
        Mapped particles with shape ``(M, d)``.
    """
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must have shape (M, d).")

    M, d = z.shape
    mapped = np.zeros((M, d), dtype=float)

    basis_1d = basis_1d if basis_1d is not None else HermiteBasis()
    tensor_basis = tensor_basis if tensor_basis is not None else TensorHermiteBasis()

    for component_dim in range(1, d + 1):
        if component_dim not in weights_by_component:
            raise ValueError(
                f"Missing weights for component dimension {component_dim}."
            )

        basis = basis_1d if component_dim == 1 else tensor_basis
        component_data = z[:, :component_dim]
        psi = basis.evaluate(component_data, degree)

        weights = np.asarray(weights_by_component[component_dim], dtype=float).reshape(-1)
        if psi.shape[1] != weights.size:
            raise ValueError(
                f"Weight size mismatch for component {component_dim}: "
                f"expected {psi.shape[1]}, got {weights.size}."
            )

        mapped[:, component_dim - 1] = psi @ weights

    return mapped


KRMap1D = KRMapComponent # backwards compatibility alias
