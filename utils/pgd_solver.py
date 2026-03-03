"""Projected Gradient Descent solver with a pluggable Dykstra projection step.

This module is a pure optimisation engine.  It has no knowledge of the
specific problem domain (e.g. optimal transport, polynomial bases).  Its
sole responsibility is to minimise an arbitrary differentiable objective
over a polyhedral feasible set, using gradient descent steps followed by
Dykstra-based projections.
"""

from typing import Any, Callable

import numpy as np


class ProjectedGradientDescent:
    """Projected Gradient Descent (PGD) optimiser.

    At each outer iteration the optimiser takes an unconstrained gradient
    step and then projects the result back onto the polyhedral feasible set
    ``{w | A w <= b}`` using a user-supplied Dykstra projection solver.

    Parameters
    ----------
    learning_rate : float
        Step size (alpha) for the gradient descent update.
    max_outer_iter : int
        Maximum number of outer PGD iterations.
    projection_solver_class : type
        A *class* (not an instance) that implements the Dykstra projection
        interface.  It must accept at least the keyword arguments ``z``,
        ``A``, ``b``, and ``max_iter``, and expose a ``solve()`` method
        that returns an object with a ``.projection`` attribute.
    **dykstra_kwargs
        Additional keyword arguments forwarded verbatim to
        ``projection_solver_class`` on every inner instantiation (e.g.
        ``max_iter`` for the inner Dykstra loop, ``track_error``,
        ``delete_spaces``).
    """

    def __init__(
        self,
        learning_rate: float,
        max_outer_iter: int,
        projection_solver_class: type,
        **dykstra_kwargs: Any,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_outer_iter = max_outer_iter
        self.projection_solver_class = projection_solver_class
        self.dykstra_kwargs = dykstra_kwargs

    def optimise(
        self,
        w_init: np.ndarray,
        objective_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        A_constraint: np.ndarray,
        b_constraint: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Run Projected Gradient Descent.

        Parameters
        ----------
        w_init : np.ndarray
            Initial guess for the decision variable vector, shape ``(n,)``.
        objective_fn : callable
            A function ``f(w) -> float`` returning the scalar cost to
            minimise.
        gradient_fn : callable
            A function ``g(w) -> np.ndarray`` returning the gradient of the
            cost with respect to ``w``.
        A_constraint : np.ndarray
            Constraint matrix of shape ``(m, n)`` defining the polyhedral
            feasible set ``A w <= b``.
        b_constraint : np.ndarray
            Constraint offset vector of shape ``(m,)``.

        Returns
        -------
        w : np.ndarray
            The optimised decision variable vector.
        history : dict
            A dictionary with the following keys:

            * ``"objective_value"`` – list of ``float``, the objective
              evaluated at ``w`` at the start of each iteration (length
              ``max_outer_iter + 1``, including the initial value).
            * ``"dykstra_inner_iters"`` – list of ``int``, the number of
              inner Dykstra cycles used at each outer iteration (length
              ``max_outer_iter``).  Recorded as the configured inner
              ``max_iter`` when the solver does not expose an explicit
              iteration count.
            * ``"projection_results"`` – list of projection result
                objects returned by the inner Dykstra solver, one per outer
                iteration.
        """
        w = w_init.copy()

        inner_max = self.dykstra_kwargs.get("max_iter", 0)

        objective_values: list[float] = [float(objective_fn(w))]
        dykstra_inner_iters: list[int] = []
        projection_results: list[Any] = []

        for _ in range(self.max_outer_iter):
            grad = gradient_fn(w)
            w_tilde = w - self.learning_rate * grad

            solver = self.projection_solver_class(
                z=w_tilde,
                A=A_constraint,
                b=b_constraint,
                **self.dykstra_kwargs,
            )
            result = solver.solve()
            projection_results.append(result)
            w = result.projection

            objective_values.append(float(objective_fn(w)))

            if (
                hasattr(result, "squared_errors")
                and result.squared_errors is not None
            ):
                dykstra_inner_iters.append(len(result.squared_errors) - 1)
            else:
                dykstra_inner_iters.append(inner_max)

        history: dict = {
            "objective_value": objective_values,
            "dykstra_inner_iters": dykstra_inner_iters,
            "projection_results": projection_results,
        }

        return w, history
