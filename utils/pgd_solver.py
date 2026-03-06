"""Projected Gradient Descent solver with a pluggable Dykstra projection step.
"""

from typing import Any, Callable

import numpy as np


class ProjectedGradientDescent:
    """Projected Gradient Descent (PGD) optimiser.

    At each outer iteration the optimiser takes an unconstrained gradient
    step and then projects the result back onto the polyhedral feasible set
    ``{w | A w <= b}`` using a user-supplied Dykstra projection solver.

    The inner Dykstra solver is run with an inexact schedule: at outer
    iteration ``t`` the stopping tolerance is ``base_tol * t**(-inexact_power)``
    and the iteration ceiling is ``int(100 * t**1.1)``.  The sequence
    ``{base_tol * t**(-inexact_power)}`` is summable for ``inexact_power > 1``,
    satisfying the standard inexact PGD convergence condition.

    Parameters
    ----------
    learning_rate : float
        Step size for the gradient descent update.
    max_outer_iter : int
        Maximum number of outer PGD iterations.
    projection_solver_class : type
        A *class* (not an instance) implementing the Dykstra projection
        interface.  It must accept ``z``, ``A``, ``b``, ``max_iter``, and
        ``min_error`` as keyword arguments and expose a ``solve()`` method
        returning an object with a ``.projection`` attribute.
    gradient_clip_value : float or None, optional
        If given, gradients are clipped element-wise to ``[-v, v]`` before
        each step.
    l1_reg : float, optional
        L1 regularisation strength.  Adds ``l1_reg * ||w||_1`` to the
        objective and ``l1_reg * sign(w)`` to the gradient.  Default ``0.0``
        (no regularisation).
    inexact_power : float, optional
        Exponent for the inexact projection schedule.  At outer iteration
        ``t``, the inner solver tolerance is ``base_tol * t**(-inexact_power)``
        and the iteration ceiling is ``int(100 * t**1.1)``.  Default ``1.1``.
    base_tol : float, optional
        Base tolerance for the inexact projection schedule.  Default ``1e-3``.
    **dykstra_kwargs
        Additional keyword arguments forwarded verbatim to
        ``projection_solver_class`` on every inner instantiation (e.g.
        ``track_error``, ``delete_spaces``).  Do not pass ``max_iter`` or
        ``min_error`` here; both are set dynamically by the inexact schedule.
    """

    def __init__(
        self,
        learning_rate: float,
        max_outer_iter: int,
        projection_solver_class: type,
        gradient_clip_value: float | None = None,
        l1_reg: float = 0.0,
        inexact_power: float = 1.1,
        base_tol: float = 1e-3,
        **dykstra_kwargs: Any,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_outer_iter = max_outer_iter
        self.projection_solver_class = projection_solver_class
        self.gradient_clip_value = gradient_clip_value
        self.l1_reg = l1_reg
        self.inexact_power = inexact_power
        self.base_tol = base_tol
        # max_iter and min_error are set dynamically by the inexact schedule;
        # remove them if an old caller accidentally passes them here.
        dykstra_kwargs.pop("max_iter", None)
        dykstra_kwargs.pop("min_error", None)
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
              ``max_outer_iter``).  Recorded as the dynamic iteration
              ceiling when the solver does not expose an explicit count.
            * ``"projection_results"`` – list of projection result objects
              returned by the inner Dykstra solver, one per outer iteration.
        """
        w = w_init.copy()

        def _objective(w: np.ndarray) -> float:
            val = float(objective_fn(w))
            if self.l1_reg:
                val += self.l1_reg * float(np.sum(np.abs(w)))
            return val

        def _gradient(w: np.ndarray) -> np.ndarray:
            g = gradient_fn(w)
            if self.l1_reg:
                g = g + self.l1_reg * np.sign(w)
            return g

        objective_values: list[float] = [_objective(w)]
        dykstra_inner_iters: list[int] = []
        projection_results: list[Any] = []

        for t in range(1, self.max_outer_iter + 1):
            grad = _gradient(w)
            if self.gradient_clip_value is not None:
                clip_value = abs(float(self.gradient_clip_value))
                grad = np.clip(grad, -clip_value, clip_value)
            w_tilde = w - self.learning_rate * grad

            current_tol = self.base_tol * float(t ** -self.inexact_power)
            current_max_iter = int(100 * (t ** 1.1))

            solver = self.projection_solver_class(
                z=w_tilde,
                A=A_constraint,
                b=b_constraint,
                min_error=current_tol,
                max_iter=current_max_iter,
                **self.dykstra_kwargs,
            )
            result = solver.solve()
            projection_results.append(result)
            w = result.projection

            objective_values.append(_objective(w))

            if (
                hasattr(result, "squared_errors")
                and result.squared_errors is not None
            ):
                dykstra_inner_iters.append(len(result.squared_errors) - 1)
            else:
                dykstra_inner_iters.append(current_max_iter)

        history: dict = {
            "objective_value": objective_values,
            "dykstra_inner_iters": dykstra_inner_iters,
            "projection_results": projection_results,
        }

        return w, history
