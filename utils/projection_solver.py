"""
Projection solvers for projecting a point onto the intersection of half-spaces
using Dykstra's algorithm and variants.

Classes:
    ConvexProjectionSolver: Abstract base class.
    DykstraProjectionSolver: Standard Dykstra's algorithm.
    DykstraMapHybridSolver: Hybrid of MAP and Dykstra's algorithm.
    DykstraStallDetectionSolver: Dykstra with stalling detection and fast-forwarding.

Features:
    - Error tracking with convergence and stalling detection.
    - Active half-space tracking.
    - Inactive half-space removal.
    - Generalised for arbitrary dimensions.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize as scipy_minimize

from utils.projection_result import ProjectionResult


class ConvexProjectionSolver(ABC):
    """Abstract base class for projecting a point onto the intersection of half-spaces."""

    @staticmethod
    def _normalise(normal: np.ndarray, offset: float) -> tuple:
        """Normalise a half-space normal vector and offset to unit length."""
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Zero-norm normal vector encountered.")
        return normal / norm, offset / norm

    @staticmethod
    def _is_in_half_space(point: np.ndarray, normal: np.ndarray,
                          offset: float) -> bool:
        """Check if a point lies within the half-space {x | <x, n> <= b}."""
        return np.dot(point, normal) <= offset

    @staticmethod
    def _project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                                 offset: float) -> np.ndarray:
        """
        Project a point onto the half-space {x | <x, n> <= b}.

        Returns the point unchanged if already feasible, otherwise projects
        onto the boundary hyperplane.
        """
        unit_normal, const_offset = ConvexProjectionSolver._normalise(normal, offset)
        if ConvexProjectionSolver._is_in_half_space(point, unit_normal, const_offset):
            return point
        return point - (np.dot(point, unit_normal) - const_offset) * unit_normal

    @staticmethod
    def _delete_inactive_half_spaces(z: np.ndarray, A: np.ndarray,
                                     b: np.ndarray) -> tuple:
        """Remove half-spaces that the point z already satisfies."""
        active = np.ones(len(b), dtype=bool)
        for m, (normal, offset) in enumerate(zip(A, b)):
            unit_normal, const_offset = ConvexProjectionSolver._normalise(normal, offset)
            if ConvexProjectionSolver._is_in_half_space(z, unit_normal, const_offset):
                active[m] = False
        return A[active], b[active]

    @staticmethod
    def _find_optimal_solution(point: np.ndarray, A: np.ndarray,
                               b: np.ndarray) -> np.ndarray:
        """
        Solve min ||x - point||^2 subject to Ax <= b via quadratic programming
        to obtain the true optimal projection for error tracking.
        """
        constraints = [
            {'type': 'ineq', 'fun': lambda x, i=i: b[i] - np.dot(A[i], x)}
            for i in range(len(b))
        ]
        result = scipy_minimize(
            fun=lambda x: np.sum((x - point) ** 2),
            x0=point,
            method='SLSQP',
            constraints=constraints,
        )
        return result.x

    @staticmethod
    def _beta_check(point: np.ndarray, A: np.ndarray, b: np.ndarray) -> int:
        """Return 1 if the point is in the intersection of all half-spaces, else 0."""
        rounded = np.around(point, decimals=10)
        for normal, offset in zip(A, b):
            unit_normal, const_offset = ConvexProjectionSolver._normalise(normal, offset)
            if not ConvexProjectionSolver._is_in_half_space(rounded, unit_normal, const_offset):
                return 0
        return 1

    def __init__(self, z: np.ndarray, A: np.ndarray, b: np.ndarray,
                 max_iter: int, track_error: bool = False,
                 min_error: float = 1e-3,
                 track_active_halfspaces: bool = False,
                 delete_spaces: bool = False):
        """
        Args:
            z: Initial point to project.
            A: Matrix of normal vectors (n_halfspaces x dim).
            b: Vector of offsets (n_halfspaces,).
            max_iter: Maximum number of iterations (full cycles).
            track_error: Whether to track squared error against the QP-optimal solution.
            min_error: Squared-error threshold below which the solver is considered converged.
            track_active_halfspaces: Whether to record per-halfspace activity each cycle.
            delete_spaces: Whether to remove initially-inactive half-spaces before solving.
        """
        self.z = z.copy()
        if delete_spaces:
            self.A, self.b = self._delete_inactive_half_spaces(z, A, b)
        else:
            self.A = A.copy()
            self.b = b.copy()

        self.max_iter = max_iter
        self.track_error = track_error
        self.min_error = min_error
        self.track_active_halfspaces = track_active_halfspaces

        self.n = self.A.shape[0]
        self.dim = len(self.z)
        self.x = self.z.copy()
        self.e = [np.zeros(self.dim) for _ in range(self.n)]

        if self.track_error:
            self.actual_projection = self._find_optimal_solution(
                self.z, self.A, self.b
            )
            self.squared_errors = np.zeros(max_iter + 1)
            self.stalled_errors = np.full(max_iter + 1, np.nan)
            self.converged_errors = np.full(max_iter + 1, np.nan)

        if self.track_active_halfspaces:
            self.active_half_spaces = np.zeros((self.n, max_iter + 1))

    @abstractmethod
    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray) -> None:
        """Update the increment vector for half-space m."""
        pass

    def _track_activity(self, cycle_index: int) -> None:
        """Record which half-spaces are active after a complete cycle."""
        if not self.track_active_halfspaces:
            return
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
            if not self._is_in_half_space(self.x + self.e[m], normal, offset):
                self.active_half_spaces[m][cycle_index] = 1

    def _track_error_at(self, i: int) -> None:
        """Track squared error, convergence, and stalling at cycle index i."""
        if not self.track_error:
            return
        distance = self.actual_projection - self.x
        error = round(np.dot(distance, distance), 10)
        self.squared_errors[i] = error

        if error < self.min_error:
            self.converged_errors[i] = error
        elif i > 0:
            prev_sq = self.squared_errors[i - 1]
            prev_st = self.stalled_errors[i - 1]
            if prev_sq == error or (not np.isnan(prev_st) and prev_st == error):
                self.stalled_errors[i] = error

    def _format_output(self) -> ProjectionResult:
        """Package solver state into a ProjectionResult."""
        return ProjectionResult(
            projection=self.x,
            squared_errors=self.squared_errors if self.track_error else None,
            stalled_errors=self.stalled_errors if self.track_error else None,
            converged_errors=self.converged_errors if self.track_error else None,
            active_half_spaces=(self.active_half_spaces
                                if self.track_active_halfspaces else None),
        )

    @abstractmethod
    def solve(self) -> ProjectionResult:
        """Run the projection algorithm and return results."""
        pass


class DykstraProjectionSolver(ConvexProjectionSolver):
    """Standard Dykstra's algorithm for projection onto intersection of half-spaces."""

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray) -> None:
        self.e[m] = self.e[m] + (x_temp - x)

    def solve(self) -> ProjectionResult:
        self._track_error_at(0)
        self._track_activity(0)

        for i in range(self.max_iter):
            for m, (normal, offset) in enumerate(zip(self.A, self.b)):
                x_temp = self.x.copy()
                self.x = self._project_onto_half_space(
                    x_temp + self.e[m], normal, offset
                )
                self._update_error(m, x_temp, self.x)

            self._track_error_at(i + 1)
            self._track_activity(i + 1)

        return self._format_output()


class DykstraMapHybridSolver(ConvexProjectionSolver):
    """
    Hybrid solver switching between MAP and Dykstra's algorithm.

    Uses MAP (zero error corrections) when the current iterate is feasible,
    and Dykstra error corrections otherwise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_dykstra = [np.zeros(self.dim) for _ in range(self.n)]
        self.e_map = [np.zeros(self.dim) for _ in range(self.n)]

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray) -> None:
        self.e_dykstra[m] = self.e_dykstra[m] + (x_temp - x)

    def solve(self) -> ProjectionResult:
        self._track_error_at(0)
        self._track_activity(0)

        for i in range(self.max_iter):
            if self._beta_check(self.x, self.A, self.b) == 1:
                self.e = self.e_dykstra
            else:
                self.e = self.e_map

            for m, (normal, offset) in enumerate(zip(self.A, self.b)):
                x_temp = self.x.copy()
                self.x = self._project_onto_half_space(
                    x_temp + self.e[m], normal, offset
                )
                self._update_error(m, x_temp, self.x)

            self._track_error_at(i + 1)
            self._track_activity(i + 1)

        return self._format_output()


class DykstraStallDetectionSolver(ConvexProjectionSolver):
    """
    Modified Dykstra's algorithm with stalling detection.

    Detects when projections stop making progress between cycles and
    fast-forwards the error corrections to escape the stall.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stalling = False
        self.m_stalling = None
        self.prev_cycle_x = np.tile(self.z, (self.n, 1))
        self.curr_cycle_x = np.zeros((self.n, self.dim))
        self._prev_active = np.zeros(self.n, dtype=bool)

    def _update_error(self, m: int, x_temp: np.ndarray, x: np.ndarray) -> None:
        self.e[m] = self.e[m] + (x_temp - x)

    def _track_activity(self, cycle_index: int) -> None:
        """Record activity and update internal stall-detection state."""
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
            is_active = not self._is_in_half_space(
                self.x + self.e[m], normal, offset
            )
            self._prev_active[m] = is_active
            if self.track_active_halfspaces:
                self.active_half_spaces[m][cycle_index] = 1 if is_active else 0

    def _handle_stalling(self) -> None:
        """Fast-forward error corrections to escape a detected stall."""
        if not (self.stalling and self.m_stalling is not None):
            return

        ff_candidates = []
        for m, (normal, offset) in enumerate(zip(self.A, self.b)):
            dp = np.dot(self.prev_cycle_x[m - 1], normal)
            if dp < offset:
                ff = np.ceil(-np.dot(self.e[m], normal) / (dp - offset))
                ff_candidates.append(ff)
            else:
                ff_candidates.append(1e6)

        n_fast_forward = int(min(ff_candidates)) - 1

        for m in range(self.n):
            self.e[m] = self.e[m] + n_fast_forward * (
                self.prev_cycle_x[m - 1] - self.prev_cycle_x[m]
            )

        self.stalling = False
        self.m_stalling = None

    def solve(self) -> ProjectionResult:
        self.stalling = False
        self._track_error_at(0)
        self._track_activity(0)

        for i in range(self.max_iter):
            for m, (normal, offset) in enumerate(zip(self.A, self.b)):
                x_temp = self.x.copy()
                self._handle_stalling()
                self.x = self._project_onto_half_space(
                    x_temp + self.e[m], normal, offset
                )
                self._update_error(m, x_temp, self.x)
                self.curr_cycle_x[m] = self.x.copy()

                if (i > 0 and not self.stalling
                        and self._prev_active[m]
                        and np.array_equal(self.curr_cycle_x[m],
                                           self.prev_cycle_x[m])):
                    self.stalling = True
                    self.m_stalling = m

            self._track_error_at(i + 1)
            self._track_activity(i + 1)
            self.prev_cycle_x[:] = self.curr_cycle_x

        return self._format_output()
