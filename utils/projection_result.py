from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ProjectionResult:
    """Container for projection solver results."""
    projection: np.ndarray
    iterations_run: int | None = None
    terminated_early: bool = False
    termination_reason: str = "max_iter"
    squared_errors: Optional[np.ndarray] = None
    stalled_errors: Optional[np.ndarray] = None
    converged_errors: Optional[np.ndarray] = None
    active_half_spaces: Optional[np.ndarray] = None
    iterates: Optional[np.ndarray] = None
