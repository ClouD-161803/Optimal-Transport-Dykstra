from .projection_result import ProjectionResult
from .projection_solver import (
    ConvexProjectionSolver,
    DykstraProjectionSolver,
    DykstraMapHybridSolver,
    DykstraStallDetectionSolver,
)
from .data_generator import generate_crescent_data_2d
from .hermite import hermite_polynomial
from .optimal_transport import HermiteBasis, KRMap1D
from .pgd_solver import ProjectedGradientDescent