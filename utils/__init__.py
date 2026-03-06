from .projection_result import ProjectionResult
from .projection_solver import (
    ConvexProjectionSolver,
    DykstraProjectionSolver,
    DykstraMapHybridSolver,
    DykstraStallDetectionSolver,
)
from .data_generator import DataGenerator, generate_crescent_data_2d, generate_crescent_data_nd
from .hermite import hermite_polynomial
from .optimal_transport import (
    HermiteBasis,
    TensorHermiteBasis,
    KRMap,
    KRMapComponent,
    KRMap1D,
)
from .pgd_solver import ProjectedGradientDescent
from .plotter import (
    AXIS_LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    DistributionPlotter,
    DykstraPlotter,
)