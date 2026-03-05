from .projection_result import ProjectionResult
from .projection_solver import (
    ConvexProjectionSolver,
    DykstraProjectionSolver,
    DykstraMapHybridSolver,
    DykstraStallDetectionSolver,
)
from .data_generator import generate_crescent_data_2d
from .hermite import hermite_polynomial
from .optimal_transport import (
    HermiteBasis,
    TensorHermiteBasis,
    KRMapComponent,
    KRMap1D,
    assemble_component_weights,
    evaluate_kr_map,
    get_tensor_identity_term_index,
    build_identity_initial_guess,
)
from .pgd_solver import ProjectedGradientDescent
from .plotter import (
    AXIS_LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    SUPTITLE_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    DistributionPlotter,
    DykstraPlotter,
)