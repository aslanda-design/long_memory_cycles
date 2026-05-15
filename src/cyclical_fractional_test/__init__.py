from .api import run_cyclical_fractional_test
from .chebyshev import (
    build_chebyshev_design,
    build_single_chebyshev_polynomial,
)
from .config import CyclicalTestConfig
from .exceptions import (
    CyclicalFractionalTestError,
    InvalidCycleError,
    InvalidConfigurationError,
    InvalidSeriesError,
)
from .grid import (
    build_d_grid,
    build_multi_cycle_candidate_grid,
    build_r_grid_around_peak,
    build_single_cycle_candidate_grid,
    candidate_iterator,
)
from .results import (
    CyclicalFractionalTestResult,
    GridCandidateResult,
    StochasticCycle,
)
from .spectral import (
    compute_document_periodogram,
    compute_psi_dynamic,
    compute_psi_multi_cycle,
    compute_psi_single_cycle,
    compute_xaa_dynamic,
    compute_xaa_multi_cycle,
    compute_xaa_single_cycle,
    find_periodogram_peak,
    find_top_periodogram_peaks,
)

__all__ = [
    # api
    "run_cyclical_fractional_test",
    # config
    "CyclicalTestConfig",
    # results
    "StochasticCycle",
    "GridCandidateResult",
    "CyclicalFractionalTestResult",
    # exceptions
    "CyclicalFractionalTestError",
    "InvalidSeriesError",
    "InvalidConfigurationError",
    "InvalidCycleError",
    # chebyshev
    "build_single_chebyshev_polynomial",
    "build_chebyshev_design",
    # spectral — periodogram
    "compute_document_periodogram",
    "find_periodogram_peak",
    "find_top_periodogram_peaks",
    # spectral — psi / XAA
    "compute_psi_single_cycle",
    "compute_psi_multi_cycle",
    "compute_xaa_single_cycle",
    "compute_xaa_multi_cycle",
    "compute_psi_dynamic",
    "compute_xaa_dynamic",
    # grid
    "build_r_grid_around_peak",
    "build_d_grid",
    "build_single_cycle_candidate_grid",
    "build_multi_cycle_candidate_grid",
    "candidate_iterator",
]
