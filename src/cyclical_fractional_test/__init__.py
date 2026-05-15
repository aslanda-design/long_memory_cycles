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
from .filters import (
    apply_filter_dynamic,
    apply_fractional_filter_single_series,
    apply_multi_cycle_filter,
    apply_single_cycle_filter,
    compute_fractional_coefficients_dynamic,
    compute_fractional_coefficients_multi_cycle,
    compute_fractional_coefficients_single_cycle,
    compute_mu,
    filter_response_and_design,
)
from .regression import (
    RegressionResult,
    compute_residual_sum_squares,
    compute_residuals,
    compute_time_variance,
    fit_filtered_regression,
)
from .spectral import (
    compute_document_periodogram,
    compute_frequency_variance_dynamic,
    compute_frequency_variance_multi_cycle,
    compute_frequency_variance_single_cycle,
    compute_psi_dynamic,
    compute_psi_multi_cycle,
    compute_psi_single_cycle,
    compute_residual_periodogram,
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
    # filters — Wave 6
    "compute_mu",
    "compute_fractional_coefficients_single_cycle",
    "compute_fractional_coefficients_multi_cycle",
    "compute_fractional_coefficients_dynamic",
    # filters — Wave 7
    "apply_fractional_filter_single_series",
    "apply_single_cycle_filter",
    "apply_multi_cycle_filter",
    "apply_filter_dynamic",
    "filter_response_and_design",
    # regression — Wave 8
    "RegressionResult",
    "fit_filtered_regression",
    "compute_residuals",
    "compute_residual_sum_squares",
    "compute_time_variance",
    # spectral — Wave 9
    "compute_residual_periodogram",
    "compute_frequency_variance_single_cycle",
    "compute_frequency_variance_multi_cycle",
    "compute_frequency_variance_dynamic",
]
