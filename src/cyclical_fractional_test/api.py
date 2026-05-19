from __future__ import annotations

import dataclasses
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from .chebyshev import build_chebyshev_design
from .config import CyclicalTestConfig
from .diagnostics import build_test_diagnostics
from .evaluation import evaluate_candidate
from .grid import build_d_grid, build_r_grid_around_peak, build_single_cycle_candidate_grid
from .results import CyclicalFractionalTestResult
from .scoring import TopKSelector
from .spectral import compute_document_periodogram, find_periodogram_peak
from .validation import validate_config, validate_series


def compute_periodogram(y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return the periodogram of a time series.

    Returns (lambdas, I_y) where lambdas are the Fourier frequencies and
    I_y are the corresponding periodogram values, normalised as I(λ_j) = |FFT|²/(2πT).
    """
    arr = validate_series(y)
    return compute_document_periodogram(arr)


def run_cyclical_fractional_test(
    y: Any,
    config: Optional[CyclicalTestConfig] = None,
    **kwargs: Any,
) -> CyclicalFractionalTestResult:
    """Run the fractional cyclic long-memory test in single-cycle mode.

    If config is None, CyclicalTestConfig defaults are used.
    kwargs override individual config fields (e.g., top_k=3).
    Only stochastic_cycle_mode='single' is supported; other modes raise NotImplementedError.
    The result includes a TestDiagnostics object in result.diagnostics.
    """
    if config is None:
        config = CyclicalTestConfig()
    if kwargs:
        config = dataclasses.replace(config, **kwargs)

    arr = validate_series(y)
    validate_config(config)

    if config.stochastic_cycle_mode != "single":
        raise NotImplementedError(
            f"stochastic_cycle_mode={config.stochastic_cycle_mode!r} is not yet "
            "supported in run_cyclical_fractional_test. "
            "Use stochastic_cycle_mode='single'."
        )

    T = len(arr)
    X = build_chebyshev_design(T, config.n_deterministic_cycles, config.include_intercept)

    lambdas_y, I_y = compute_document_periodogram(arr)
    r_peak = find_periodogram_peak(I_y, exclude_zero=config.exclude_zero_frequency)

    r_candidates = build_r_grid_around_peak(r_peak, config.r_window, T)

    #Here, we could optimize the grid by choosing the d values given by the normal function
    d_grid = build_d_grid(config.d_grid)

    selector = TopKSelector(k=config.top_k, statistic_mode=config.statistic_mode)

    n_evaluated = 0
    n_valid = 0
    warnings: List[str] = []

    for cycles in build_single_cycle_candidate_grid(r_candidates, d_grid):
        candidate_result = evaluate_candidate(arr, X, cycles, config)
        selector.consider(candidate_result)
        n_evaluated += 1
        n_valid += 1
        logger.info(
            "candidate R=%d D=%.2f  XA=%.6f",
            cycles[0].R,
            cycles[0].D,
            candidate_result.xa,
        )

    top_k_results = selector.get_top_k()
    best_result = selector.get_best()

    #The second order auto-regresive model would go here

    diagnostics = build_test_diagnostics(
        n_candidates_evaluated=n_evaluated,
        n_valid_candidates=n_valid,
        n_failed_candidates=0,
        warnings=warnings,
        lambdas_y=lambdas_y,
        periodogram_y=I_y,
        r_peak=r_peak,
        r_candidates=r_candidates,
        d_grid=d_grid,
        config=config,
    )

    return CyclicalFractionalTestResult(
        best_result=best_result,
        top_k_results=top_k_results,
        r_peak=r_peak,
        r_candidates=r_candidates,
        d_grid=d_grid,
        config=config,
        n_candidates_evaluated=n_evaluated,
        diagnostics=diagnostics,
    )
