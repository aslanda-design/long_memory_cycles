from __future__ import annotations

import dataclasses
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from .chebyshev import build_chebyshev_design
from .config import CyclicalTestConfig
from .diagnostics import build_test_diagnostics
from .evaluation import evaluate_candidate, evaluate_r_with_adaptive_d
from .grid import (
    build_d_grid_for_strategy,
    build_r_grid_around_peak,
    build_single_cycle_candidate_grid,
)
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
    kwargs override individual config fields (e.g., top_k=3 or error_model="ar1").
    Only stochastic_cycle_mode='single' is supported; other modes raise NotImplementedError.
    The result includes a TestDiagnostics object and per-candidate AR nuisance estimates.
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
    periodogram = I_y[:len(I_y) // 2]
    r_peak = find_periodogram_peak(periodogram, exclude_zero=config.exclude_zero_frequency)

    r_candidates = build_r_grid_around_peak(r_peak, config.r_window, T)

    # The reported grid is the full fixed grid, or the coarse seed for adaptive search.
    d_grid = build_d_grid_for_strategy(config)

    selector = TopKSelector(k=config.top_k, statistic_mode=config.statistic_mode)

    n_evaluated = 0
    warnings: List[str] = []
    adaptive_info: Optional[dict] = None

    if config.d_search_strategy == "fixed_grid":
        for cycles in build_single_cycle_candidate_grid(r_candidates, d_grid):
            candidate_result = evaluate_candidate(arr, X, cycles, config)
            selector.consider(candidate_result)
            n_evaluated += 1
            logger.info(
                "candidate R=%d D=%.2f  XA=%.6f",
                cycles[0].R,
                cycles[0].D,
                candidate_result.xa,
            )
    else:
        # Adaptive coarse-to-fine search: one best candidate per frequency R.
        best_coarse_d: List[float] = []
        final_d: List[float] = []
        n_coarse = 0
        n_fine = 0
        for R in r_candidates:
            search = evaluate_r_with_adaptive_d(arr, X, int(R), config)
            selector.consider(search.best_result)
            n_evaluated += search.n_candidates_evaluated
            n_coarse += search.n_coarse_evaluated
            n_fine += search.n_fine_evaluated
            best_coarse_d.append(search.best_coarse_d)
            final_d.append(search.best_d)
            logger.info(
                "R=%d adaptive best D=%.2f (coarse D=%.2f, %d candidates)",
                int(R),
                search.best_d,
                search.best_coarse_d,
                search.n_candidates_evaluated,
            )
        adaptive_info = {
            "d_coarse_grid": d_grid,
            "d_fine_step": config.d_fine_step,
            "d_fine_radius": config.d_fine_radius,
            "best_coarse_d_per_r": best_coarse_d,
            "final_d_per_r": final_d,
            "n_coarse_evaluations": n_coarse,
            "n_fine_evaluations": n_fine,
        }

    n_valid = n_evaluated
    top_k_results = selector.get_top_k()
    best_result = selector.get_best()

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
        adaptive_info=adaptive_info,
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
