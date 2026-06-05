from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from .chebyshev import build_chebyshev_design
from .config import CyclicalTestConfig
from .diagnostics import build_test_diagnostics
from .evaluation import evaluate_candidate, evaluate_r_with_adaptive_d
from .exceptions import InvalidConfigurationError
from .grid import (
    build_d_grid_for_strategy,
    build_r_grid_around_peak,
    build_single_cycle_candidate_grid,
)
from .results import CyclicalFractionalTestResult, GridCandidateResult
from .scoring import TopKSelector, score_candidate
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
    threshold: Optional[float] = None,
    **kwargs: Any,
) -> CyclicalFractionalTestResult:
    """Run the fractional cyclic long-memory test in single-cycle mode.

    If config is None, CyclicalTestConfig defaults are used.
    kwargs override individual config fields (e.g., top_k=3 or error_model="ar1").
    Only stochastic_cycle_mode='single' is supported; other modes raise NotImplementedError.
    The result includes a TestDiagnostics object and per-candidate AR nuisance estimates.

    If threshold is given (a positive float), the result's under_threshold_results
    collects every evaluated candidate whose statistic score (|TEST| or |TEST*|,
    per config.statistic_mode) falls below it, grouped by frequency R so each R maps
    to its passing D values. When threshold is None this object stays None.
    """
    if config is None:
        config = CyclicalTestConfig()
    if kwargs:
        config = dataclasses.replace(config, **kwargs)

    arr = validate_series(y)
    validate_config(config)

    threshold_value = _validate_threshold(threshold) if threshold is not None else None

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

    # When a threshold is requested, collect every evaluated candidate that scores
    # below it, grouped by R; otherwise leave the object as None.
    under_threshold_results: Optional[Dict[int, List[GridCandidateResult]]] = (
        {} if threshold_value is not None else None
    )

    n_evaluated = 0
    warnings: List[str] = []
    adaptive_info: Optional[dict] = None

    if config.d_search_strategy == "fixed_grid":
        for cycles in build_single_cycle_candidate_grid(r_candidates, d_grid):
            candidate_result = evaluate_candidate(arr, X, cycles, config)
            selector.consider(candidate_result)
            if under_threshold_results is not None:
                _record_if_under_threshold(
                    under_threshold_results, candidate_result, threshold_value, config.statistic_mode
                )
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
            if under_threshold_results is not None:
                for candidate in search.all_results:
                    _record_if_under_threshold(
                        under_threshold_results, candidate, threshold_value, config.statistic_mode
                    )
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

    if under_threshold_results is not None:
        # Order R ascending and, within each R, list the passing D candidates best-first.
        under_threshold_results = {
            R: sorted(candidates, key=lambda c: score_candidate(c, config.statistic_mode))
            for R, candidates in sorted(under_threshold_results.items())
        }

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
        under_threshold_results=under_threshold_results,
    )


def _validate_threshold(threshold: Any) -> float:
    """Return threshold as a positive finite float, or raise InvalidConfigurationError."""
    if isinstance(threshold, bool):
        raise InvalidConfigurationError("threshold must be a real number, got bool.")
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"threshold must be numeric: {exc}") from exc
    if not np.isfinite(value):
        raise InvalidConfigurationError(f"threshold must be finite, got {threshold!r}.")
    if value <= 0.0:
        raise InvalidConfigurationError(f"threshold must be > 0, got {value}.")
    return value


def _record_if_under_threshold(
    bucket: Dict[int, List[GridCandidateResult]],
    candidate: GridCandidateResult,
    threshold: float,
    statistic_mode: str,
) -> None:
    """Append candidate to bucket[R] when its statistic score is below threshold."""
    if score_candidate(candidate, statistic_mode) < threshold:
        bucket.setdefault(int(candidate.cycles[0].R), []).append(candidate)
