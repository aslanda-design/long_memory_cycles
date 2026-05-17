from __future__ import annotations

from typing import Sequence

import numpy as np

from .exceptions import InvalidConfigurationError, InvalidCycleError
from .filters import filter_response_and_design
from .regression import compute_time_variance, fit_filtered_regression
from .results import GridCandidateResult, StochasticCycle
from .scoring import compute_test_star_statistic, compute_test_statistic
from .spectral import (
    compute_frequency_variance_dynamic,
    compute_psi_dynamic,
    compute_residual_periodogram,
    compute_xa_dynamic,
    compute_xaa_dynamic,
)


def evaluate_candidate(
    y: np.ndarray,
    X: np.ndarray,
    cycles: Sequence[StochasticCycle],
    config: object,
) -> GridCandidateResult:
    """Evaluate one grid candidate and return its full set of statistics.

    Orchestrates Waves 5–11 for a single (cycles, config) combination.
    Does not build grids, select R*, perform scoring, or rank results.
    """
    _validate_evaluate_candidate(y, X, cycles, config)

    mode = config.stochastic_cycle_mode
    cycles_t = tuple(cycles)
    T = len(y)

    psi = compute_psi_dynamic(T, cycles_t, mode, config.drop_singular_frequency)
    xaa = compute_xaa_dynamic(psi, mode)

    y_f, X_f = filter_response_and_design(y, X, cycles_t, mode=mode)
    reg = fit_filtered_regression(y_f, X_f)

    _, I_resid = compute_residual_periodogram(reg.residuals)
    variance_time = compute_time_variance(reg.residuals)
    variance_frequency = compute_frequency_variance_dynamic(
        I_resid, cycles_t, mode=mode, drop_frequency=config.drop_singular_frequency
    )

    xa = compute_xa_dynamic(psi, I_resid, cycles=cycles_t, mode=mode)
    test_value = compute_test_statistic(T, xa, xaa, variance_time)
    test_star_value = compute_test_star_statistic(T, xa, xaa, variance_frequency)

    return GridCandidateResult(
        cycles=cycles_t,
        test_value=test_value,
        test_star_value=test_star_value,
        abs_test_value=abs(test_value),
        abs_test_star_value=abs(test_star_value),
        xa=xa,
        xaa=xaa,
        variance_time=variance_time,
        variance_frequency=variance_frequency,
        betas=reg.betas,
        residuals=reg.residuals,
        residual_sum_squares=reg.residual_sum_squares,
    )


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know the exact
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_evaluate_candidate(
    y: object, X: object, cycles: object, config: object
) -> None:
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"y must be numeric: {exc}") from exc
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise InvalidConfigurationError("y must be a non-empty 1-D array.")

    try:
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"X must be numeric: {exc}") from exc
    if X_arr.ndim != 2:
        raise InvalidConfigurationError(
            f"X must be 2-D, got shape {X_arr.shape}."
        )
    if X_arr.shape[0] != len(y_arr):
        raise InvalidConfigurationError(
            f"X.shape[0]={X_arr.shape[0]} must equal len(y)={len(y_arr)}."
        )

    try:
        cycle_list = list(cycles)
    except TypeError as exc:
        raise InvalidConfigurationError(
            f"cycles must be iterable, got {type(cycles).__name__}."
        ) from exc
    if len(cycle_list) == 0:
        raise InvalidConfigurationError("cycles must not be empty.")

    mode = getattr(config, "stochastic_cycle_mode", None)
    if mode == "single" and len(cycle_list) != 1:
        raise InvalidCycleError(
            f"stochastic_cycle_mode='single' requires exactly 1 cycle, "
            f"got {len(cycle_list)}."
        )
