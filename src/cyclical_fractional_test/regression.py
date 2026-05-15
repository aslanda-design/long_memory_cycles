from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import InvalidConfigurationError


@dataclass
class RegressionResult:
    """OLS fit of the fractional-filtered model YD = XD beta + residuals."""

    betas: np.ndarray           # Estimated coefficients, shape (p,).
    fitted_values: np.ndarray   # X_filtered @ betas, shape (T,).
    residuals: np.ndarray       # y_filtered - fitted_values, shape (T,).
    residual_sum_squares: float  # sum(residuals^2).
    rank: int                   # Numerical rank of X_filtered.
    condition_number: float     # Condition number of X_filtered.


def fit_filtered_regression(
    y_filtered: np.ndarray,
    X_filtered: np.ndarray,
) -> RegressionResult:
    """Fit y_filtered = X_filtered @ beta + residuals via OLS (np.linalg.lstsq)."""

    _validate_fit_filtered_regression(y_filtered, X_filtered)
    
    y_arr = np.asarray(y_filtered, dtype=float)
    X_arr = np.asarray(X_filtered, dtype=float)
    betas, _, rank, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
    fitted_values = X_arr @ betas
    residuals = compute_residuals(y_arr, fitted_values)
    rss = compute_residual_sum_squares(residuals)
    cond = float(np.linalg.cond(X_arr))
    return RegressionResult(
        betas=betas,
        fitted_values=fitted_values,
        residuals=residuals,
        residual_sum_squares=rss,
        rank=int(rank),
        condition_number=cond,
    )


def compute_residuals(
    y_filtered: np.ndarray,
    fitted_values: np.ndarray,
) -> np.ndarray:
    """Return y_filtered - fitted_values."""
    _validate_compute_residuals(y_filtered, fitted_values)
    return np.asarray(y_filtered, dtype=float) - np.asarray(fitted_values, dtype=float)


def compute_residual_sum_squares(residuals: np.ndarray) -> float:
    """Return RSS = sum_t residuals[t]^2."""
    _validate_1d_array(residuals, "residuals")
    return float(np.sum(np.asarray(residuals, dtype=float) ** 2))


def compute_time_variance(residuals: np.ndarray) -> float:
    """Return VAR(R,D) = (1/T) sum_t residuals[t]^2 = mean(residuals^2).

    Second moment of residuals — not centered variance.
    """
    _validate_1d_array(residuals, "residuals")
    return float(np.mean(np.asarray(residuals, dtype=float) ** 2))


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_fit_filtered_regression(y_filtered: object, X_filtered: object) -> None:
    try:
        y_arr = np.asarray(y_filtered, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"y_filtered must be numeric: {exc}") from exc
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise InvalidConfigurationError("y_filtered must be a non-empty 1-D array.")
    if not np.all(np.isfinite(y_arr)):
        raise InvalidConfigurationError("y_filtered contains non-finite values.")
    try:
        X_arr = np.asarray(X_filtered, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"X_filtered must be numeric: {exc}") from exc
    if X_arr.ndim != 2:
        raise InvalidConfigurationError(f"X_filtered must be 2-D, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise InvalidConfigurationError("X_filtered contains non-finite values.")
    if X_arr.shape[0] != len(y_arr):
        raise InvalidConfigurationError(
            f"X_filtered.shape[0]={X_arr.shape[0]} must equal len(y_filtered)={len(y_arr)}."
        )
    if X_arr.shape[1] < 1:
        raise InvalidConfigurationError("X_filtered must have at least 1 column.")


def _validate_compute_residuals(y_filtered: object, fitted_values: object) -> None:
    try:
        y_arr = np.asarray(y_filtered, dtype=float)
        f_arr = np.asarray(fitted_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"Inputs must be numeric: {exc}") from exc
    if y_arr.shape != f_arr.shape:
        raise InvalidConfigurationError(
            f"y_filtered and fitted_values must have the same shape; "
            f"got {y_arr.shape} and {f_arr.shape}."
        )


def _validate_1d_array(arr: object, name: str) -> None:
    try:
        arr_np = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigurationError(f"{name} must be numeric: {exc}") from exc
    if arr_np.ndim != 1 or arr_np.size == 0:
        raise InvalidConfigurationError(f"{name} must be a non-empty 1-D array.")
