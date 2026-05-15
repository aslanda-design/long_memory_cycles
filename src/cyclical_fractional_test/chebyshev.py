from __future__ import annotations

import numpy as np

from .exceptions import InvalidConfigurationError


def build_single_chebyshev_polynomial(T: int, order: int) -> np.ndarray:
    """Build the deterministic Chebyshev polynomial P_k(t) for t = 1, ..., T.

    P_0(t) = 1
    P_k(t) = 2 * cos(k * pi * (t - 0.5) / T)   for k >= 1
    """
    _validate_single_polynomial(T, order)
    
    if order == 0:
        return np.ones(T, dtype=float)
    t = np.arange(1, T + 1, dtype=float)
    return 2.0 * np.cos(order * np.pi * (t - 0.5) / T)


def build_chebyshev_design(
    T: int,
    n_cycles: int,
    include_intercept: bool = False,
) -> np.ndarray:
    """Build the deterministic Chebyshev design matrix.

    Without intercept: columns [P_1, ..., P_n_cycles], shape (T, n_cycles).
    With intercept:    columns [P_0, P_1, ..., P_n_cycles], shape (T, n_cycles+1).
    Generates exactly n_cycles columns — no zero-padded extras.
    """
    _validate_chebyshev_design(T, n_cycles, include_intercept)

    start_order = 0 if include_intercept else 1
    n_cols = n_cycles + 1 if include_intercept else n_cycles
    X = np.empty((T, n_cols), dtype=float)
    for col_idx, k in enumerate(range(start_order, n_cycles + 1)):
        X[:, col_idx] = build_single_chebyshev_polynomial(T, k)
    return X


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact 
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_single_polynomial(T: int, order: int) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T <= 0:
        raise InvalidConfigurationError(f"T must be > 0, got {T}.")
    if isinstance(order, bool) or not isinstance(order, int):
        raise InvalidConfigurationError(
            f"order must be an int, got {type(order).__name__}."
        )
    if order < 0:
        raise InvalidConfigurationError(f"order must be >= 0, got {order}.")


def _validate_chebyshev_design(T: int, n_cycles: int, include_intercept: bool) -> None:
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T <= 0:
        raise InvalidConfigurationError(f"T must be > 0, got {T}.")
    if isinstance(n_cycles, bool) or not isinstance(n_cycles, int):
        raise InvalidConfigurationError(
            f"n_cycles must be an int, got {type(n_cycles).__name__}."
        )
    if n_cycles < 1:
        raise InvalidConfigurationError(f"n_cycles must be >= 1, got {n_cycles}.")
    if not isinstance(include_intercept, bool):
        raise InvalidConfigurationError(
            f"include_intercept must be a bool, got {type(include_intercept).__name__}."
        )
