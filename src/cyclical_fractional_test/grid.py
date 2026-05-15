from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from .exceptions import InvalidConfigurationError
from .results import StochasticCycle
from .validation import validate_d_grid


def build_r_grid_around_peak(r_star: int, r_window: int, T: int) -> np.ndarray:
    """Build the candidate R grid around R*: [max(1, R*−w), ..., min(T−1, R*+w)]."""
    _validate_r_grid(r_star, r_window, T)
    r_min = max(1, r_star - r_window)
    r_max = min(T - 1, r_star + r_window)
    return np.arange(r_min, r_max + 1)


def build_d_grid(d_grid: Any = None) -> np.ndarray:
    """Build the candidate grid for D.

    With None, the default is [0.0, 0.1, ..., 1.0].
    User-supplied grids are validated and returned as float arrays.
    """
    if d_grid is None:
        return np.linspace(0.0, 1.0, 11)
    return validate_d_grid(d_grid)


def build_single_cycle_candidate_grid(
    r_grid: np.ndarray,
    d_grid: np.ndarray,
) -> List[Tuple[StochasticCycle, ...]]:
    """Combine R and D grids into single-cycle candidates.

    Each result is a length-1 tuple so the shape stays compatible with the
    future multi-cycle path.
    """
    return [
        (StochasticCycle(R=int(r), D=float(d)),)
        for r in r_grid
        for d in d_grid
    ]


def build_multi_cycle_candidate_grid(**kwargs: Any) -> None:
    """Reserved for multi-cycle grid construction."""
    raise NotImplementedError(
        "Multi-cycle candidate grid construction is not implemented yet. "
        "Use stochastic_cycle_mode='single' for the current release."
    )


def candidate_iterator(
    r_grid: np.ndarray,
    d_grid: np.ndarray,
    stochastic_cycle_mode: str = "single",
    **kwargs: Any,
) -> List[Tuple[StochasticCycle, ...]]:
    """Build candidates for the selected stochastic-cycle mode.

    "multi_peak_single_cycle" shares the single-cycle grid because peak selection
    happens before this function is called.
    """
    if stochastic_cycle_mode in ("single", "multi_peak_single_cycle"):
        return build_single_cycle_candidate_grid(r_grid, d_grid)
    if stochastic_cycle_mode == "multi_cycle":
        return build_multi_cycle_candidate_grid(**kwargs)
    raise InvalidConfigurationError(
        f"Unknown stochastic_cycle_mode: {stochastic_cycle_mode!r}. "
        f"Must be one of 'single', 'multi_peak_single_cycle', 'multi_cycle'."
    )


# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact 
# reason why the process failed.
# ---------------------------------------------------------------------------


def _validate_r_grid(r_star: int, r_window: int, T: int) -> None:
    if isinstance(r_star, bool) or not isinstance(r_star, int):
        raise InvalidConfigurationError(
            f"r_star must be an int, got {type(r_star).__name__}."
        )
    if isinstance(r_window, bool) or not isinstance(r_window, int):
        raise InvalidConfigurationError(
            f"r_window must be an int, got {type(r_window).__name__}."
        )
    if isinstance(T, bool) or not isinstance(T, int):
        raise InvalidConfigurationError(f"T must be an int, got {type(T).__name__}.")
    if T < 2:
        raise InvalidConfigurationError(f"T must be >= 2, got {T}.")
    if r_window < 0:
        raise InvalidConfigurationError(f"r_window must be >= 0, got {r_window}.")
    if r_star < 1 or r_star > T - 1:
        raise InvalidConfigurationError(
            f"r_star must satisfy 1 <= r_star <= T-1={T - 1}, got r_star={r_star}."
        )
