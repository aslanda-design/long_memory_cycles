from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from .exceptions import InvalidConfigurationError
from .results import StochasticCycle
from .validation import validate_d_coarse_grid, validate_d_grid

# D grids are rounded to this many decimals to avoid noise like 0.30000000000000004.
_D_ROUND_DECIMALS = 12


def build_r_grid_around_peak(r_peak: int, r_window: int, T: int) -> np.ndarray:
    """Build the candidate R grid around R*: [max(1, R*−w), ..., min(T−1, R*+w)]."""
    _validate_r_grid(r_peak, r_window, T)
    r_min = max(1, r_peak - r_window)
    r_max = min(T - 1, r_peak + r_window)
    return np.arange(r_min, r_max + 1)


def build_d_grid(d_grid: Any = None) -> np.ndarray:
    """Build the candidate grid for D.

    With None, the default is [0.0, 0.1, ..., 1.0].
    User-supplied grids are validated and returned as float arrays.
    """
    if d_grid is None:
        return np.linspace(0.0, 1.0, 11)
    return validate_d_grid(d_grid)


def build_default_d_coarse_grid() -> np.ndarray:
    """Return the default coarse D grid [0.0, 0.1, ..., 1.0] with stable rounding."""
    return np.round(np.linspace(0.0, 1.0, 11), _D_ROUND_DECIMALS)


def build_d_fine_grid(
    center: float,
    radius: float,
    step: float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Build the local fine D grid around center, clipped to [lower, upper].

    For center=0.3, radius=0.09, step=0.01 this returns [0.21, ..., 0.39].
    Values out of range are clipped and de-duplicated, so boundary centers
    (0.0 or 1.0) yield a one-sided grid.
    """
    _validate_build_d_fine_grid(center, radius, step, lower, upper)
    raw = np.arange(center - radius, center + radius + step / 2.0, step)
    raw = np.round(raw, _D_ROUND_DECIMALS)
    clipped = np.round(np.clip(raw, lower, upper), _D_ROUND_DECIMALS)
    return np.unique(clipped)


def build_d_grid_for_strategy(config: Any) -> np.ndarray:
    """Return the D grid that seeds the search for the configured strategy.

    For "fixed_grid" this is the full evaluation grid (config.d_grid or default).
    For "adaptive" this is the coarse grid; local refinement happens per R later.
    """
    if config.d_search_strategy == "fixed_grid":
        return build_d_grid(config.d_grid)
    if config.d_coarse_grid is None:
        return build_default_d_coarse_grid()
    return np.round(validate_d_coarse_grid(config.d_coarse_grid), _D_ROUND_DECIMALS)


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


def _validate_build_d_fine_grid(
    center: float, radius: float, step: float, lower: float, upper: float
) -> None:
    for value, name in [
        (center, "center"),
        (radius, "radius"),
        (step, "step"),
        (lower, "lower"),
        (upper, "upper"),
    ]:
        if isinstance(value, bool):
            raise InvalidConfigurationError(f"{name} must be numeric, got bool.")
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigurationError(
                f"{name} must be numeric, got {type(value).__name__}."
            ) from exc
        if not np.isfinite(v):
            raise InvalidConfigurationError(f"{name} must be finite, got {value!r}.")
    if float(step) <= 0.0:
        raise InvalidConfigurationError(f"step must be > 0, got {step}.")
    if float(radius) <= 0.0:
        raise InvalidConfigurationError(f"radius must be > 0, got {radius}.")
    if float(lower) > float(upper):
        raise InvalidConfigurationError(
            f"lower must be <= upper, got lower={lower}, upper={upper}."
        )


def _validate_r_grid(r_peak: int, r_window: int, T: int) -> None:
    if isinstance(r_peak, bool) or not isinstance(r_peak, int):
        raise InvalidConfigurationError(
            f"r_peak must be an int, got {type(r_peak).__name__}."
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
    if r_peak < 1 or r_peak > T - 1:
        raise InvalidConfigurationError(
            f"r_peak must satisfy 1 <= r_peak <= T-1={T - 1}, got r_peak={r_peak}."
        )
