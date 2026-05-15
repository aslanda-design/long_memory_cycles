from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CyclicalTestConfig:
    """Configuration shared by the pieces of the cyclical test."""

    n_deterministic_cycles: int = 4  # Number m of Chebyshev terms P_1(t), ..., P_m(t).
    include_intercept: bool = False  # Include P_0(t)=1 in the regression.
    d_grid: Optional[np.ndarray] = field(default=None)  # D grid; None means default.
    r_window: int = 10  # Half-window around the periodogram peak R*.
    top_k: int = 1  # Number of best candidates to keep.
    variance_mode: str = "time"  # Variance estimator: "time", "frequency", or "both".
    statistic_mode: str = "test"  # Statistic formula: "test" or "test_star".
    stochastic_cycle_mode: str = "single"  # Cycle search mode used by the grid.
    drop_singular_frequency: bool = True  # Drop j=R where psi is singular.
    exclude_zero_frequency: bool = True  # Ignore zero frequency when locating R*.
    return_residuals_for_top_k: bool = True  # Store residuals for retained candidates.
