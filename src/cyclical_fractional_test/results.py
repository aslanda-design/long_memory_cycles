from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CyclicalTestConfig


@dataclass
class StochasticCycle:
    """Candidate stochastic cycle used by the test.

    It represents the factor (1 - 2cos(2πR/T)L + L²)^D applied to the series.
    """

    R: int  # Candidate index for the cyclic frequency.
    D: float  # Fractional integration parameter for this cycle.


@dataclass
class GridCandidateResult:
    """Values obtained when evaluating one candidate from the grid.

    Most entries are optional while the numerical core is still being filled in.
    """

    cycles: Tuple[StochasticCycle, ...]  # Cycle tuple represented by this grid point.
    test_value: Optional[float] = None  # TEST statistic for the candidate.
    test_star_value: Optional[float] = None  # TEST* statistic for the candidate.
    abs_test_value: Optional[float] = None  # Absolute value used for TEST ranking.
    abs_test_star_value: Optional[float] = None  # Absolute value for TEST* ranking.
    xa: Optional[float] = None  # XA(R,D) scalar used in the statistic.
    xaa: Optional[float] = None  # XAA(R) scalar used in the statistic.
    variance_time: Optional[float] = None  # Time-domain variance estimate.
    variance_frequency: Optional[float] = None  # Frequency-domain variance estimate.
    betas: Optional[np.ndarray] = None  # Estimated deterministic-cycle coefficients.
    residuals: Optional[np.ndarray] = None  # Regression residuals for this candidate.
    residual_sum_squares: Optional[float] = None  # Sum of squared residuals.


@dataclass
class CyclicalFractionalTestResult:
    """Container returned by the cyclical fractional long-memory test."""

    best_result: Optional[GridCandidateResult] = None  # Best candidate found.
    top_k_results: List[GridCandidateResult] = field(default_factory=list)  # Retained top-k candidates.
    r_star: Optional[int] = None  # Main periodogram peak used to build the R grid.
    r_candidates: Optional[np.ndarray] = None  # R values considered around r_star.
    d_grid: Optional[np.ndarray] = None  # D values evaluated in the grid.
    config: Optional[CyclicalTestConfig] = None  # Configuration used in the run.
    n_candidates_evaluated: Optional[int] = None  # Total grid points evaluated.
    diagnostics: Optional[Any] = None  # TestDiagnostics; populated by run_cyclical_fractional_test.
