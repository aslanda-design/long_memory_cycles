from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .config import CyclicalTestConfig
from .results import CyclicalFractionalTestResult
from .validation import validate_config, validate_series


def run_cyclical_fractional_test(
    y: Any,
    config: Optional[CyclicalTestConfig] = None,
    **kwargs: Any,
) -> CyclicalFractionalTestResult:
    """Run the fractional cyclic long-memory test.

    For now this entry point validates the series and configuration, then raises
    NotImplementedError until the mathematical core is added.
    """
    if config is None:
        config = CyclicalTestConfig()

    # Keep validation in front so caller errors are reported before implementation gaps.
    arr = validate_series(y)
    validate_config(config)

    raise NotImplementedError(
        "The mathematical core of run_cyclical_fractional_test (Chebyshev design, "
        "periodogram, fractional filter, regression, XA/XAA, and TEST statistic) "
        "will be implemented in later waves."
    )
