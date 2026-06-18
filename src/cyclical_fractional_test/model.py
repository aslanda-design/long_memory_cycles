from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

import numpy as np

from .api import run_cyclical_fractional_test
from .chebyshev import build_chebyshev_design, build_chebyshev_design_at
from .config import CyclicalTestConfig
from .exceptions import InvalidConfigurationError, NotFittedError
from .prediction import (
    compute_ma_weights,
    forecast_out_of_sample,
    reconstruct_in_sample,
)
from .regression import estimate_innovation_variance
from .results import CyclicalFractionalTestResult, StochasticCycle
from .validation import validate_series

_Z_FOR_TWO_SIDED = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}


class CyclicalFractionalModel:
    """Scikit-learn-style estimator for fractional cyclic long memory.

    `fit` runs the cyclical fractional test on a series and stores the selected
    model (best (R, D), OLS coefficients, AR error coefficients, residuals).
    `predict(n)` returns the reconstructed series for t = 1, ..., n: in-sample
    one-step-ahead values within the observed range, and out-of-sample forecasts
    beyond the training length. Hyperparameters mirror CyclicalTestConfig and can
    be passed directly or via **kwargs (overriding the config).
    """

    def __init__(
        self,
        config: Optional[CyclicalTestConfig] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = CyclicalTestConfig()
        if kwargs:
            config = dataclasses.replace(config, **kwargs)
        self.config = config
        self.threshold = threshold

    def fit(self, y: Any) -> "CyclicalFractionalModel":
        """Run the test on y and store the selected model. Returns self."""
        arr = validate_series(y)
        result = run_cyclical_fractional_test(
            arr, config=self.config, threshold=self.threshold
        )
        best = result.best_result
        if best is None or best.betas is None or best.residuals is None:
            raise InvalidConfigurationError(
                "fit did not produce a usable candidate; check the configuration."
            )

        T = len(arr)
        self.result_ = result
        self.y_train_ = arr
        self.n_train_ = T
        self.X_train_ = build_chebyshev_design(
            T, self.config.n_deterministic_cycles, self.config.include_intercept
        )
        self.cycles_ = best.cycles
        self.R_ = best.cycles[0].R
        self.D_ = best.cycles[0].D
        self.betas_ = np.asarray(best.betas, dtype=float)
        self.residuals_ = np.asarray(best.residuals, dtype=float)
        self.error_model_ = best.error_model
        self.ar_coefficients_ = np.asarray(best.ar_coefficients, dtype=float)
        self.innovation_variance_ = estimate_innovation_variance(
            self.residuals_, self.ar_coefficients_
        )
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return the model's reconstruction of the series for t = 1, ..., n.

        For n <= T these are in-sample one-step-ahead values; for n > T the tail
        T+1, ..., n is forecast out of sample.
        """
        self._check_fitted()
        _validate_n(n)
        mode = self.config.stochastic_cycle_mode

        in_sample = reconstruct_in_sample(
            self.y_train_,
            self.X_train_,
            self.cycles_,
            self.betas_,
            self.residuals_,
            self.ar_coefficients_,
            mode,
        )
        if n <= self.n_train_:
            return in_sample[:n]

        horizon = n - self.n_train_
        X_future = self._future_design(horizon)
        forecast = forecast_out_of_sample(
            self.y_train_,
            self.X_train_,
            X_future,
            self.cycles_,
            self.betas_,
            self.residuals_,
            self.ar_coefficients_,
            mode,
            horizon,
        )
        return np.concatenate([in_sample, forecast])

    def predict_interval(
        self, n: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction bounds for predict(n) at level 1 − alpha.

        In-sample bounds use the one-step innovation standard deviation σ̂. Forecast
        bounds widen with the accumulated MA(∞) weights of the inverse cyclic filter
        and AR error: the horizon-k variance is σ̂² Σ_{l<k} ψ_l². Parameter-estimation
        uncertainty is not included.
        """
        self._check_fitted()
        _validate_n(n)
        z = _z_score(alpha)
        center = self.predict(n)
        sigma = float(np.sqrt(self.innovation_variance_))

        std = np.full(n, sigma, dtype=float)
        if n > self.n_train_:
            horizon = n - self.n_train_
            weights = compute_ma_weights(
                self.cycles_,
                self.ar_coefficients_,
                self.config.stochastic_cycle_mode,
                self.n_train_,
                horizon,
            )
            cumulative = np.sqrt(np.cumsum(weights ** 2))
            std[self.n_train_ :] = sigma * cumulative
        return center - z * std, center + z * std

    def _future_design(self, horizon: int) -> np.ndarray:
        t_future = np.arange(self.n_train_ + 1, self.n_train_ + horizon + 1, dtype=float)
        return build_chebyshev_design_at(
            t_future,
            self.n_train_,
            self.config.n_deterministic_cycles,
            self.config.include_intercept,
        )

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise NotFittedError(
                "This CyclicalFractionalModel is not fitted yet. Call fit before predict."
            )


def _validate_n(n: Any) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise InvalidConfigurationError(f"n must be an int, got {type(n).__name__}.")
    if int(n) < 1:
        raise InvalidConfigurationError(f"n must be >= 1, got {n}.")


def _z_score(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)):
        raise InvalidConfigurationError(
            f"alpha must be a float, got {type(alpha).__name__}."
        )
    alpha = float(alpha)
    if alpha not in _Z_FOR_TWO_SIDED:
        raise InvalidConfigurationError(
            f"alpha must be one of {sorted(_Z_FOR_TWO_SIDED)}, got {alpha}."
        )
    return _Z_FOR_TWO_SIDED[alpha]
