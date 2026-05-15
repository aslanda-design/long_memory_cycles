"""Integration test: partial pipeline Waves 7–9."""
import numpy as np

from cyclical_fractional_test.filters import filter_response_and_design
from cyclical_fractional_test.regression import (
    compute_time_variance,
    fit_filtered_regression,
)
from cyclical_fractional_test.spectral import (
    compute_frequency_variance_dynamic,
    compute_residual_periodogram,
)
from cyclical_fractional_test.results import StochasticCycle


def test_partial_pipeline_single_cycle_runs_end_to_end():
    """D=0 → filter is identity; verify the full Waves 7–9 pipeline connects."""
    rng = np.random.default_rng(42)
    T = 20
    y = rng.standard_normal(T)
    X = rng.standard_normal((T, 2))
    cycles = [StochasticCycle(R=2, D=0.0)]

    # Wave 7 — filter
    y_f, X_f = filter_response_and_design(y, X, cycles, mode="single")

    # D=0 → identity filter
    np.testing.assert_allclose(y_f, y, atol=1e-12)
    np.testing.assert_allclose(X_f, X, atol=1e-12)

    # Wave 8 — regression
    reg = fit_filtered_regression(y_f, X_f)
    assert reg.residuals.shape == (T,)
    assert np.all(np.isfinite(reg.residuals))

    # Wave 9 — periodogram and variances
    lambdas, I_resid = compute_residual_periodogram(reg.residuals)
    assert len(lambdas) == T
    assert len(I_resid) == T
    assert np.all(np.isfinite(I_resid))

    var_time = compute_time_variance(reg.residuals)
    assert np.isfinite(var_time)
    assert var_time >= 0.0

    var_freq = compute_frequency_variance_dynamic(
        I_resid, cycles, mode="single", drop_frequency=True
    )
    assert np.isfinite(var_freq)
    assert var_freq >= 0.0


def test_partial_pipeline_non_trivial_D():
    """Run with D=0.4 to verify no numerical errors across the full chain."""
    rng = np.random.default_rng(7)
    T = 30
    y = rng.standard_normal(T)
    X = rng.standard_normal((T, 3))
    cycles = [StochasticCycle(R=4, D=0.4)]

    y_f, X_f = filter_response_and_design(y, X, cycles, mode="single")
    reg = fit_filtered_regression(y_f, X_f)
    lambdas, I_resid = compute_residual_periodogram(reg.residuals)
    var_time = compute_time_variance(reg.residuals)
    var_freq = compute_frequency_variance_dynamic(I_resid, cycles, mode="single")

    assert np.all(np.isfinite(y_f))
    assert np.all(np.isfinite(X_f))
    assert np.isfinite(var_time)
    assert np.isfinite(var_freq)
    assert var_time >= 0.0
    assert var_freq >= 0.0
