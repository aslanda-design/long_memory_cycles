import numpy as np
import pytest

from cyclical_fractional_test.exceptions import InvalidConfigurationError
from cyclical_fractional_test.regression import (
    RegressionResult,
    compute_residual_sum_squares,
    compute_residuals,
    compute_time_variance,
    estimate_ar_ols,
    fit_filtered_regression,
)


# ---------------------------------------------------------------------------
# compute_residuals and compute_residual_sum_squares
# ---------------------------------------------------------------------------


def test_compute_residuals_simple():
    y = np.array([1.0, 2.0, 3.0])
    fitted = np.array([1.0, 1.5, 4.0])
    np.testing.assert_allclose(compute_residuals(y, fitted), [0.0, 0.5, -1.0])


def test_compute_residuals_shape_mismatch_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_residuals(np.ones(3), np.ones(4))


def test_compute_residual_sum_squares_known():
    r = np.array([1.0, -2.0, 3.0])
    assert np.isclose(compute_residual_sum_squares(r), 14.0)


def test_compute_residual_sum_squares_zero():
    assert compute_residual_sum_squares(np.zeros(5)) == 0.0


# ---------------------------------------------------------------------------
# fit_filtered_regression
# ---------------------------------------------------------------------------


def test_fit_recovers_simple_beta():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    beta_true = np.array([2.0])
    y = X @ beta_true
    result = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, beta_true, atol=1e-10)
    np.testing.assert_allclose(result.residuals, np.zeros(4), atol=1e-10)


def test_fit_recovers_multiple_betas():
    rng = np.random.default_rng(99)
    T, p = 20, 3
    X = rng.standard_normal((T, p))
    beta_true = np.array([2.0, -1.0, 0.5])
    y = X @ beta_true
    result = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, beta_true, atol=1e-8)


def test_fit_output_shapes():
    T, p = 15, 4
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, p))
    y = rng.standard_normal(T)
    result = fit_filtered_regression(y, X)
    assert result.betas.shape == (p,)
    assert result.fitted_values.shape == (T,)
    assert result.residuals.shape == (T,)


def test_fit_returns_regression_result():
    result = fit_filtered_regression(np.ones(5), np.ones((5, 1)))
    assert isinstance(result, RegressionResult)


def test_fit_rank_and_condition_number_are_set():
    result = fit_filtered_regression(np.ones(4), np.eye(4))
    assert isinstance(result.rank, int)
    assert np.isfinite(result.condition_number)


def test_fit_does_not_modify_inputs():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(10)
    X = rng.standard_normal((10, 2))
    y_orig, X_orig = y.copy(), X.copy()
    fit_filtered_regression(y, X)
    np.testing.assert_array_equal(y, y_orig)
    np.testing.assert_array_equal(X, X_orig)


@pytest.mark.parametrize("bad_y, bad_X", [
    (np.ones((5, 1)), np.ones((5, 2))),
    (np.ones(5), np.ones(5)),
    (np.ones(5), np.ones((7, 2))),
])
def test_fit_rejects_invalid_shapes(bad_y, bad_X):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(bad_y, bad_X)


def test_fit_rejects_nan_in_inputs():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.array([1.0, float("nan"), 3.0]), np.ones((3, 1)))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.ones(3), np.array([[1.0], [float("nan")], [3.0]]))


# ---------------------------------------------------------------------------
# compute_time_variance
# ---------------------------------------------------------------------------


def test_time_variance_is_mean_of_squares():
    r = np.array([1.0, -2.0, 3.0])
    assert np.isclose(compute_time_variance(r), 14.0 / 3.0)


def test_time_variance_not_centered():
    r = np.array([1.0, 2.0, 3.0])
    assert np.isclose(compute_time_variance(r), np.mean(r ** 2))
    assert not np.isclose(compute_time_variance(r), np.var(r))


def test_time_variance_zero_residuals():
    assert compute_time_variance(np.zeros(5)) == 0.0


def test_time_variance_rejects_empty_or_2d():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_time_variance(np.array([]))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_time_variance(np.ones((3, 2)))


# ---------------------------------------------------------------------------
# estimate_ar_ols
# ---------------------------------------------------------------------------


def test_estimate_ar_ols_order_zero_returns_empty_array():
    result = estimate_ar_ols(np.array([1.0, 2.0, 3.0]), order=0)
    assert result.shape == (0,)


def test_estimate_ar_ols_recovers_ar1_coefficient():
    residuals = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(estimate_ar_ols(residuals, order=1), [0.5])


def test_estimate_ar_ols_recovers_ar2_coefficients():
    phi_1, phi_2 = 0.5, -0.2
    residuals = [1.0, 0.25]
    for _ in range(12):
        residuals.append(phi_1 * residuals[-1] + phi_2 * residuals[-2])
    np.testing.assert_allclose(
        estimate_ar_ols(np.array(residuals), order=2),
        [phi_1, phi_2],
        atol=1e-12,
    )


@pytest.mark.parametrize("order", [-1, 3, True])
def test_estimate_ar_ols_rejects_unsupported_order(order):
    with pytest.raises(InvalidConfigurationError):
        estimate_ar_ols(np.ones(5), order=order)


def test_estimate_ar_ols_rejects_insufficient_residuals():
    with pytest.raises(InvalidConfigurationError):
        estimate_ar_ols(np.ones(2), order=2)
