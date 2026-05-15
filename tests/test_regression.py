"""Tests for Wave 8: fit_filtered_regression, compute_residuals, compute_residual_sum_squares."""
import numpy as np
import pytest

from cyclical_fractional_test.regression import (
    RegressionResult,
    compute_residual_sum_squares,
    compute_residuals,
    fit_filtered_regression,
)
from cyclical_fractional_test.exceptions import InvalidConfigurationError


# ---------------------------------------------------------------------------
# compute_residuals
# ---------------------------------------------------------------------------


def test_compute_residuals_simple():
    y = np.array([1.0, 2.0, 3.0])
    fitted = np.array([1.0, 1.5, 4.0])
    r = compute_residuals(y, fitted)
    np.testing.assert_allclose(r, [0.0, 0.5, -1.0])


def test_compute_residuals_shape_mismatch_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_residuals(np.ones(3), np.ones(4))


def test_compute_residuals_output_is_ndarray():
    r = compute_residuals(np.array([1.0, 2.0]), np.array([0.5, 1.5]))
    assert isinstance(r, np.ndarray)


# ---------------------------------------------------------------------------
# compute_residual_sum_squares
# ---------------------------------------------------------------------------


def test_compute_residual_sum_squares_known():
    r = np.array([1.0, -2.0, 3.0])
    assert np.isclose(compute_residual_sum_squares(r), 14.0)


def test_compute_residual_sum_squares_zero():
    assert compute_residual_sum_squares(np.zeros(5)) == 0.0


def test_compute_residual_sum_squares_returns_float():
    result = compute_residual_sum_squares(np.array([1.0, 2.0]))
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# fit_filtered_regression — simple single-regressor case
# ---------------------------------------------------------------------------


def test_fit_recovers_simple_beta():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    beta_true = np.array([2.0])
    y = X @ beta_true
    result = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, beta_true, atol=1e-10)
    np.testing.assert_allclose(result.residuals, np.zeros(4), atol=1e-10)
    assert np.isclose(result.residual_sum_squares, 0.0, atol=1e-10)


def test_fit_recovers_multiple_betas():
    rng = np.random.default_rng(99)
    T, p = 20, 3
    X = rng.standard_normal((T, p))
    beta_true = np.array([2.0, -1.0, 0.5])
    y = X @ beta_true
    result = fit_filtered_regression(y, X)
    np.testing.assert_allclose(result.betas, beta_true, atol=1e-8)
    np.testing.assert_allclose(result.residuals, np.zeros(T), atol=1e-8)


# ---------------------------------------------------------------------------
# fit_filtered_regression — output shapes
# ---------------------------------------------------------------------------


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
    X = np.ones((5, 1))
    y = np.ones(5)
    result = fit_filtered_regression(y, X)
    assert isinstance(result, RegressionResult)


def test_fit_rank_and_condition_number_set():
    X = np.eye(4)
    y = np.ones(4)
    result = fit_filtered_regression(y, X)
    assert isinstance(result.rank, int)
    assert isinstance(result.condition_number, float)
    assert np.isfinite(result.condition_number)


# ---------------------------------------------------------------------------
# fit_filtered_regression — validation
# ---------------------------------------------------------------------------


def test_fit_y_2d_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.ones((5, 1)), np.ones((5, 2)))


def test_fit_X_1d_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.ones(5), np.ones(5))


def test_fit_length_mismatch_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(np.ones(5), np.ones((7, 2)))


def test_fit_nan_in_y_raises():
    y = np.array([1.0, float("nan"), 3.0])
    X = np.ones((3, 1))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(y, X)


def test_fit_nan_in_X_raises():
    y = np.ones(3)
    X = np.array([[1.0], [float("nan")], [3.0]])
    with pytest.raises((InvalidConfigurationError, ValueError)):
        fit_filtered_regression(y, X)


# ---------------------------------------------------------------------------
# fit_filtered_regression — does not modify inputs
# ---------------------------------------------------------------------------


def test_fit_does_not_modify_y():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(10)
    X = rng.standard_normal((10, 2))
    y_copy = y.copy()
    fit_filtered_regression(y, X)
    np.testing.assert_array_equal(y, y_copy)


def test_fit_does_not_modify_X():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(10)
    X = rng.standard_normal((10, 2))
    X_copy = X.copy()
    fit_filtered_regression(y, X)
    np.testing.assert_array_equal(X, X_copy)
