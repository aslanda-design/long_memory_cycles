"""Tests for Wave 9: compute_residual_periodogram."""
import numpy as np
import pytest

from cyclical_fractional_test.spectral import (
    compute_document_periodogram,
    compute_residual_periodogram,
)
from cyclical_fractional_test.exceptions import InvalidConfigurationError


def test_matches_document_periodogram():
    rng = np.random.default_rng(0)
    residuals = rng.standard_normal(20)
    lambdas1, I1 = compute_residual_periodogram(residuals)
    lambdas2, I2 = compute_document_periodogram(residuals)
    np.testing.assert_array_equal(lambdas1, lambdas2)
    np.testing.assert_array_equal(I1, I2)


def test_output_shapes():
    T = 16
    residuals = np.random.default_rng(1).standard_normal(T)
    lambdas, I_res = compute_residual_periodogram(residuals)
    assert len(lambdas) == T
    assert len(I_res) == T


def test_output_non_negative_periodogram():
    residuals = np.random.default_rng(2).standard_normal(10)
    _, I_res = compute_residual_periodogram(residuals)
    assert np.all(I_res >= 0.0)


def test_all_zeros_residuals():
    residuals = np.zeros(8)
    _, I_res = compute_residual_periodogram(residuals)
    np.testing.assert_allclose(I_res, 0.0)


def test_non_finite_residuals_raises():
    residuals = np.array([1.0, float("nan"), 3.0, 4.0, 5.0])
    with pytest.raises((InvalidConfigurationError, ValueError, Exception)):
        compute_residual_periodogram(residuals)


def test_empty_residuals_raises():
    with pytest.raises((InvalidConfigurationError, ValueError, Exception)):
        compute_residual_periodogram(np.array([]))
