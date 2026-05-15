"""Tests for Wave 8: compute_time_variance."""
import numpy as np
import pytest

from cyclical_fractional_test.regression import compute_time_variance
from cyclical_fractional_test.exceptions import InvalidConfigurationError


def test_known_value():
    # VAR = (1 + 4 + 9) / 3 = 14/3
    r = np.array([1.0, -2.0, 3.0])
    assert np.isclose(compute_time_variance(r), 14.0 / 3.0)


def test_zero_residuals():
    assert compute_time_variance(np.zeros(5)) == 0.0


def test_is_mean_of_squares_not_centered_variance():
    r = np.array([1.0, 2.0, 3.0])
    # mean(r^2) = (1 + 4 + 9) / 3 = 14/3
    # np.var(r) = mean(r^2) - mean(r)^2 = 14/3 - 4 = 2/3
    assert not np.isclose(compute_time_variance(r), np.var(r))
    assert np.isclose(compute_time_variance(r), np.mean(r ** 2))


def test_returns_float():
    assert isinstance(compute_time_variance(np.ones(4)), float)


def test_empty_array_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_time_variance(np.array([]))


def test_2d_array_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_time_variance(np.ones((3, 2)))
