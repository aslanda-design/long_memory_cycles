"""Tests for Wave 7: apply_fractional_filter_single_series and apply_single_cycle_filter."""
import numpy as np
import pytest

from cyclical_fractional_test.filters import (
    apply_fractional_filter_single_series,
    apply_single_cycle_filter,
    compute_mu,
)
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.exceptions import InvalidConfigurationError


# ---------------------------------------------------------------------------
# apply_fractional_filter_single_series
# ---------------------------------------------------------------------------


def test_identity_filter_returns_original():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    coeffs = np.array([1.0, 0.0, 0.0, 0.0])
    out = apply_fractional_filter_single_series(x, coeffs)
    np.testing.assert_allclose(out, x)


def test_known_coefficients_orientation():
    """Verify causal orientation: out[t] = sum_{j=0}^{t} C_j x[t-j]."""
    x = np.array([1.0, 2.0, 3.0])
    coeffs = np.array([1.0, 10.0, 100.0])
    out = apply_fractional_filter_single_series(x, coeffs)
    # out[0] = 1*1 = 1
    # out[1] = 1*2 + 10*1 = 12
    # out[2] = 1*3 + 10*2 + 100*1 = 123
    np.testing.assert_allclose(out, [1.0, 12.0, 123.0])


def test_output_length_equals_input_length():
    x = np.ones(7)
    coeffs = np.ones(7)
    out = apply_fractional_filter_single_series(x, coeffs)
    assert len(out) == len(x)


def test_output_dtype_is_float():
    x = np.array([1, 2, 3], dtype=int)
    coeffs = np.array([1, 0, 0], dtype=int)
    out = apply_fractional_filter_single_series(x, coeffs)
    assert np.issubdtype(out.dtype, np.floating)


def test_coefficients_shorter_than_x_raises():
    x = np.ones(5)
    coeffs = np.ones(3)
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_fractional_filter_single_series(x, coeffs)


def test_empty_x_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_fractional_filter_single_series(np.array([]), np.array([1.0]))


def test_2d_x_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_fractional_filter_single_series(np.ones((3, 2)), np.ones(6))


# ---------------------------------------------------------------------------
# apply_single_cycle_filter — D = 0 (identity)
# ---------------------------------------------------------------------------


def test_single_cycle_D_zero_returns_original():
    """(1 - 2mu L + L^2)^0 = 1 → filter is identity."""
    T = 5
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    cycle = StochasticCycle(R=2, D=0.0)
    out = apply_single_cycle_filter(x, cycle, T)
    np.testing.assert_allclose(out, x)


# ---------------------------------------------------------------------------
# apply_single_cycle_filter — D = 1 (first-order polynomial filter)
# ---------------------------------------------------------------------------


def test_single_cycle_D_one_matches_second_order_filter():
    """(1 - 2mu L + L^2)^1: out[t] = x[t] - 2mu x[t-1] + x[t-2] for t >= 2."""
    T = 6
    R = 2
    D = 1.0
    mu = compute_mu(T, R)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = apply_single_cycle_filter(x, StochasticCycle(R=R, D=D), T)

    expected = np.zeros(T)
    expected[0] = x[0]
    expected[1] = x[1] - 2.0 * mu * x[0]
    for t in range(2, T):
        expected[t] = x[t] - 2.0 * mu * x[t - 1] + x[t - 2]

    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_single_cycle_output_shape():
    T = 10
    x = np.random.default_rng(0).standard_normal(T)
    out = apply_single_cycle_filter(x, StochasticCycle(R=3, D=0.4), T)
    assert out.shape == (T,)


def test_single_cycle_wrong_length_raises():
    x = np.ones(8)
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_single_cycle_filter(x, StochasticCycle(R=2, D=0.5), T=10)
