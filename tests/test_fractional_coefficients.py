"""Tests for Wave 6: compute_mu and compute_fractional_coefficients_single_cycle."""
import numpy as np
import pytest

from cyclical_fractional_test.filters import (
    compute_mu,
    compute_fractional_coefficients_single_cycle,
)
from cyclical_fractional_test.exceptions import InvalidConfigurationError


# ---------------------------------------------------------------------------
# compute_mu
# ---------------------------------------------------------------------------


def test_compute_mu_known_value():
    T, R = 10, 2
    result = compute_mu(T, R)
    expected = np.cos(2 * np.pi * R / T)
    assert np.isclose(result, expected)


def test_compute_mu_returns_float():
    result = compute_mu(10, 2)
    assert isinstance(result, float)


@pytest.mark.parametrize("R", [0, -1, -10])
def test_compute_mu_invalid_R_non_positive(R):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_mu(10, R)


@pytest.mark.parametrize("R", [10, 11, 100])
def test_compute_mu_invalid_R_too_large(R):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_mu(10, R)


@pytest.mark.parametrize("T", [0, 1, -1, -10])
def test_compute_mu_invalid_T(T):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_mu(T, 1)


def test_compute_mu_bool_T_rejected():
    with pytest.raises((InvalidConfigurationError, TypeError)):
        compute_mu(True, 1)


def test_compute_mu_bool_R_rejected():
    with pytest.raises((InvalidConfigurationError, TypeError)):
        compute_mu(10, True)


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle — D = 0
# ---------------------------------------------------------------------------


def test_d_zero_produces_identity_filter():
    """(1 - 2mu L + L^2)^0 = 1  →  C_0=1, C_j=0 for j>=1."""
    T, R, D = 8, 2, 0.0
    coeffs = compute_fractional_coefficients_single_cycle(T, R, D)
    expected = np.zeros(T)
    expected[0] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle — D = 1
# ---------------------------------------------------------------------------


def test_d_one_produces_first_order_filter():
    """(1 - 2mu L + L^2)^1 = 1 - 2mu L + L^2  →  C_0=1, C_1=-2mu, C_2=1, rest 0."""
    T, R, D = 8, 2, 1.0
    mu = compute_mu(T, R)
    coeffs = compute_fractional_coefficients_single_cycle(T, R, D)
    expected = np.zeros(T)
    expected[0] = 1.0
    expected[1] = -2.0 * mu
    expected[2] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)


def test_d_one_non_zero_mu():
    """Same D=1 verification for R where mu != 0."""
    T, R, D = 10, 1, 1.0
    mu = compute_mu(T, R)
    coeffs = compute_fractional_coefficients_single_cycle(T, R, D)
    expected = np.zeros(T)
    expected[0] = 1.0
    expected[1] = -2.0 * mu
    expected[2] = 1.0
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle — recurrence verification
# ---------------------------------------------------------------------------


def test_recurrence_holds_for_fractional_D():
    """Verify C_j = [2mu(j-1-D)C_{j-1} + (2D-j+2)C_{j-2}] / j for all j>=2."""
    T, R, D = 5, 1, 0.5
    mu = compute_mu(T, R)
    coeffs = compute_fractional_coefficients_single_cycle(T, R, D)

    assert np.isclose(coeffs[0], 1.0)
    assert np.isclose(coeffs[1], -2.0 * D * mu)

    for j in range(2, T):
        expected_j = (
            2.0 * mu * (j - 1 - D) * coeffs[j - 1]
            + (2.0 * D - j + 2) * coeffs[j - 2]
        ) / j
        assert np.isclose(coeffs[j], expected_j), (
            f"Recurrence failed at j={j}: got {coeffs[j]}, expected {expected_j}"
        )


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle — shape, dtype, finiteness
# ---------------------------------------------------------------------------


def test_output_length_matches_T():
    T, R, D = 20, 3, 0.4
    coeffs = compute_fractional_coefficients_single_cycle(T, R, D)
    assert len(coeffs) == T


def test_output_is_ndarray():
    coeffs = compute_fractional_coefficients_single_cycle(20, 3, 0.4)
    assert isinstance(coeffs, np.ndarray)


def test_output_dtype_is_float():
    coeffs = compute_fractional_coefficients_single_cycle(20, 3, 0.4)
    assert np.issubdtype(coeffs.dtype, np.floating)


def test_output_all_finite():
    coeffs = compute_fractional_coefficients_single_cycle(20, 3, 0.4)
    assert np.all(np.isfinite(coeffs))


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle — T = 1 raises error
# ---------------------------------------------------------------------------


def test_t_one_raises():
    """T=1 has no valid R (need 1 <= R < 1), so must raise."""
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(1, 1, 0.5)


# ---------------------------------------------------------------------------
# compute_fractional_coefficients_single_cycle — validation guards
# ---------------------------------------------------------------------------


def test_nan_D_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(10, 2, float("nan"))


def test_inf_D_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(10, 2, float("inf"))


def test_bool_D_raises():
    with pytest.raises((InvalidConfigurationError, TypeError)):
        compute_fractional_coefficients_single_cycle(10, 2, True)


def test_invalid_T_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(0, 2, 0.5)


def test_invalid_R_zero_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(10, 0, 0.5)


def test_invalid_R_equals_T_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_single_cycle(10, 10, 0.5)
