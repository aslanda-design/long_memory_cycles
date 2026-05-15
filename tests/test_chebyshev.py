import numpy as np
import pytest

from cyclical_fractional_test import InvalidConfigurationError
from cyclical_fractional_test.chebyshev import (
    build_chebyshev_design,
    build_single_chebyshev_polynomial,
)


# ---------------------------------------------------------------------------
# build_single_chebyshev_polynomial
# ---------------------------------------------------------------------------


def test_build_single_chebyshev_polynomial_order_zero_returns_ones():
    result = build_single_chebyshev_polynomial(T=5, order=0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (5,)
    np.testing.assert_allclose(result, np.ones(5))


def test_build_single_chebyshev_polynomial_order_one_matches_formula():
    T = 4
    t = np.arange(1, T + 1, dtype=float)
    expected = 2.0 * np.cos(1 * np.pi * (t - 0.5) / T)
    result = build_single_chebyshev_polynomial(T=T, order=1)
    np.testing.assert_allclose(result, expected)


def test_build_single_chebyshev_polynomial_order_two_matches_formula():
    T = 4
    t = np.arange(1, T + 1, dtype=float)
    expected = 2.0 * np.cos(2 * np.pi * (t - 0.5) / T)
    result = build_single_chebyshev_polynomial(T=T, order=2)
    np.testing.assert_allclose(result, expected)


def test_build_single_chebyshev_polynomial_larger_T():
    T = 100
    order = 3
    t = np.arange(1, T + 1, dtype=float)
    expected = 2.0 * np.cos(order * np.pi * (t - 0.5) / T)
    result = build_single_chebyshev_polynomial(T=T, order=order)
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# build_chebyshev_design — shape
# ---------------------------------------------------------------------------


def test_build_chebyshev_design_without_intercept_has_expected_shape():
    result = build_chebyshev_design(T=10, n_cycles=4, include_intercept=False)
    assert result.shape == (10, 4)


def test_build_chebyshev_design_with_intercept_has_expected_shape():
    result = build_chebyshev_design(T=10, n_cycles=4, include_intercept=True)
    assert result.shape == (10, 5)
    np.testing.assert_allclose(result[:, 0], np.ones(10))


# ---------------------------------------------------------------------------
# build_chebyshev_design — column correctness
# ---------------------------------------------------------------------------


def test_build_chebyshev_design_columns_match_single_polynomials():
    T, n_cycles = 8, 3
    X = build_chebyshev_design(T=T, n_cycles=n_cycles, include_intercept=False)
    for col_idx, k in enumerate(range(1, n_cycles + 1)):
        expected = build_single_chebyshev_polynomial(T=T, order=k)
        np.testing.assert_allclose(X[:, col_idx], expected)


def test_build_chebyshev_design_with_intercept_columns_match():
    T, n_cycles = 8, 3
    X = build_chebyshev_design(T=T, n_cycles=n_cycles, include_intercept=True)
    np.testing.assert_allclose(X[:, 0], np.ones(T))
    for col_idx, k in enumerate(range(1, n_cycles + 1), start=1):
        expected = build_single_chebyshev_polynomial(T=T, order=k)
        np.testing.assert_allclose(X[:, col_idx], expected)


def test_build_chebyshev_design_n_cycles_1_no_intercept():
    result = build_chebyshev_design(T=5, n_cycles=1, include_intercept=False)
    assert result.shape == (5, 1)
    np.testing.assert_allclose(
        result[:, 0], build_single_chebyshev_polynomial(T=5, order=1)
    )


# ---------------------------------------------------------------------------
# Error cases — build_single_chebyshev_polynomial
# ---------------------------------------------------------------------------


def test_build_single_chebyshev_polynomial_rejects_zero_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=0, order=1)


def test_build_single_chebyshev_polynomial_rejects_negative_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=-3, order=1)


def test_build_single_chebyshev_polynomial_rejects_float_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=5.0, order=1)  # type: ignore


def test_build_single_chebyshev_polynomial_rejects_bool_T():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=True, order=1)  # type: ignore


def test_build_single_chebyshev_polynomial_rejects_negative_order():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=5, order=-1)


def test_build_single_chebyshev_polynomial_rejects_float_order():
    with pytest.raises(InvalidConfigurationError):
        build_single_chebyshev_polynomial(T=5, order=1.5)  # type: ignore


# ---------------------------------------------------------------------------
# Error cases — build_chebyshev_design
# ---------------------------------------------------------------------------


def test_build_chebyshev_design_rejects_non_positive_T():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=-1, n_cycles=4)


def test_build_chebyshev_design_rejects_float_T():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10.0, n_cycles=4)  # type: ignore


def test_build_chebyshev_design_rejects_zero_n_cycles():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=0)


def test_build_chebyshev_design_rejects_negative_n_cycles():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=-2)


def test_build_chebyshev_design_rejects_float_n_cycles():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=2.5)  # type: ignore


def test_build_chebyshev_design_rejects_non_boolean_intercept():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=4, include_intercept=1)  # type: ignore


def test_build_chebyshev_design_rejects_string_intercept():
    with pytest.raises(InvalidConfigurationError):
        build_chebyshev_design(T=10, n_cycles=4, include_intercept="yes")  # type: ignore
