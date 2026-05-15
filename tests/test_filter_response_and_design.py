"""Tests for Wave 7: filter_response_and_design."""
import numpy as np
import pytest

from cyclical_fractional_test.filters import filter_response_and_design
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.exceptions import InvalidConfigurationError


def _make_inputs(T=10, p=3, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(T)
    X = rng.standard_normal((T, p))
    cycles = [StochasticCycle(R=2, D=0.4)]
    return y, X, cycles


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


def test_output_shapes():
    T, p = 10, 3
    y, X, cycles = _make_inputs(T=T, p=p)
    y_f, X_f = filter_response_and_design(y, X, cycles, mode="single")
    assert y_f.shape == (T,)
    assert X_f.shape == (T, p)


def test_single_column_X():
    T = 8
    y = np.ones(T)
    X = np.ones((T, 1))
    cycles = [StochasticCycle(R=2, D=0.3)]
    y_f, X_f = filter_response_and_design(y, X, cycles)
    assert y_f.shape == (T,)
    assert X_f.shape == (T, 1)


# ---------------------------------------------------------------------------
# Does not modify inputs
# ---------------------------------------------------------------------------


def test_does_not_modify_y():
    y, X, cycles = _make_inputs()
    y_orig = y.copy()
    filter_response_and_design(y, X, cycles)
    np.testing.assert_array_equal(y, y_orig)


def test_does_not_modify_X():
    y, X, cycles = _make_inputs()
    X_orig = X.copy()
    filter_response_and_design(y, X, cycles)
    np.testing.assert_array_equal(X, X_orig)


# ---------------------------------------------------------------------------
# D = 0 → identity: filtered == original
# ---------------------------------------------------------------------------


def test_D_zero_y_unchanged():
    T = 8
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    X = np.ones((T, 2))
    cycles = [StochasticCycle(R=2, D=0.0)]
    y_f, _ = filter_response_and_design(y, X, cycles)
    np.testing.assert_allclose(y_f, y)


def test_D_zero_X_unchanged():
    T = 8
    y = np.ones(T)
    X = np.tile(np.arange(T, dtype=float), (2, 1)).T
    cycles = [StochasticCycle(R=2, D=0.0)]
    _, X_f = filter_response_and_design(y, X, cycles)
    np.testing.assert_allclose(X_f, X)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_y_2d_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        filter_response_and_design(np.ones((5, 1)), np.ones((5, 2)), [StochasticCycle(R=2, D=0.3)])


def test_X_1d_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        filter_response_and_design(np.ones(5), np.ones(5), [StochasticCycle(R=2, D=0.3)])


def test_mismatched_lengths_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        filter_response_and_design(
            np.ones(5), np.ones((7, 2)), [StochasticCycle(R=2, D=0.3)]
        )
