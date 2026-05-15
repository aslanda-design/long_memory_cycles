"""Tests for Wave 9: compute_frequency_variance_single_cycle and _multi_cycle."""
import numpy as np
import pytest

from cyclical_fractional_test.spectral import (
    compute_frequency_variance_multi_cycle,
    compute_frequency_variance_single_cycle,
)
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.exceptions import InvalidConfigurationError


_I = np.array([1.0, 2.0, 3.0, 4.0])  # T = 4


# ---------------------------------------------------------------------------
# compute_frequency_variance_single_cycle — without dropping
# ---------------------------------------------------------------------------


def test_no_drop_sums_all_values():
    T = len(_I)
    expected = (2.0 * np.pi / T) * np.sum(_I)
    result = compute_frequency_variance_single_cycle(_I, R=2, drop_frequency=False)
    assert np.isclose(result, expected)


# ---------------------------------------------------------------------------
# compute_frequency_variance_single_cycle — drop R
# ---------------------------------------------------------------------------


def test_drop_R_excludes_position_R():
    T = len(_I)
    R = 2
    # _I[2] = 3.0 is excluded → sum = 1+2+4 = 7
    expected = (2.0 * np.pi / T) * 7.0
    result = compute_frequency_variance_single_cycle(_I, R=R, drop_frequency=True)
    assert np.isclose(result, expected)


def test_drop_false_keeps_R():
    T = len(_I)
    R = 2
    expected_all = (2.0 * np.pi / T) * np.sum(_I)
    result = compute_frequency_variance_single_cycle(_I, R=R, drop_frequency=False)
    assert np.isclose(result, expected_all)


def test_drop_first_position():
    I = np.array([10.0, 1.0, 1.0, 1.0])
    T = len(I)
    expected = (2.0 * np.pi / T) * 3.0
    result = compute_frequency_variance_single_cycle(I, R=0, drop_frequency=True)
    assert np.isclose(result, expected)


def test_returns_float():
    result = compute_frequency_variance_single_cycle(_I, R=1)
    assert isinstance(result, float)


def test_R_out_of_range_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_single_cycle(_I, R=10)


def test_negative_R_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_single_cycle(_I, R=-1)


# ---------------------------------------------------------------------------
# compute_frequency_variance_multi_cycle — excludes all R values
# ---------------------------------------------------------------------------


def test_multi_cycle_excludes_all_Rs():
    I = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    T = len(I)
    cycles = [StochasticCycle(R=1, D=0.4), StochasticCycle(R=3, D=0.2)]
    # excludes I[1]=2 and I[3]=4 → sum = 1+3+5 = 9
    expected = (2.0 * np.pi / T) * 9.0
    result = compute_frequency_variance_multi_cycle(I, cycles, drop_frequency=True)
    assert np.isclose(result, expected)


def test_multi_cycle_drop_false_sums_all():
    I = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    T = len(I)
    cycles = [StochasticCycle(R=1, D=0.4), StochasticCycle(R=3, D=0.2)]
    expected = (2.0 * np.pi / T) * np.sum(I)
    result = compute_frequency_variance_multi_cycle(I, cycles, drop_frequency=False)
    assert np.isclose(result, expected)


def test_multi_cycle_single_entry_matches_single_cycle():
    cycle = StochasticCycle(R=2, D=0.5)
    result_multi = compute_frequency_variance_multi_cycle(_I, [cycle], drop_frequency=True)
    result_single = compute_frequency_variance_single_cycle(_I, R=2, drop_frequency=True)
    assert np.isclose(result_multi, result_single)


def test_multi_cycle_empty_cycles_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_multi_cycle(_I, [])
