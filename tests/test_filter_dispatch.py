"""Tests for Wave 7: apply_filter_dynamic and apply_multi_cycle_filter dispatchers."""
import numpy as np
import pytest

from cyclical_fractional_test.filters import (
    apply_filter_dynamic,
    apply_multi_cycle_filter,
    apply_single_cycle_filter,
)
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidCycleError


T = 10
_rng = np.random.default_rng(7)
_x = _rng.standard_normal(T)


# ---------------------------------------------------------------------------
# apply_filter_dynamic — single mode
# ---------------------------------------------------------------------------


def test_single_dispatch_matches_direct_call():
    cycles = [StochasticCycle(R=2, D=0.5)]
    expected = apply_single_cycle_filter(_x, cycles[0], T)
    result = apply_filter_dynamic(_x, cycles, T, mode="single")
    np.testing.assert_allclose(result, expected)


def test_multi_peak_single_cycle_matches_single():
    cycles = [StochasticCycle(R=3, D=0.3)]
    result_single = apply_filter_dynamic(_x, cycles, T, mode="single")
    result_mpsc = apply_filter_dynamic(_x, cycles, T, mode="multi_peak_single_cycle")
    np.testing.assert_array_equal(result_single, result_mpsc)


def test_default_mode_is_single():
    cycles = [StochasticCycle(R=2, D=0.4)]
    result_default = apply_filter_dynamic(_x, cycles, T)
    result_explicit = apply_filter_dynamic(_x, cycles, T, mode="single")
    np.testing.assert_array_equal(result_default, result_explicit)


# ---------------------------------------------------------------------------
# apply_filter_dynamic — wrong number of cycles for single mode
# ---------------------------------------------------------------------------


def test_single_mode_empty_cycles_raises():
    with pytest.raises((InvalidCycleError, ValueError)):
        apply_filter_dynamic(_x, [], T, mode="single")


def test_single_mode_two_cycles_raises():
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=3, D=0.2)]
    with pytest.raises((InvalidCycleError, ValueError)):
        apply_filter_dynamic(_x, cycles, T, mode="single")


# ---------------------------------------------------------------------------
# apply_filter_dynamic — unknown mode
# ---------------------------------------------------------------------------


def test_unknown_mode_raises():
    cycles = [StochasticCycle(R=2, D=0.4)]
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_filter_dynamic(_x, cycles, T, mode="unknown")


# ---------------------------------------------------------------------------
# apply_multi_cycle_filter — sequential application
# ---------------------------------------------------------------------------


def test_multi_cycle_sequential_matches_chained_singles():
    cycle1 = StochasticCycle(R=2, D=0.4)
    cycle2 = StochasticCycle(R=3, D=0.2)
    expected = apply_single_cycle_filter(
        apply_single_cycle_filter(_x, cycle1, T), cycle2, T
    )
    result = apply_multi_cycle_filter(_x, [cycle1, cycle2], T)
    np.testing.assert_allclose(result, expected)


def test_multi_cycle_single_cycle_matches_single_filter():
    cycle = StochasticCycle(R=2, D=0.5)
    expected = apply_single_cycle_filter(_x, cycle, T)
    result = apply_multi_cycle_filter(_x, [cycle], T)
    np.testing.assert_allclose(result, expected)


def test_multi_cycle_empty_cycles_raises():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        apply_multi_cycle_filter(_x, [], T)


def test_multi_cycle_dispatch_via_dynamic():
    cycle1 = StochasticCycle(R=2, D=0.4)
    cycle2 = StochasticCycle(R=3, D=0.2)
    cycles = [cycle1, cycle2]
    result_dynamic = apply_filter_dynamic(_x, cycles, T, mode="multi_cycle")
    result_direct = apply_multi_cycle_filter(_x, cycles, T)
    np.testing.assert_array_equal(result_dynamic, result_direct)
