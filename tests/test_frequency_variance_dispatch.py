"""Tests for Wave 9: compute_frequency_variance_dynamic dispatcher."""
import numpy as np
import pytest

from cyclical_fractional_test.spectral import (
    compute_frequency_variance_dynamic,
    compute_frequency_variance_single_cycle,
)
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidCycleError


_I = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# mode="single" — happy path
# ---------------------------------------------------------------------------


def test_single_dispatch_matches_direct_call():
    cycles = [StochasticCycle(R=2, D=0.4)]
    expected = compute_frequency_variance_single_cycle(_I, R=2, drop_frequency=True)
    result = compute_frequency_variance_dynamic(_I, cycles, mode="single")
    assert np.isclose(result, expected)


def test_multi_peak_single_cycle_matches_single():
    cycles = [StochasticCycle(R=2, D=0.4)]
    result_s = compute_frequency_variance_dynamic(_I, cycles, mode="single")
    result_m = compute_frequency_variance_dynamic(_I, cycles, mode="multi_peak_single_cycle")
    assert np.isclose(result_s, result_m)


def test_drop_frequency_forwarded():
    cycles = [StochasticCycle(R=2, D=0.4)]
    result_drop = compute_frequency_variance_dynamic(_I, cycles, drop_frequency=True)
    result_keep = compute_frequency_variance_dynamic(_I, cycles, drop_frequency=False)
    assert result_drop != result_keep


# ---------------------------------------------------------------------------
# mode="single" — wrong number of cycles
# ---------------------------------------------------------------------------


def test_single_empty_cycles_raises():
    with pytest.raises((InvalidCycleError, ValueError)):
        compute_frequency_variance_dynamic(_I, [], mode="single")


def test_single_two_cycles_raises():
    cycles = [StochasticCycle(R=1, D=0.3), StochasticCycle(R=2, D=0.2)]
    with pytest.raises((InvalidCycleError, ValueError)):
        compute_frequency_variance_dynamic(_I, cycles, mode="single")


# ---------------------------------------------------------------------------
# mode="multi_cycle"
# ---------------------------------------------------------------------------


def test_multi_cycle_dispatch_excludes_all_Rs():
    cycles = [StochasticCycle(R=1, D=0.3), StochasticCycle(R=3, D=0.2)]
    T = len(_I)
    # excludes _I[1]=2 and _I[3]=4 → sum = 1+3+5 = 9
    expected = (2.0 * np.pi / T) * 9.0
    result = compute_frequency_variance_dynamic(_I, cycles, mode="multi_cycle")
    assert np.isclose(result, expected)


# ---------------------------------------------------------------------------
# unknown mode
# ---------------------------------------------------------------------------


def test_unknown_mode_raises():
    cycles = [StochasticCycle(R=2, D=0.4)]
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_dynamic(_I, cycles, mode="unknown")


def test_empty_string_mode_raises():
    cycles = [StochasticCycle(R=2, D=0.4)]
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_dynamic(_I, cycles, mode="")
