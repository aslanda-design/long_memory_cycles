"""Tests for Wave 6: compute_fractional_coefficients_dynamic dispatcher."""
import numpy as np
import pytest

from cyclical_fractional_test.filters import (
    compute_fractional_coefficients_dynamic,
    compute_fractional_coefficients_single_cycle,
)
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.exceptions import (
    InvalidConfigurationError,
    InvalidCycleError,
)


# ---------------------------------------------------------------------------
# mode="single" — happy path
# ---------------------------------------------------------------------------


def test_single_mode_matches_direct_call():
    T, R, D = 20, 2, 0.5
    cycles = [StochasticCycle(R=R, D=D)]
    expected = compute_fractional_coefficients_single_cycle(T, R, D)
    result = compute_fractional_coefficients_dynamic(T, cycles, mode="single")
    np.testing.assert_allclose(result, expected)


def test_multi_peak_single_cycle_mode_matches_direct_call():
    """multi_peak_single_cycle should delegate to the single-cycle path."""
    T, R, D = 15, 3, 0.3
    cycles = [StochasticCycle(R=R, D=D)]
    expected = compute_fractional_coefficients_single_cycle(T, R, D)
    result = compute_fractional_coefficients_dynamic(
        T, cycles, mode="multi_peak_single_cycle"
    )
    np.testing.assert_allclose(result, expected)


def test_default_mode_is_single():
    T, R, D = 10, 2, 0.4
    cycles = [StochasticCycle(R=R, D=D)]
    result_default = compute_fractional_coefficients_dynamic(T, cycles)
    result_explicit = compute_fractional_coefficients_dynamic(T, cycles, mode="single")
    np.testing.assert_array_equal(result_default, result_explicit)


# ---------------------------------------------------------------------------
# mode="single" — wrong number of cycles
# ---------------------------------------------------------------------------


def test_single_mode_empty_cycles_raises():
    with pytest.raises((InvalidCycleError, ValueError)):
        compute_fractional_coefficients_dynamic(10, [], mode="single")


def test_single_mode_two_cycles_raises():
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=4, D=0.2)]
    with pytest.raises((InvalidCycleError, ValueError)):
        compute_fractional_coefficients_dynamic(10, cycles, mode="single")


def test_multi_peak_single_cycle_mode_two_cycles_raises():
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=4, D=0.2)]
    with pytest.raises((InvalidCycleError, ValueError)):
        compute_fractional_coefficients_dynamic(
            10, cycles, mode="multi_peak_single_cycle"
        )


# ---------------------------------------------------------------------------
# mode="multi_cycle" — placeholder raises NotImplementedError
# ---------------------------------------------------------------------------


def test_multi_cycle_mode_raises_not_implemented():
    cycles = [StochasticCycle(R=2, D=0.4), StochasticCycle(R=4, D=0.2)]
    with pytest.raises(NotImplementedError):
        compute_fractional_coefficients_dynamic(10, cycles, mode="multi_cycle")


def test_multi_cycle_mode_message_is_informative():
    cycles = [StochasticCycle(R=2, D=0.4)]
    with pytest.raises(NotImplementedError, match="[Mm]ulti"):
        compute_fractional_coefficients_dynamic(10, cycles, mode="multi_cycle")


# ---------------------------------------------------------------------------
# unknown mode
# ---------------------------------------------------------------------------


def test_unknown_mode_raises():
    cycles = [StochasticCycle(R=2, D=0.4)]
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_dynamic(10, cycles, mode="unknown")


def test_unknown_mode_empty_string_raises():
    cycles = [StochasticCycle(R=2, D=0.4)]
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_fractional_coefficients_dynamic(10, cycles, mode="")
