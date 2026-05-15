import numpy as np
import pytest

from cyclical_fractional_test import InvalidCycleError, StochasticCycle
from cyclical_fractional_test.spectral import (
    compute_psi_dynamic,
    compute_psi_single_cycle,
    compute_xaa_dynamic,
    compute_xaa_single_cycle,
)


# ---------------------------------------------------------------------------
# compute_psi_dynamic — single mode
# ---------------------------------------------------------------------------


def test_compute_psi_dynamic_single_mode_uses_single_cycle():
    T = 100
    cycles = (StochasticCycle(R=25, D=0.4),)
    result = compute_psi_dynamic(T=T, cycles=cycles, stochastic_cycle_mode="single")
    expected = compute_psi_single_cycle(T=T, R=25)
    np.testing.assert_allclose(result, expected)


def test_compute_psi_dynamic_single_mode_passes_drop_flag():
    T = 100
    cycles = (StochasticCycle(R=25, D=0.4),)
    result = compute_psi_dynamic(
        T=T, cycles=cycles, stochastic_cycle_mode="single",
        drop_singular_frequency=True,
    )
    assert result[25] == 0.0
    assert result[75] == 0.0


def test_compute_psi_dynamic_rejects_single_mode_with_multiple_cycles():
    cycles = (
        StochasticCycle(R=25, D=0.4),
        StochasticCycle(R=30, D=0.2),
    )
    with pytest.raises(InvalidCycleError):
        compute_psi_dynamic(T=100, cycles=cycles, stochastic_cycle_mode="single")


# ---------------------------------------------------------------------------
# compute_psi_dynamic — multi_peak_single_cycle mode
# ---------------------------------------------------------------------------


def test_compute_psi_dynamic_multi_peak_single_cycle_same_as_single():
    T = 100
    cycles = (StochasticCycle(R=25, D=0.4),)
    result_single = compute_psi_dynamic(
        T=T, cycles=cycles, stochastic_cycle_mode="single"
    )
    result_mpsc = compute_psi_dynamic(
        T=T, cycles=cycles, stochastic_cycle_mode="multi_peak_single_cycle"
    )
    np.testing.assert_allclose(result_single, result_mpsc)


def test_compute_psi_dynamic_multi_peak_rejects_multiple_cycles():
    cycles = (StochasticCycle(R=25, D=0.4), StochasticCycle(R=30, D=0.2))
    with pytest.raises(InvalidCycleError):
        compute_psi_dynamic(
            T=100, cycles=cycles, stochastic_cycle_mode="multi_peak_single_cycle"
        )


# ---------------------------------------------------------------------------
# compute_psi_dynamic — multi_cycle placeholder
# ---------------------------------------------------------------------------


def test_compute_psi_dynamic_multi_cycle_placeholder():
    cycles = (StochasticCycle(R=25, D=0.4),)
    with pytest.raises(NotImplementedError):
        compute_psi_dynamic(T=100, cycles=cycles, stochastic_cycle_mode="multi_cycle")


# ---------------------------------------------------------------------------
# compute_xaa_dynamic — single mode
# ---------------------------------------------------------------------------


def test_compute_xaa_dynamic_single_mode_matches_direct_call():
    T = 100
    psi = compute_psi_single_cycle(T=T, R=25)
    result = compute_xaa_dynamic(psi=psi, stochastic_cycle_mode="single")
    expected = compute_xaa_single_cycle(psi)
    np.testing.assert_allclose(result, expected)


def test_compute_xaa_dynamic_multi_peak_single_cycle_same_as_single():
    T = 100
    psi = compute_psi_single_cycle(T=T, R=25)
    result_s = compute_xaa_dynamic(psi=psi, stochastic_cycle_mode="single")
    result_m = compute_xaa_dynamic(psi=psi, stochastic_cycle_mode="multi_peak_single_cycle")
    np.testing.assert_allclose(result_s, result_m)


# ---------------------------------------------------------------------------
# compute_xaa_dynamic — multi_cycle placeholder
# ---------------------------------------------------------------------------


def test_compute_xaa_dynamic_multi_cycle_placeholder():
    psi = np.ones(100)
    with pytest.raises(NotImplementedError):
        compute_xaa_dynamic(psi=psi, stochastic_cycle_mode="multi_cycle")


# ---------------------------------------------------------------------------
# Round-trip: psi_dynamic → xaa_dynamic
# ---------------------------------------------------------------------------


def test_psi_xaa_dynamic_round_trip():
    T = 100
    cycles = (StochasticCycle(R=30, D=0.5),)
    psi = compute_psi_dynamic(T=T, cycles=cycles, stochastic_cycle_mode="single")
    xaa = compute_xaa_dynamic(psi=psi, stochastic_cycle_mode="single")

    # Verify against direct calls
    psi_direct = compute_psi_single_cycle(T=T, R=30)
    xaa_direct = compute_xaa_single_cycle(psi_direct)
    np.testing.assert_allclose(xaa, xaa_direct)
