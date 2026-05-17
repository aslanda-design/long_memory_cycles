import numpy as np
import pytest

from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidSeriesError
from cyclical_fractional_test.results import StochasticCycle
from cyclical_fractional_test.spectral import (
    compute_document_periodogram,
    compute_frequency_variance_dynamic,
    compute_frequency_variance_multi_cycle,
    compute_frequency_variance_single_cycle,
    compute_psi_dynamic,
    compute_psi_single_cycle,
    compute_residual_periodogram,
    compute_xa_dynamic,
    compute_xa_single_cycle,
    compute_xaa_dynamic,
    compute_xaa_single_cycle,
    find_periodogram_peak,
    find_top_periodogram_peaks,
)


# ---------------------------------------------------------------------------
# compute_document_periodogram
# ---------------------------------------------------------------------------


def test_periodogram_shapes_and_nonnegativity():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    lambdas, I = compute_document_periodogram(x)
    assert lambdas.shape == (4,)
    assert I.shape == (4,)
    assert np.all(I >= 0)


def test_periodogram_matches_fft_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(x)
    _, I = compute_document_periodogram(x)
    np.testing.assert_allclose(I, np.abs(np.fft.fft(x)) ** 2 / (2.0 * np.pi * T))


def test_periodogram_lambdas_formula():
    T = 8
    lambdas, _ = compute_document_periodogram(np.ones(T))
    np.testing.assert_allclose(lambdas, 2.0 * np.pi * np.arange(T) / T)


def test_periodogram_detects_known_frequency():
    T = 64
    j0 = 5
    x = np.cos(2.0 * np.pi * j0 * np.arange(T, dtype=float) / T)
    _, I = compute_document_periodogram(x)
    peak = find_periodogram_peak(I, exclude_zero=True)
    assert peak == j0 or peak == T - j0


def test_periodogram_rejects_non_finite():
    with pytest.raises(InvalidSeriesError):
        compute_document_periodogram(np.array([1.0, np.nan]))
    with pytest.raises(InvalidSeriesError):
        compute_document_periodogram(np.array([1.0, np.inf, 3.0]))


# ---------------------------------------------------------------------------
# find_periodogram_peak
# ---------------------------------------------------------------------------


def test_find_peak_respects_exclude_zero():
    I = np.array([100.0, 2.0, 5.0, 3.0])
    assert find_periodogram_peak(I, exclude_zero=True) == 2
    assert find_periodogram_peak(I, exclude_zero=False) == 0


def test_find_peak_returns_int():
    assert isinstance(find_periodogram_peak(np.array([0.0, 1.0, 5.0, 2.0])), int)


def test_find_peak_rejects_empty_or_nan():
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([]))
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([1.0, np.nan, 2.0]))


# ---------------------------------------------------------------------------
# find_top_periodogram_peaks
# ---------------------------------------------------------------------------


def test_top_peaks_sorted_descending():
    I = np.array([0.0, 10.0, 5.0, 20.0])
    result = find_top_periodogram_peaks(I, n_peaks=2, exclude_zero=True)
    np.testing.assert_array_equal(result, np.array([3, 1]))


def test_top_peaks_rejects_n_peaks_exceeding_available():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([0.0, 1.0, 2.0]), n_peaks=5, exclude_zero=True)


def test_top_peaks_rejects_non_positive_n_peaks():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([1.0, 2.0, 3.0]), n_peaks=0)


# ---------------------------------------------------------------------------
# compute_residual_periodogram
# ---------------------------------------------------------------------------


def test_residual_periodogram_matches_document_periodogram():
    residuals = np.random.default_rng(0).standard_normal(20)
    lam1, I1 = compute_residual_periodogram(residuals)
    lam2, I2 = compute_document_periodogram(residuals)
    np.testing.assert_array_equal(lam1, lam2)
    np.testing.assert_array_equal(I1, I2)


def test_residual_periodogram_all_zeros_returns_zeros():
    _, I = compute_residual_periodogram(np.zeros(8))
    np.testing.assert_allclose(I, 0.0)


def test_residual_periodogram_output_length_matches_input():
    T = 16
    lambdas, I = compute_residual_periodogram(np.random.default_rng(1).standard_normal(T))
    assert len(lambdas) == T and len(I) == T


# ---------------------------------------------------------------------------
# compute_psi_single_cycle
# ---------------------------------------------------------------------------


def test_psi_shape_and_finiteness():
    psi = compute_psi_single_cycle(T=100, R=25)
    assert psi.shape == (100,)
    assert np.all(np.isfinite(psi))


def test_psi_singular_frequencies_zeroed():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    assert psi[R] == 0.0
    assert psi[T - R] == 0.0


def test_psi_R_equals_T_half_single_zero():
    T, R = 100, 50
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    assert psi[50] == 0.0
    assert np.all(np.isfinite(psi))


def test_psi_matches_formula_at_non_singular_index():
    T, R, j = 100, 25, 10
    psi = compute_psi_single_cycle(T=T, R=R)
    lam_j = 2.0 * np.pi * j / T
    lam_R = 2.0 * np.pi * R / T
    expected = np.log(np.abs(2.0 * (np.cos(lam_j) - np.cos(lam_R))))
    np.testing.assert_allclose(psi[j], expected)


def test_psi_drop_false_gives_neg_inf_at_exact_singular():
    psi = compute_psi_single_cycle(T=100, R=25, drop_singular_frequency=False)
    assert psi[25] == -np.inf


@pytest.mark.parametrize("R", [0, -1, 100])
def test_psi_rejects_invalid_R(R):
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=R)


def test_psi_rejects_non_integer_and_bool_R():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=25.5)  # type: ignore
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=True)  # type: ignore


# ---------------------------------------------------------------------------
# compute_xaa_single_cycle
# ---------------------------------------------------------------------------


def test_xaa_matches_formula():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R)
    xaa = compute_xaa_single_cycle(psi)
    np.testing.assert_allclose(xaa, float((2.0 / T) * np.sum(psi ** 2)))


def test_xaa_all_zero_psi_gives_zero():
    assert compute_xaa_single_cycle(np.zeros(50)) == pytest.approx(0.0)


def test_xaa_is_positive():
    psi = compute_psi_single_cycle(T=100, R=25)
    assert compute_xaa_single_cycle(psi) > 0


def test_xaa_rejects_non_finite_or_empty():
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(np.array([]))


# ---------------------------------------------------------------------------
# compute_xa_single_cycle
# ---------------------------------------------------------------------------


def test_xa_known_value():
    psi = np.array([1.0, 2.0, 3.0])
    I = np.array([4.0, 5.0, 6.0])
    expected = -(2.0 * np.pi / 3.0) * (1 * 4 + 2 * 5 + 3 * 6)
    assert np.isclose(compute_xa_single_cycle(psi, I), expected)


def test_xa_sign_is_negative_for_positive_inputs():
    assert compute_xa_single_cycle(np.ones(4), np.ones(4)) < 0.0


def test_xa_zero_inputs_return_zero():
    assert compute_xa_single_cycle(np.zeros(5), np.ones(5)) == 0.0
    assert compute_xa_single_cycle(np.ones(5), np.zeros(5)) == 0.0


def test_xa_rejects_mismatched_shapes():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_xa_single_cycle(np.ones(3), np.ones(4))


def test_xa_rejects_non_finite():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_xa_single_cycle(np.array([1.0, np.nan, 3.0]), np.ones(3))
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_xa_single_cycle(np.ones(3), np.array([1.0, np.inf, 3.0]))


# ---------------------------------------------------------------------------
# compute_frequency_variance_single_cycle
# ---------------------------------------------------------------------------


def test_freq_var_no_drop_sums_all():
    I = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(I)
    expected = (2.0 * np.pi / T) * np.sum(I)
    assert np.isclose(
        compute_frequency_variance_single_cycle(I, R=2, drop_frequency=False), expected
    )


def test_freq_var_drop_excludes_position_R():
    I = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(I)
    expected = (2.0 * np.pi / T) * (1.0 + 2.0 + 4.0)
    assert np.isclose(
        compute_frequency_variance_single_cycle(I, R=2, drop_frequency=True), expected
    )


def test_freq_var_multi_cycle_excludes_all_Rs():
    I = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    T = len(I)
    cycles = [StochasticCycle(R=1, D=0.4), StochasticCycle(R=3, D=0.2)]
    expected = (2.0 * np.pi / T) * (1.0 + 3.0 + 5.0)
    assert np.isclose(
        compute_frequency_variance_multi_cycle(I, cycles, drop_frequency=True), expected
    )


def test_freq_var_rejects_R_out_of_range():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_single_cycle(np.ones(4), R=10)


def test_freq_var_multi_cycle_rejects_empty_cycles():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        compute_frequency_variance_multi_cycle(np.ones(4), [])


# ---------------------------------------------------------------------------
# Dispatcher smoke tests
# ---------------------------------------------------------------------------


def _cycle(R=3, D=0.4):
    return StochasticCycle(R=R, D=D)


def test_psi_dynamic_single_matches_direct():
    T, R = 30, 5
    cycles = [_cycle(R=R)]
    expected = compute_psi_single_cycle(T=T, R=R)
    np.testing.assert_allclose(
        compute_psi_dynamic(T, cycles, stochastic_cycle_mode="single"), expected
    )


def test_psi_dynamic_multi_cycle_not_implemented():
    with pytest.raises(NotImplementedError):
        compute_psi_dynamic(30, [_cycle(), _cycle(R=5)], stochastic_cycle_mode="multi_cycle")


def test_xaa_dynamic_single_matches_direct():
    T, R = 30, 5
    psi = compute_psi_single_cycle(T=T, R=R)
    np.testing.assert_allclose(
        compute_xaa_dynamic(psi, stochastic_cycle_mode="single"),
        compute_xaa_single_cycle(psi),
    )


def test_xa_dynamic_single_matches_direct():
    T = 20
    residuals = np.random.default_rng(1).standard_normal(T)
    psi = compute_psi_single_cycle(T=T, R=3)
    _, I = compute_residual_periodogram(residuals)
    expected = compute_xa_single_cycle(psi, I)
    np.testing.assert_allclose(
        compute_xa_dynamic(psi, I, [_cycle(R=3)], mode="single"), expected
    )


def test_freq_var_dynamic_single_matches_direct():
    I = np.random.default_rng(2).random(20)
    expected = compute_frequency_variance_single_cycle(I, R=3, drop_frequency=True)
    result = compute_frequency_variance_dynamic(I, [_cycle(R=3)], mode="single", drop_frequency=True)
    np.testing.assert_allclose(result, expected)
