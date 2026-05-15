import numpy as np
import pytest

from cyclical_fractional_test import InvalidConfigurationError, InvalidSeriesError
from cyclical_fractional_test.spectral import (
    compute_document_periodogram,
    find_periodogram_peak,
    find_top_periodogram_peaks,
)


# ---------------------------------------------------------------------------
# compute_document_periodogram
# ---------------------------------------------------------------------------


def test_compute_document_periodogram_returns_expected_shapes():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    lambdas, periodogram = compute_document_periodogram(x)
    assert lambdas.shape == (4,)
    assert periodogram.shape == (4,)
    assert np.all(periodogram >= 0)


def test_compute_document_periodogram_matches_fft_formula():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    T = len(x)
    _, periodogram = compute_document_periodogram(x)
    expected = np.abs(np.fft.fft(x)) ** 2 / (2.0 * np.pi * T)
    np.testing.assert_allclose(periodogram, expected)


def test_compute_document_periodogram_lambdas_formula():
    x = np.ones(8)
    T = len(x)
    lambdas, _ = compute_document_periodogram(x)
    expected_lambdas = 2.0 * np.pi * np.arange(T) / T
    np.testing.assert_allclose(lambdas, expected_lambdas)


def test_compute_document_periodogram_nonnegative():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    _, periodogram = compute_document_periodogram(x)
    assert np.all(periodogram >= 0)


def test_periodogram_detects_known_frequency_peak():
    T = 64
    j0 = 5
    t = np.arange(T, dtype=float)
    x = np.cos(2.0 * np.pi * j0 * t / T)
    _, periodogram = compute_document_periodogram(x)
    peak = find_periodogram_peak(periodogram, exclude_zero=True)
    # cos signal has equal energy at j0 and T-j0; accept either
    assert peak == j0 or peak == T - j0


def test_compute_document_periodogram_rejects_nan():
    with pytest.raises(InvalidSeriesError):
        compute_document_periodogram(np.array([1.0, np.nan]))


def test_compute_document_periodogram_rejects_inf():
    with pytest.raises(InvalidSeriesError):
        compute_document_periodogram(np.array([1.0, np.inf, 3.0]))


def test_compute_document_periodogram_accepts_list_input():
    lambdas, periodogram = compute_document_periodogram([1.0, 2.0, 3.0, 4.0])
    assert len(lambdas) == 4
    assert len(periodogram) == 4


# ---------------------------------------------------------------------------
# find_periodogram_peak
# ---------------------------------------------------------------------------


def test_find_periodogram_peak_excludes_zero():
    periodogram = np.array([100.0, 2.0, 5.0, 3.0])
    assert find_periodogram_peak(periodogram, exclude_zero=True) == 2
    assert find_periodogram_peak(periodogram, exclude_zero=False) == 0


def test_find_periodogram_peak_returns_int():
    periodogram = np.array([0.0, 1.0, 5.0, 2.0])
    result = find_periodogram_peak(periodogram)
    assert isinstance(result, int)


def test_find_periodogram_peak_simple_case():
    periodogram = np.array([0.0, 1.0, 3.0, 2.0])
    assert find_periodogram_peak(periodogram, exclude_zero=True) == 2


def test_find_periodogram_peak_rejects_empty():
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([]))


def test_find_periodogram_peak_rejects_nan():
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([1.0, np.nan, 2.0]))


def test_find_periodogram_peak_rejects_multidimensional():
    with pytest.raises(InvalidConfigurationError):
        find_periodogram_peak(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# find_top_periodogram_peaks
# ---------------------------------------------------------------------------


def test_find_top_periodogram_peaks_returns_sorted_indices():
    periodogram = np.array([0.0, 10.0, 5.0, 20.0])
    result = find_top_periodogram_peaks(periodogram, n_peaks=2, exclude_zero=True)
    np.testing.assert_array_equal(result, np.array([3, 1]))


def test_find_top_periodogram_peaks_single_peak():
    periodogram = np.array([0.0, 3.0, 7.0, 1.0])
    result = find_top_periodogram_peaks(periodogram, n_peaks=1, exclude_zero=True)
    assert result[0] == 2


def test_find_top_periodogram_peaks_no_exclude_zero():
    periodogram = np.array([100.0, 10.0, 5.0])
    result = find_top_periodogram_peaks(periodogram, n_peaks=2, exclude_zero=False)
    np.testing.assert_array_equal(result, np.array([0, 1]))


def test_find_top_periodogram_peaks_rejects_non_positive_n_peaks():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([1.0, 2.0, 3.0]), n_peaks=0)


def test_find_top_periodogram_peaks_rejects_float_n_peaks():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([1.0, 2.0, 3.0]), n_peaks=1.5)  # type: ignore


def test_find_top_periodogram_peaks_rejects_n_peaks_exceeds_length():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(
            np.array([0.0, 1.0, 2.0]), n_peaks=5, exclude_zero=True
        )


def test_find_top_periodogram_peaks_rejects_empty():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([]), n_peaks=1)


def test_find_top_periodogram_peaks_rejects_nan():
    with pytest.raises(InvalidConfigurationError):
        find_top_periodogram_peaks(np.array([1.0, np.nan, 2.0]), n_peaks=1)
