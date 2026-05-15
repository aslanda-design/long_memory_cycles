import numpy as np
import pytest

from cyclical_fractional_test import InvalidConfigurationError
from cyclical_fractional_test.spectral import (
    compute_psi_single_cycle,
    compute_xaa_single_cycle,
)


# ---------------------------------------------------------------------------
# compute_psi_single_cycle — basic correctness
# ---------------------------------------------------------------------------


def test_compute_psi_single_cycle_returns_expected_shape():
    psi = compute_psi_single_cycle(T=100, R=25)
    assert psi.shape == (100,)
    assert np.all(np.isfinite(psi))


def test_compute_psi_single_cycle_sets_singular_frequency_to_zero():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    assert psi[R] == 0.0
    mirror = T - R  # = 75
    assert psi[mirror] == 0.0


def test_compute_psi_single_cycle_sets_only_singular_when_R_equals_T_half():
    # When R = T/2, j=R and j=T-R are the same index — only one zero.
    T, R = 100, 50
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    assert psi[50] == 0.0
    assert np.all(np.isfinite(psi))


def test_compute_psi_single_cycle_matches_formula_non_singular():
    T, R, j = 100, 25, 10
    psi = compute_psi_single_cycle(T=T, R=R)
    lambda_j = 2.0 * np.pi * j / T
    lambda_R = 2.0 * np.pi * R / T
    expected = np.log(np.abs(2.0 * (np.cos(lambda_j) - np.cos(lambda_R))))
    np.testing.assert_allclose(psi[j], expected)


def test_compute_psi_single_cycle_drop_false_gives_neg_inf_at_exact_singular():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=False)
    # j=R is *exactly* singular: lambda_j[R] == lambda_R (same arithmetic expression),
    # so cos(lambda_j[R]) - cos(lambda_R) is identically 0 and log(0) == -inf.
    assert psi[R] == -np.inf
    # j=T-R is *approximately* singular: floating-point imprecision in cos means
    # the argument is ≈ 0 but not exactly, producing a large finite negative value.
    mirror = T - R
    assert psi[mirror] < -30


def test_compute_psi_single_cycle_non_singular_values_unchanged_by_drop_flag():
    T, R, j = 100, 25, 10
    psi_drop = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=True)
    psi_keep = compute_psi_single_cycle(T=T, R=R, drop_singular_frequency=False)
    np.testing.assert_allclose(psi_drop[j], psi_keep[j])


# ---------------------------------------------------------------------------
# compute_psi_single_cycle — validation errors
# ---------------------------------------------------------------------------


def test_compute_psi_single_cycle_rejects_R_zero():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=0)


def test_compute_psi_single_cycle_rejects_R_equals_T():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=100)


def test_compute_psi_single_cycle_rejects_R_negative():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=-1)


def test_compute_psi_single_cycle_rejects_non_integer_R():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=25.5)  # type: ignore


def test_compute_psi_single_cycle_rejects_bool_R():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100, R=True)  # type: ignore


def test_compute_psi_single_cycle_rejects_T_less_than_2():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=1, R=1)


def test_compute_psi_single_cycle_rejects_float_T():
    with pytest.raises(InvalidConfigurationError):
        compute_psi_single_cycle(T=100.0, R=25)  # type: ignore


# ---------------------------------------------------------------------------
# compute_xaa_single_cycle — correctness
# ---------------------------------------------------------------------------


def test_compute_xaa_single_cycle_returns_positive_float():
    psi = compute_psi_single_cycle(T=100, R=25)
    xaa = compute_xaa_single_cycle(psi)
    assert isinstance(xaa, float)
    assert xaa > 0


def test_compute_xaa_single_cycle_matches_formula():
    T, R = 100, 25
    psi = compute_psi_single_cycle(T=T, R=R)
    xaa = compute_xaa_single_cycle(psi)
    expected = float((2.0 / T) * np.sum(psi ** 2))
    np.testing.assert_allclose(xaa, expected)


def test_compute_xaa_single_cycle_all_zero_psi_gives_zero():
    psi = np.zeros(50)
    assert compute_xaa_single_cycle(psi) == pytest.approx(0.0)


def test_compute_xaa_single_cycle_scales_with_T():
    # XAA = (2/T) * sum(psi²) — doubling T with same psi halves XAA
    psi = compute_psi_single_cycle(T=100, R=20)
    xaa_100 = compute_xaa_single_cycle(psi)
    # Manually scale: same sum, different T
    xaa_manual_200 = float((2.0 / 200) * np.sum(psi ** 2))
    assert xaa_100 == pytest.approx(2 * xaa_manual_200)


# ---------------------------------------------------------------------------
# compute_xaa_single_cycle — validation errors
# ---------------------------------------------------------------------------


def test_compute_xaa_single_cycle_rejects_non_finite_psi():
    psi = np.array([1.0, np.nan, 2.0])
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(psi)


def test_compute_xaa_single_cycle_rejects_inf_psi():
    psi = np.array([1.0, -np.inf, 2.0])
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(psi)


def test_compute_xaa_single_cycle_rejects_empty_psi():
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(np.array([]))


def test_compute_xaa_single_cycle_rejects_2d_psi():
    with pytest.raises(InvalidConfigurationError):
        compute_xaa_single_cycle(np.ones((5, 5)))
