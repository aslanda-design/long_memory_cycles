import numpy as np
import pytest

from cyclical_fractional_test import (
    CyclicalFractionalTestResult,
    CyclicalTestConfig,
    GridCandidateResult,
    StochasticCycle,
    run_cyclical_fractional_test,
)
from cyclical_fractional_test.diagnostics import PeriodogramSummary, TestDiagnostics
from cyclical_fractional_test.exceptions import InvalidConfigurationError, InvalidSeriesError


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_stochastic_cycle_fields():
    c = StochasticCycle(R=25, D=0.4)
    assert c.R == 25 and c.D == 0.4


def test_grid_candidate_result_defaults_to_none():
    result = GridCandidateResult(cycles=(StochasticCycle(R=10, D=0.3),))
    assert result.test_value is None
    assert result.betas is None


def test_cyclical_result_empty_defaults():
    result = CyclicalFractionalTestResult()
    assert result.best_result is None
    assert result.top_k_results == []
    assert result.config is None


def test_config_default_values():
    cfg = CyclicalTestConfig()
    assert cfg.n_deterministic_cycles == 4
    assert cfg.top_k == 1
    assert cfg.stochastic_cycle_mode == "single"
    assert cfg.d_grid is None
    assert cfg.drop_singular_frequency is True


def test_config_custom_values():
    cfg = CyclicalTestConfig(n_deterministic_cycles=2, top_k=3, r_window=5, include_intercept=True)
    assert cfg.n_deterministic_cycles == 2
    assert cfg.top_k == 3
    assert cfg.r_window == 5
    assert cfg.include_intercept is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(**overrides):
    defaults = dict(
        n_deterministic_cycles=2,
        d_grid=np.array([0.0, 0.5]),
        r_window=1,
        top_k=2,
        stochastic_cycle_mode="single",
    )
    defaults.update(overrides)
    return CyclicalTestConfig(**defaults)


def _series(T=50, seed=0):
    return np.random.default_rng(seed).standard_normal(T)


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


def test_returns_cyclical_fractional_test_result():
    assert isinstance(
        run_cyclical_fractional_test(_series(), config=_small_config()),
        CyclicalFractionalTestResult,
    )


def test_best_result_not_none():
    assert run_cyclical_fractional_test(_series(), config=_small_config()).best_result is not None


def test_top_k_results_length():
    result = run_cyclical_fractional_test(_series(), config=_small_config(top_k=2))
    assert len(result.top_k_results) == 2


def test_best_result_equals_first_top_k():
    result = run_cyclical_fractional_test(_series(), config=_small_config(top_k=3))
    assert result.best_result is result.top_k_results[0]


def test_betas_length_matches_n_deterministic_cycles():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=3,
            include_intercept=False,
            d_grid=np.array([0.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert all(len(c.betas) == 3 for c in result.top_k_results)


def test_include_intercept_adds_extra_beta():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=3,
            include_intercept=True,
            d_grid=np.array([0.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert all(len(c.betas) == 4 for c in result.top_k_results)


def test_all_test_values_are_finite():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert all(np.isfinite(c.test_value) and np.isfinite(c.test_star_value) for c in result.top_k_results)


def test_residuals_length_matches_T():
    T = 50
    result = run_cyclical_fractional_test(_series(T=T), config=_small_config())
    assert all(c.residuals.shape == (T,) for c in result.top_k_results)


# ---------------------------------------------------------------------------
# Config and kwargs
# ---------------------------------------------------------------------------


def test_kwargs_override_config_fields():
    config = _small_config(top_k=1)
    result = run_cyclical_fractional_test(_series(), config=config, top_k=2)
    assert len(result.top_k_results) == 2


def test_kwargs_work_without_config():
    result = run_cyclical_fractional_test(
        _series(),
        n_deterministic_cycles=2,
        d_grid=np.array([0.0]),
        r_window=0,
        top_k=1,
        stochastic_cycle_mode="single",
    )
    assert result.best_result is not None


def test_default_config_runs():
    result = run_cyclical_fractional_test(_series(T=60))
    assert result.best_result is not None


def test_multi_cycle_mode_not_implemented():
    config = CyclicalTestConfig(
        n_deterministic_cycles=2,
        d_grid=np.array([0.0]),
        r_window=1,
        top_k=1,
        stochastic_cycle_mode="multi_cycle",
    )
    with pytest.raises(NotImplementedError):
        run_cyclical_fractional_test(_series(), config=config)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_series", [
    np.array([1.0, float("nan"), 3.0, 4.0, 5.0]),
    np.array([1.0, float("inf"), 3.0, 4.0, 5.0]),
    np.ones((10, 2)),
    np.array([1.0, 2.0]),
])
def test_rejects_invalid_series(bad_series):
    with pytest.raises((InvalidSeriesError, ValueError)):
        run_cyclical_fractional_test(bad_series, config=_small_config())


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_rejects_invalid_top_k(bad_top_k):
    with pytest.raises((InvalidConfigurationError, ValueError)):
        run_cyclical_fractional_test(_series(T=30), config=_small_config(top_k=bad_top_k))


def test_rejects_non_finite_d_grid():
    with pytest.raises((InvalidConfigurationError, ValueError)):
        run_cyclical_fractional_test(
            _series(T=30),
            config=_small_config(d_grid=np.array([0.0, float("nan")])),
        )


# ---------------------------------------------------------------------------
# Top-k ordering and counting
# ---------------------------------------------------------------------------


def test_top_k_ordered_ascending_by_abs_test_value():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=42),
        config=_small_config(d_grid=np.array([0.0, 0.3, 0.7, 1.0]), r_window=2, top_k=4),
    )
    scores = [abs(c.test_value) for c in result.top_k_results]
    assert scores == sorted(scores)


def test_n_candidates_evaluated_matches_grid_size():
    result = run_cyclical_fractional_test(
        _series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_grid=np.array([0.0, 1.0]),
            r_window=1,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert result.n_candidates_evaluated == len(result.r_candidates) * 2


def test_fewer_candidates_than_top_k_returns_all():
    result = run_cyclical_fractional_test(
        _series(),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_grid=np.array([0.0]),
            r_window=0,
            top_k=10,
            stochastic_cycle_mode="single",
        ),
    )
    assert len(result.top_k_results) == 1
    assert result.n_candidates_evaluated == 1


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_present_and_correct_type():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert isinstance(result.diagnostics, TestDiagnostics)


def test_diagnostics_counters_consistent():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    diag = result.diagnostics
    assert diag.n_candidates_evaluated > 0
    assert diag.n_valid_candidates == diag.n_candidates_evaluated
    assert diag.n_failed_candidates == 0


def test_diagnostics_r_peak_and_candidates_match_result():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    assert result.diagnostics.r_peak == result.r_peak
    assert result.diagnostics.r_candidates_count == len(result.r_candidates)


def test_diagnostics_periodogram_summary_consistent():
    result = run_cyclical_fractional_test(_series(), config=_small_config())
    ps = result.diagnostics.periodogram_summary
    assert isinstance(ps, PeriodogramSummary)
    assert ps.peak_index == result.r_peak


def test_diagnostics_does_not_affect_ordering():
    result = run_cyclical_fractional_test(
        _series(T=60, seed=42),
        config=_small_config(d_grid=np.array([0.0, 0.5]), r_window=2, top_k=3),
    )
    assert result.best_result is result.top_k_results[0]
    scores = [abs(c.test_value) for c in result.top_k_results]
    assert scores == sorted(scores)


def test_diagnostics_default_d_grid_has_11_values():
    result = run_cyclical_fractional_test(
        _series(T=60),
        config=CyclicalTestConfig(
            n_deterministic_cycles=2,
            d_grid=None,
            r_window=0,
            top_k=1,
            stochastic_cycle_mode="single",
        ),
    )
    assert result.diagnostics.d_grid_count == 11


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_full_pipeline_with_structured_series():
    from cyclical_fractional_test.chebyshev import build_chebyshev_design

    rng = np.random.default_rng(2024)
    T = 30
    X_ref = build_chebyshev_design(T, 2)
    y = 0.5 * X_ref[:, 0] + 0.2 * X_ref[:, 1] + 0.1 * rng.standard_normal(T)

    config = CyclicalTestConfig(
        n_deterministic_cycles=2,
        include_intercept=False,
        d_grid=np.array([0.0, 0.5]),
        r_window=1,
        top_k=2,
        stochastic_cycle_mode="single",
    )
    result = run_cyclical_fractional_test(y, config=config)

    assert isinstance(result, CyclicalFractionalTestResult)
    assert result.best_result is result.top_k_results[0]
    for cand in result.top_k_results:
        assert np.isfinite(cand.test_value)
        assert cand.betas.shape == (2,)
        assert cand.residuals.shape == (T,)
    scores = [abs(c.test_value) for c in result.top_k_results]
    assert scores == sorted(scores)
