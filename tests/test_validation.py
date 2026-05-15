import numpy as np
import pytest

from cyclical_fractional_test import (
    CyclicalTestConfig,
    InvalidCycleError,
    InvalidConfigurationError,
    InvalidSeriesError,
    StochasticCycle,
    run_cyclical_fractional_test,
)
from cyclical_fractional_test.validation import (
    validate_boolean,
    validate_config,
    validate_cycle,
    validate_cycles,
    validate_d_grid,
    validate_mode,
    validate_n_deterministic_cycles,
    validate_r_window,
    validate_series,
    validate_top_k,
)


# ---------------------------------------------------------------------------
# validate_series
# ---------------------------------------------------------------------------


def test_validate_series_accepts_numeric_list():
    result = validate_series([1, 2, 3, 4, 5])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (5,)


def test_validate_series_accepts_numpy_array():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = validate_series(arr)
    assert result.ndim == 1
    assert result.dtype == float


def test_validate_series_rejects_none():
    with pytest.raises(InvalidSeriesError):
        validate_series(None)


def test_validate_series_rejects_empty_array():
    with pytest.raises(InvalidSeriesError):
        validate_series([])


def test_validate_series_rejects_multidimensional_array():
    with pytest.raises(InvalidSeriesError):
        validate_series(np.array([[1, 2], [3, 4]]))


def test_validate_series_rejects_nan():
    with pytest.raises(InvalidSeriesError):
        validate_series([1, np.nan, 3, 4, 5])


def test_validate_series_rejects_inf():
    with pytest.raises(InvalidSeriesError):
        validate_series([1, np.inf, 3, 4, 5])


def test_validate_series_rejects_non_numeric_values():
    with pytest.raises(InvalidSeriesError):
        validate_series([1, "a", 3, 4, 5])


def test_validate_series_rejects_series_below_min_length():
    with pytest.raises(InvalidSeriesError):
        validate_series([1.0, 2.0])


# ---------------------------------------------------------------------------
# validate_n_deterministic_cycles
# ---------------------------------------------------------------------------


def test_validate_n_deterministic_cycles_accepts_positive_integer():
    assert validate_n_deterministic_cycles(4) == 4


def test_validate_n_deterministic_cycles_rejects_zero():
    with pytest.raises(InvalidConfigurationError):
        validate_n_deterministic_cycles(0)


def test_validate_n_deterministic_cycles_rejects_negative():
    with pytest.raises(InvalidConfigurationError):
        validate_n_deterministic_cycles(-1)


def test_validate_n_deterministic_cycles_rejects_float():
    with pytest.raises(InvalidConfigurationError):
        validate_n_deterministic_cycles(4.5)


# ---------------------------------------------------------------------------
# validate_top_k
# ---------------------------------------------------------------------------


def test_validate_top_k_accepts_positive_integer():
    assert validate_top_k(5) == 5


def test_validate_top_k_rejects_zero():
    with pytest.raises(InvalidConfigurationError):
        validate_top_k(0)


def test_validate_top_k_rejects_negative():
    with pytest.raises(InvalidConfigurationError):
        validate_top_k(-3)


def test_validate_top_k_rejects_float():
    with pytest.raises(InvalidConfigurationError):
        validate_top_k(1.0)


# ---------------------------------------------------------------------------
# validate_r_window
# ---------------------------------------------------------------------------


def test_validate_r_window_accepts_zero():
    assert validate_r_window(0) == 0


def test_validate_r_window_accepts_positive_integer():
    assert validate_r_window(10) == 10


def test_validate_r_window_rejects_negative():
    with pytest.raises(InvalidConfigurationError):
        validate_r_window(-1)


def test_validate_r_window_rejects_float():
    with pytest.raises(InvalidConfigurationError):
        validate_r_window(10.0)


# ---------------------------------------------------------------------------
# validate_d_grid
# ---------------------------------------------------------------------------


def test_validate_d_grid_accepts_none():
    assert validate_d_grid(None) is None


def test_validate_d_grid_accepts_valid_grid():
    result = validate_d_grid([0.0, 0.1, 0.5, 1.0])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float


def test_validate_d_grid_rejects_empty_grid():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([])


def test_validate_d_grid_rejects_nan():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.1, np.nan, 0.5])


def test_validate_d_grid_rejects_inf():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.1, np.inf])


def test_validate_d_grid_rejects_values_below_zero():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([-0.1, 0.2])


def test_validate_d_grid_rejects_values_above_one():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.2, 1.2])


def test_validate_d_grid_rejects_non_numeric():
    with pytest.raises(InvalidConfigurationError):
        validate_d_grid([0.1, "bad"])


# ---------------------------------------------------------------------------
# validate_mode
# ---------------------------------------------------------------------------


def test_validate_config_accepts_valid_modes():
    cfg = CyclicalTestConfig(
        variance_mode="time",
        statistic_mode="test",
        stochastic_cycle_mode="single",
    )
    validated = validate_config(cfg)
    assert validated is cfg


def test_validate_config_rejects_invalid_variance_mode():
    cfg = CyclicalTestConfig(variance_mode="unknown")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_statistic_mode():
    cfg = CyclicalTestConfig(statistic_mode="bad_mode")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


def test_validate_config_rejects_invalid_stochastic_cycle_mode():
    cfg = CyclicalTestConfig(stochastic_cycle_mode="triple")
    with pytest.raises(InvalidConfigurationError):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# validate_boolean
# ---------------------------------------------------------------------------


def test_validate_boolean_accepts_true():
    assert validate_boolean(True, "some_flag") is True


def test_validate_boolean_accepts_false():
    assert validate_boolean(False, "some_flag") is False


def test_validate_config_rejects_non_boolean_flags():
    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(include_intercept="yes"))  # type: ignore

    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(drop_singular_frequency=1))  # type: ignore

    with pytest.raises(InvalidConfigurationError):
        validate_config(CyclicalTestConfig(exclude_zero_frequency="false"))  # type: ignore


# ---------------------------------------------------------------------------
# validate_cycle
# ---------------------------------------------------------------------------


def test_validate_cycle_accepts_valid_cycle():
    c = StochasticCycle(R=25, D=0.4)
    assert validate_cycle(c) is c


def test_validate_cycle_rejects_non_integer_R():
    c = StochasticCycle(R=25.5, D=0.4)  # type: ignore
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


def test_validate_cycle_rejects_non_positive_R():
    c = StochasticCycle(R=0, D=0.4)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


def test_validate_cycle_rejects_R_greater_or_equal_T_when_T_given():
    c = StochasticCycle(R=100, D=0.4)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c, T=100)


def test_validate_cycle_rejects_negative_D():
    c = StochasticCycle(R=10, D=-0.1)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


def test_validate_cycle_rejects_D_above_one():
    c = StochasticCycle(R=10, D=1.1)
    with pytest.raises(InvalidCycleError):
        validate_cycle(c)


# ---------------------------------------------------------------------------
# validate_cycles
# ---------------------------------------------------------------------------


def test_validate_cycles_accepts_single_cycle_tuple():
    cycles = (StochasticCycle(R=25, D=0.4),)
    result = validate_cycles(cycles)
    assert isinstance(result, tuple)
    assert len(result) == 1


def test_validate_cycles_rejects_empty_sequence():
    with pytest.raises(InvalidCycleError):
        validate_cycles([])


def test_validate_cycles_rejects_multi_cycle_when_not_allowed():
    cycles = [StochasticCycle(R=25, D=0.4), StochasticCycle(R=30, D=0.2)]
    with pytest.raises(InvalidCycleError):
        validate_cycles(cycles, allow_multi_cycle=False)


def test_validate_cycles_accepts_multi_cycle_when_allowed():
    cycles = [StochasticCycle(R=25, D=0.4), StochasticCycle(R=30, D=0.2)]
    result = validate_cycles(cycles, allow_multi_cycle=True)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# api integration
# ---------------------------------------------------------------------------


def test_run_test_validates_series_before_not_implemented():
    with pytest.raises(NotImplementedError):
        run_cyclical_fractional_test([1.0, 2.0, 3.0, 4.0, 5.0])


def test_run_test_invalid_series_raises_invalid_series_error():
    with pytest.raises(InvalidSeriesError):
        run_cyclical_fractional_test([1, np.nan, 3, 4, 5])


def test_run_test_invalid_config_raises_invalid_configuration_error():
    cfg = CyclicalTestConfig(top_k=0)
    with pytest.raises(InvalidConfigurationError):
        run_cyclical_fractional_test([1.0, 2.0, 3.0, 4.0, 5.0], config=cfg)
