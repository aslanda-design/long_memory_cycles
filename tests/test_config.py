import pytest
from cyclical_fractional_test import CyclicalTestConfig


def test_default_config_can_be_created():
    cfg = CyclicalTestConfig()
    assert cfg.n_deterministic_cycles == 4
    assert cfg.include_intercept is False
    assert cfg.d_grid is None
    assert cfg.r_window == 10
    assert cfg.top_k == 1
    assert cfg.variance_mode == "time"
    assert cfg.statistic_mode == "test"
    assert cfg.stochastic_cycle_mode == "single"
    assert cfg.drop_singular_frequency is True
    assert cfg.exclude_zero_frequency is True
    assert cfg.return_residuals_for_top_k is True


def test_custom_config_can_be_created():
    import numpy as np

    grid = np.array([0.0, 0.5, 1.0])
    cfg = CyclicalTestConfig(
        n_deterministic_cycles=2,
        include_intercept=True,
        d_grid=grid,
        r_window=5,
        top_k=3,
        variance_mode="frequency",
        statistic_mode="test_star",
        stochastic_cycle_mode="multi_cycle",
        drop_singular_frequency=False,
        exclude_zero_frequency=False,
        return_residuals_for_top_k=False,
    )
    assert cfg.n_deterministic_cycles == 2
    assert cfg.include_intercept is True
    assert cfg.r_window == 5
    assert cfg.top_k == 3
    assert cfg.variance_mode == "frequency"
    assert cfg.statistic_mode == "test_star"
    assert cfg.stochastic_cycle_mode == "multi_cycle"
    assert cfg.drop_singular_frequency is False
    assert cfg.exclude_zero_frequency is False
    assert cfg.return_residuals_for_top_k is False


def test_config_does_not_use_mutable_default_for_d_grid():
    cfg1 = CyclicalTestConfig()
    cfg2 = CyclicalTestConfig()
    assert cfg1.d_grid is None
    assert cfg2.d_grid is None
    # Mutating one should not affect the other (they are both None, but this
    # confirms field isolation if d_grid were a list).
    cfg1.d_grid = [0.1, 0.2]
    assert cfg2.d_grid is None
