import pytest
from cyclical_fractional_test import (
    CyclicalFractionalTestResult,
    CyclicalTestConfig,
    GridCandidateResult,
    StochasticCycle,
)


def test_stochastic_cycle_can_be_created():
    cycle = StochasticCycle(R=25, D=0.4)
    assert cycle.R == 25
    assert cycle.D == 0.4


def test_grid_candidate_result_can_be_created_minimally():
    cycle = StochasticCycle(R=10, D=0.3)
    result = GridCandidateResult(cycles=(cycle,))
    assert result.test_value is None
    assert result.test_star_value is None
    assert result.betas is None
    assert result.residuals is None
    assert result.xa is None
    assert result.xaa is None


def test_final_result_can_be_created_minimally():
    result = CyclicalFractionalTestResult()
    assert result.best_result is None
    assert result.top_k_results == []
    assert result.r_star is None
    assert result.r_candidates is None
    assert result.config is None
    assert result.diagnostics == {}


def test_cycles_are_stored_as_sequence():
    c1 = StochasticCycle(R=10, D=0.2)
    c2 = StochasticCycle(R=20, D=0.5)
    result = GridCandidateResult(cycles=(c1, c2))
    assert len(result.cycles) == 2
    assert result.cycles[0].R == 10
    assert result.cycles[1].R == 20


def test_final_result_stores_config():
    cfg = CyclicalTestConfig(top_k=3)
    result = CyclicalFractionalTestResult(config=cfg)
    assert result.config is cfg
    assert result.config.top_k == 3
