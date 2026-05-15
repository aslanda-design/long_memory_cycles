# cyclical-fractional-test

A Python package implementing a statistical test for **fractional cyclic long memory** in time series.

## What is this?

Given a time series Y(t), t=1,...,T, this package tests for the presence of fractional cyclic integration at candidate frequency R with fractional parameter D. The test combines:

- A deterministic basis of Chebyshev polynomials P_1(t),...,P_m(t).
- A stochastic fractional cyclic filter `(1 - 2cos(2πR/T)L + L²)^D`.
- A grid search over (R, D) candidates, returning the top-k combinations whose test statistic is closest to zero.

## Current status: Waves 0–5 complete

| Wave | Content | Status |
|------|---------|--------|
| 0 | Package skeleton, pyproject.toml, CI setup | ✅ |
| 1 | Config/result dataclasses, exceptions, full input validation | ✅ |
| 2 | Chebyshev deterministic design matrix | ✅ |
| 3 | FFT-based periodogram, peak detection | ✅ |
| 4 | R and D candidate grids, candidate iterator | ✅ |
| 5 | ψ(λ_j, R), XAA(R), single/multi-cycle dispatchers | ✅ |
| 6+ | Fractional filtering, regression, XA, TEST statistic, top-k selection | 🔜 |

The entry point `run_cyclical_fractional_test` validates inputs and then raises `NotImplementedError` — the building blocks from Waves 2–5 are tested independently and will be wired together in Wave 6.

## Installation

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Building blocks available today

```python
from cyclical_fractional_test import (
    CyclicalTestConfig,
    build_chebyshev_design,
    compute_document_periodogram,
    find_periodogram_peak,
    build_r_grid_around_peak,
    build_d_grid,
    candidate_iterator,
    compute_psi_single_cycle,
    compute_xaa_single_cycle,
    StochasticCycle,
)
import numpy as np

y = np.random.randn(200)
T = len(y)
config = CyclicalTestConfig(n_deterministic_cycles=4, r_window=10, top_k=5)

# Deterministic basis
X = build_chebyshev_design(T, config.n_deterministic_cycles)        # (200, 4)

# Periodogram and peak
_, periodogram = compute_document_periodogram(y)
r_star = find_periodogram_peak(periodogram)                         # int

# Candidate grids
r_grid = build_r_grid_around_peak(r_star, config.r_window, T)
d_grid = build_d_grid(config.d_grid)
candidates = candidate_iterator(r_grid, d_grid)                     # list of (StochasticCycle,)

# ψ and XAA for one candidate
cycle = candidates[0][0]
psi = compute_psi_single_cycle(T, cycle.R)
xaa = compute_xaa_single_cycle(psi)
```

## Package structure

```
src/cyclical_fractional_test/
├── __init__.py       # public API exports
├── api.py            # run_cyclical_fractional_test entry point
├── config.py         # CyclicalTestConfig dataclass
├── results.py        # StochasticCycle, GridCandidateResult, CyclicalFractionalTestResult
├── exceptions.py     # CyclicalFractionalTestError hierarchy
├── validation.py     # defensive input validation layer
├── chebyshev.py      # Chebyshev polynomial design matrix  [Wave 2]
├── spectral.py       # periodogram, ψ, XAA, dispatchers    [Waves 3 & 5]
└── grid.py           # R/D grids, candidate iterator        [Wave 4]

tests/
├── test_config.py
├── test_results.py
├── test_validation.py
├── test_chebyshev.py
├── test_periodogram.py
├── test_grid.py
├── test_psi_xaa_single_cycle.py
└── test_psi_xaa_dynamic_dispatch.py
```
