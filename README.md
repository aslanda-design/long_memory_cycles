# cyclical-fractional-test

A Python package implementing a statistical test for **fractional cyclic long memory** in time series.

## What is this?

Given a time series Y(t), t=1,...,T, this package tests for the presence of fractional cyclic integration at candidate frequency R with fractional parameter D. The test combines:

- A deterministic basis of Chebyshev polynomials P_1(t),...,P_m(t).
- A stochastic fractional cyclic filter `(1 - 2cos(2πR/T)L + L²)^D`.
- A residual error specification: white noise (default), AR(1), or AR(2).
- A search over (R, D) candidates, returning the top-k combinations whose test statistic is closest to zero. By default the search over D is **adaptive** (a coarse grid refined locally around the best coarse value); a fixed grid is still available.

## Current status: Waves 0–16 complete (+ adaptive D search)

| Wave | Content | Status |
|------|---------|--------|
| 0 | Package skeleton, pyproject.toml, CI setup | ✅ |
| 1 | Config/result dataclasses, exceptions, full input validation | ✅ |
| 2 | Chebyshev deterministic design matrix | ✅ |
| 3 | FFT-based periodogram, peak detection | ✅ |
| 4 | R and D candidate grids, candidate iterator | ✅ |
| 5 | ψ(λ_j, R), XAA(R), single/multi-cycle dispatchers | ✅ |
| 6 | Cyclic frequency cosine mu, fractional filter coefficients C_{j,D}(mu), dispatcher | ✅ |
| 7 | Apply fractional cyclic filter to series and design matrix, dispatchers | ✅ |
| 8 | OLS regression on filtered data, residuals, RSS, time-domain variance VAR | ✅ |
| 9 | Residual periodogram, frequency-domain variance VAR*, dispatchers | ✅ |
| 10 | XA(R,D) single/multi-cycle and dispatcher | ✅ |
| 11 | TEST / TEST* statistic, candidate scoring, TopKSelector | ✅ |
| 12 | `evaluate_candidate` — full metrics for one (R,D) candidate | ✅ |
| 13 | `run_cyclical_fractional_test` — complete single-cycle API | ✅ |
| 14 | Diagnostics module: TestDiagnostics, PeriodogramSummary, VarianceComparison | ✅ |
| 15 | Mathematical documentation: background, mapping, data-flow diagram, implementation notes | ✅ |
| 16 | White-noise, AR(1), and AR(2) residual error specifications | ✅ |
| — | Adaptive coarse-to-fine D search (default; `d_search_strategy`) | ✅ |
| 17+ | Multi-cycle full support (psi_multi, xa_multi, xaa_multi) | 🔜 |

### Adaptive D search (default)

The statistical test is unchanged — only how candidate `D` values are chosen. For each candidate frequency `R`:

1. Evaluate a coarse grid of `D` values (default `[0.0, 0.1, …, 1.0]`).
2. Pick the best coarse `D` using the same scoring rule that ranks candidates (`abs(TEST)` or `abs(TEST*)`).
3. Evaluate a local fine grid around it (default step `0.01`, radius `0.09`, e.g. best `0.3` → `0.21…0.39`, clipped to `[0,1]`).
4. Keep the best `D` from both stages for that `R`.

Because the statistic is asymptotically normal for fixed `R`, the objective in `D` is locally well-behaved, so this two-stage search approximates a dense grid at much lower cost. Set `d_search_strategy="fixed_grid"` (with `d_grid`) to recover the original Cartesian-grid behavior; in adaptive mode `d_grid` is ignored.

## Documentation

Detailed reference documents live in [`docs/`](docs/):

- [Mathematical background](docs/mathematical_background.md) — full derivation of ψ, XAA, XA, TEST / TEST*, the fractional filter, and AR residual adjustments
- [Original test mapping](docs/original_test_mapping.md) — table mapping each step of the source document to source files and functions
- [Data flow diagram](docs/data_flow_diagram.md) — Mermaid flowcharts of the global pipeline and the per-candidate pipeline
- [Implementation notes](docs/implementation_notes.md) — 14 non-obvious decisions with rationale (index conventions, singularity handling, filter special cases, adaptive D search, ...)

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
    StochasticCycle,
    build_chebyshev_design,
    compute_document_periodogram,
    find_periodogram_peak,
    build_r_grid_around_peak,
    build_d_grid,
    build_default_d_coarse_grid,
    build_d_fine_grid,
    build_d_grid_for_strategy,
    candidate_iterator,
    compute_psi_single_cycle,
    compute_xaa_single_cycle,
    compute_mu,
    compute_fractional_coefficients_single_cycle,
    filter_response_and_design,
    fit_filtered_regression,
    estimate_ar_ols,
    compute_residual_periodogram,
    compute_ar_spectral_adjustment,
    compute_time_variance,
    compute_frequency_variance_dynamic,
    compute_xa_single_cycle,
    compute_xa_ar1_single_cycle,
    compute_xa_ar2_single_cycle,
    compute_xa_error_model,
    compute_xaa_ar1_single_cycle,
    compute_xaa_ar2_single_cycle,
    compute_xaa_error_model,
    compute_test_statistic,
    compute_test_star_statistic,
    TopKSelector,
    GridCandidateResult,
    evaluate_candidate,
    evaluate_r_with_adaptive_d,
)
import numpy as np

y = np.random.randn(200)
T = len(y)
config = CyclicalTestConfig(n_deterministic_cycles=4, r_window=10, top_k=5)

# Deterministic basis
X = build_chebyshev_design(T, config.n_deterministic_cycles)        # (200, 4)

# Periodogram and peak
_, periodogram = compute_document_periodogram(y)
r_peak = find_periodogram_peak(periodogram)                         # int

# Candidate grids
r_grid = build_r_grid_around_peak(r_peak, config.r_window, T)
d_grid = build_d_grid(config.d_grid)
candidates = candidate_iterator(r_grid, d_grid)                     # list of (StochasticCycle,)

# ψ and white-noise XAA for one candidate
cycle = candidates[0][0]
psi = compute_psi_single_cycle(T, cycle.R)
xaa = compute_xaa_single_cycle(psi)

# Fractional filter coefficients for one candidate (Wave 6)
mu = compute_mu(T, cycle.R)                                         # cos(2πR/T)
coeffs = compute_fractional_coefficients_single_cycle(T, cycle.R, cycle.D)  # shape (T,)

# Filter series and design matrix, then regress (Waves 7–8)
cycles = (StochasticCycle(R=cycle.R, D=cycle.D),)
y_f, X_f = filter_response_and_design(y, X, cycles, mode="single") # (T,), (T, 4)
reg = fit_filtered_regression(y_f, X_f)                             # RegressionResult

# Residual periodogram and variances (Wave 9)
lambdas_resid, I_resid = compute_residual_periodogram(reg.residuals)  # (T,), (T,)
var_time = compute_time_variance(reg.residuals)                     # float
var_freq = compute_frequency_variance_dynamic(                      # float
    I_resid, cycles, mode="single", drop_frequency=True
)

# AR nuisance parameters and spectral adjustment (Wave 16)
error_model = "ar1"                                                 # or "white_noise", "ar2"
ar_order = {"white_noise": 0, "ar1": 1, "ar2": 2}[error_model]
ar_coefficients = estimate_ar_ols(reg.residuals, ar_order)          # shape (ar_order,)
ar_adjustment = compute_ar_spectral_adjustment(lambdas_resid, ar_coefficients)
xaa = compute_xaa_error_model(psi, lambdas_resid, error_model, ar_coefficients)

# XA, TEST / TEST*, and top-k ranking (Waves 10–11 & 16)
xa = compute_xa_error_model(psi, I_resid, error_model, ar_adjustment)
test = compute_test_statistic(T, xa, xaa, var_time)                 # float
test_star = compute_test_star_statistic(T, xa, xaa, var_freq)       # float

selector = TopKSelector(k=3, statistic_mode="standard")
selector.consider(GridCandidateResult(
    cycles=cycles, test_value=test, test_star_value=test_star
))
best = selector.get_best()                                          # closest |TEST| to 0

# Full pipeline in one call (Waves 13 & 16) — adaptive D search by default
from cyclical_fractional_test import CyclicalTestConfig, run_cyclical_fractional_test
import numpy as np

result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        n_deterministic_cycles=4,
        r_window=5,
        top_k=3,
        stochastic_cycle_mode="single",
        error_model="ar1",
        # d_search_strategy defaults to "adaptive"; tune the search with:
        # d_coarse_grid=None,   # None → [0.0, 0.1, ..., 1.0]
        # d_fine_step=0.01,
        # d_fine_radius=0.09,
    ),
)
print(result.best_result.cycles)    # best (R, D)
print(result.best_result.test_value)
print(result.best_result.ar_coefficients)  # estimated AR nuisance parameters
print(result.top_k_results)         # top-3 candidates (one per frequency R)

# Recover the original fixed-grid behavior explicitly
fixed = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(n_deterministic_cycles=4, r_window=5, top_k=3),
    d_search_strategy="fixed_grid",
    d_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
)

# Diagnostics (Wave 14)
diag = result.diagnostics
print(diag.n_candidates_evaluated)          # distinct (R,D) pairs evaluated (coarse + fine − reused)
print(diag.d_search_strategy)               # "adaptive"
print(diag.best_coarse_d_per_r)             # best coarse D for each R
print(diag.final_d_per_r)                   # refined D for each R
print(diag.r_peak)                          # dominant frequency index
print(diag.periodogram_summary.peak_value)  # periodogram value at R*
```

`compute_xaa_error_model` and `compute_xa_error_model` dispatch independently
over `error_model` and `stochastic_cycle_mode`. The AR(1)/AR(2) multi-cycle
routes are represented explicitly, but still raise `NotImplementedError`
until the pending multi-cycle ψ, XAA, and XA mathematics is implemented.

## Package structure

```
src/cyclical_fractional_test/
├── __init__.py       # public API exports
├── api.py            # run_cyclical_fractional_test entry point
├── config.py         # CyclicalTestConfig dataclass
├── results.py        # StochasticCycle, GridCandidateResult, AdaptiveDSearchResult, CyclicalFractionalTestResult
├── exceptions.py     # CyclicalFractionalTestError hierarchy
├── validation.py     # defensive input validation layer
├── chebyshev.py      # Chebyshev polynomial design matrix      [Wave 2]
├── spectral.py       # periodogram, ψ, XAA, VAR*, XA, AR adjustments [Waves 3, 5, 9, 10 & 16]
├── grid.py           # R/D grids, candidate iterator, adaptive coarse/fine D grids [Wave 4 +]
├── filters.py        # mu, coefficients, filter application     [Waves 6 & 7]
├── regression.py     # OLS regression, residuals, VAR, AR OLS  [Waves 8 & 16]
├── scoring.py        # TEST / TEST* statistics, TopKSelector    [Wave 11]
├── evaluation.py     # evaluate_candidate, evaluate_r_with_adaptive_d [Wave 12 +]
└── diagnostics.py    # TestDiagnostics, PeriodogramSummary, VarianceComparison [Wave 14]

docs/
├── mathematical_background.md   # full derivation of the statistical test
├── original_test_mapping.md     # step-by-step mapping to source files
├── data_flow_diagram.md         # Mermaid pipeline diagrams
└── implementation_notes.md      # 14 non-obvious implementation decisions

tests/
├── test_config.py
├── test_results.py
├── test_validation.py
├── test_chebyshev.py
├── test_periodogram.py
├── test_grid.py
├── test_psi_xaa_single_cycle.py
├── test_psi_xaa_dynamic_dispatch.py
├── test_fractional_coefficients.py
├── test_fractional_coefficients_dispatch.py
├── test_apply_filter.py
├── test_filter_response_and_design.py
├── test_filter_dispatch.py
├── test_regression.py
├── test_time_variance.py
├── test_residual_periodogram.py
├── test_frequency_variance.py
├── test_frequency_variance_dispatch.py
├── test_partial_pipeline_waves_7_9.py
├── test_xa_single_cycle.py
├── test_xa_dispatch.py
├── test_scoring.py
├── test_top_k_selector.py
├── test_partial_pipeline_waves_10_11.py
├── test_evaluate_candidate_single_cycle.py
├── test_evaluate_candidate_dispatch.py
├── test_api_single_cycle.py
├── test_api_top_k.py
├── test_api_config_overrides.py
├── test_api_invalid_inputs.py
├── test_full_single_cycle_pipeline.py
├── test_diagnostics.py
├── test_api_diagnostics.py
└── test_docs_exist.py
```
