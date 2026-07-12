# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is mandatory to update this file with the latest version of the repository when changes are done on it.

## Commands

```bash
# Install in editable mode (required before running tests)
python3 -m pip install -e ".[dev]"

# Install development and documentation tooling
python3 -m pip install -e ".[dev,docs]"

# Run all tests
python3 -m pytest

# Run a single test file
python3 -m pytest tests/test_validation.py

# Run a single test by name
python3 -m pytest tests/test_validation.py::test_validate_series_rejects_nan

# Run with coverage
python3 -m pytest --cov=cyclical_fractional_test --cov-report=term-missing

# Preview docs
python3 -m mkdocs serve

# Build and validate package artifacts
python3 -m build
python3 -m twine check dist/*
```

## Architecture

This package implements a statistical test for **fractional cyclic long memory** in time series. Waves 0–18 are complete, plus an adaptive coarse-to-fine `D` search layered on top. The full pipeline (`run_cyclical_fractional_test`) is functional for single-cycle and multi-cycle modes, includes diagnostics plus optional AR(1)/AR(2) residual adjustments, and is wrapped by a scikit-learn-style `CyclicalFractionalModel` (`fit`/`predict`/`predict_interval`) for series reconstruction and forecasting.

### Module responsibilities

| Module | Role |
|--------|------|
| `config.py` | `CyclicalTestConfig` dataclass — all tunable knobs (incl. the adaptive-`D`-search fields `d_search_strategy`, `d_coarse_grid`, `d_fine_step`, `d_fine_radius`), no logic |
| `results.py` | Result dataclasses: `StochasticCycle`, `GridCandidateResult`, `AdaptiveDSearchResult` (exposes `all_results`, every candidate evaluated for its R), `CyclicalFractionalTestResult` (exposes optional `under_threshold_results`); candidate results expose the residual error model and estimated AR nuisance coefficients |
| `exceptions.py` | Exception hierarchy rooted at `CyclicalFractionalTestError` |
| `validation.py` | Every public validator function; **no math, only defensive checks** |
| `chebyshev.py` | `build_single_chebyshev_polynomial`, `build_chebyshev_design`, plus `evaluate_single_chebyshev_polynomial` / `build_chebyshev_design_at` (same basis at arbitrary t with the training length held fixed, used to extrapolate beyond T) — deterministic basis |
| `spectral.py` | FFT periodogram, peak detection, ψ(λ_j, R), XAA(R), residual periodogram, VAR*, XA, AR spectral adjustments, dispatchers |
| `grid.py` | R and D candidate grids, Cartesian candidate builder, `candidate_iterator` dispatcher, adaptive coarse/fine D grids (`build_default_d_coarse_grid`, `build_d_fine_grid`, `build_d_grid_for_strategy`) |
| `filters.py` | `compute_mu`, `compute_fractional_coefficients_from_mu` (arbitrary length, fixed mu — used by prediction), `compute_fractional_coefficients_*`, filter application functions, dispatchers |
| `regression.py` | `RegressionResult`, `fit_filtered_regression`, `compute_residuals`, `compute_time_variance`, `estimate_ar_ols`, `estimate_innovation_variance` |
| `scoring.py` | `compute_test_statistic`, `compute_test_star_statistic`, `score_candidate`, `TopKSelector` |
| `evaluation.py` | `evaluate_candidate` — orchestrates Waves 5–11 and 16 for one (R,D) candidate; `evaluate_r_with_adaptive_d` — coarse-to-fine D search for one R |
| `diagnostics.py` | `TestDiagnostics`, `PeriodogramSummary`, `VarianceComparison`, `summarize_periodogram`, `compare_variance_definitions`, `build_candidate_diagnostics`, `build_test_diagnostics` |
| `api.py` | `run_cyclical_fractional_test` — full single-cycle and multi-cycle pipeline: validate → grid → evaluate → rank → diagnostics |
| `prediction.py` | `reconstruct_in_sample`, `forecast_ar`, `forecast_out_of_sample`, `compute_ma_weights` — series reconstruction/forecast math; no test logic, no grids |
| `model.py` | `CyclicalFractionalModel` — scikit-learn-style estimator (`fit`/`predict`/`predict_interval`) wrapping the pipeline |
| `__init__.py` | Re-exports the full public surface; users import only from here |

### Data flow (current state)

```
run_cyclical_fractional_test(y, config)
    └── validate_series(y)                      → np.ndarray[float, 1-D]
    └── validate_config(config)                 → CyclicalTestConfig (unchanged)
    └── build_chebyshev_design(T, n_cycles)     → (T, m) design matrix
    └── compute_document_periodogram(y)         → (lambdas_y, I_y)
    └── find_periodogram_peak(I_y)              → R*
    └── build_r_grid_around_peak(R*, w, T)      → r_candidates
    └── build_d_grid_for_strategy(config)        → reported D grid (coarse seed or fixed grid)
    └── if d_search_strategy == "adaptive" (default):
        └── for each R in r_candidates:
            └── evaluate_r_with_adaptive_d(y, X, R, config) → AdaptiveDSearchResult
                └── evaluate coarse D grid, pick best by score_candidate
                └── evaluate fine D grid around best coarse D (reuse overlap)
            └── TopKSelector.consider(search.best_result)
    └── else (d_search_strategy == "fixed_grid"):
        └── for each (R, D) in Cartesian grid:
            └── evaluate_candidate(y, X, cycles, config) → GridCandidateResult
            └── TopKSelector.consider(result)
    └── build_test_diagnostics(...)             → TestDiagnostics
    └── CyclicalFractionalTestResult            [best_result, top_k_results, diagnostics, ...]

Available building blocks:
    build_chebyshev_design(T, n_cycles)         → (T, n_cycles) design matrix
    compute_document_periodogram(y)             → (lambdas, periodogram) via np.fft.fft
    find_periodogram_peak(periodogram)          → R*  (int)
    build_r_grid_around_peak(R*, w, T)          → R candidates array
    build_d_grid()                              → D candidates array [0.0..1.0]
    build_default_d_coarse_grid()               → coarse D grid [0.0..1.0] (11 values)
    build_d_fine_grid(center, radius, step)     → local fine D grid clipped to [0,1]
    build_d_grid_for_strategy(config)           → coarse grid (adaptive) or fixed grid
    candidate_iterator(r_grid, d_grid)          → list of (StochasticCycle(...),) tuples
    compute_psi_single_cycle(T, R)              → psi array shape (T,)
    compute_xaa_single_cycle(psi)               → float
    compute_mu(T, R)                            → cos(2πR/T) float
    compute_fractional_coefficients_single_cycle(T, R, D) → coeffs array shape (T,)
    filter_response_and_design(y, X, cycles)    → (y_filtered, X_filtered)
    fit_filtered_regression(y_f, X_f)           → RegressionResult
    compute_residual_periodogram(residuals)     → (lambdas, I_residuals)
    compute_time_variance(residuals)            → float  [mean of squared residuals]
    estimate_ar_ols(residuals, order)            → AR nuisance coefficients shape (order,)
    compute_ar_spectral_adjustment(lambdas, coefficients) → g(λ_j; φ_hat)
    compute_xaa_error_model(psi, lambdas, error_model, coefficients, mode) → float
    compute_xaa_ar1_single_cycle(psi, lambdas, coefficients) → float
    compute_xaa_ar2_single_cycle(psi, lambdas, coefficients) → float
    compute_frequency_variance_dynamic(I, cycles, mode) → float  [VAR*]
    compute_xa_single_cycle(psi, I_resid)       → float  [XA(R,D)]
    compute_xa_error_model(psi, I_resid, error_model, g, mode) → float
    compute_xa_ar1_single_cycle(psi, I_resid, g) → float
    compute_xa_ar2_single_cycle(psi, I_resid, g) → float
    compute_xa_dynamic(psi, I_resid, cycles, mode) → float
    compute_test_statistic(T, xa, xaa, var_t)   → float  [TEST]
    compute_test_star_statistic(T, xa, xaa, var_f) → float  [TEST*]
    score_candidate(candidate, statistic_mode)  → float  [|TEST| or |TEST*|]
    TopKSelector(k, statistic_mode)             → keeps the k candidates closest to 0
    evaluate_candidate(y, X, cycles, config)    → GridCandidateResult  [one candidate]
    evaluate_r_with_adaptive_d(y, X, R, config) → AdaptiveDSearchResult  [coarse→fine D for one R]
    summarize_periodogram(lambdas, periodogram) → PeriodogramSummary
    compare_variance_definitions(var_t, var_f)  → VarianceComparison
    build_candidate_diagnostics(candidate)      → dict
    build_test_diagnostics(...)                 → TestDiagnostics
    run_cyclical_fractional_test(y, config)     → CyclicalFractionalTestResult  [full pipeline]
```

### evaluate_candidate (Wave 12)

`evaluate_candidate` orchestrates the full per-candidate pipeline (Waves 5–11 and 16) for a single `cycles` tuple. It does not build grids, find R*, score, or rank. The data flow inside it is:

```
compute_psi_dynamic
→ filter_response_and_design → fit_filtered_regression
→ compute_residual_periodogram → compute_time_variance
→ estimate_ar_ols → compute_ar_spectral_adjustment
→ compute_xaa_error_model → compute_xa_error_model
→ compute_frequency_variance_dynamic
→ compute_test_statistic → compute_test_star_statistic
→ GridCandidateResult
```

### run_cyclical_fractional_test (Wave 13+)

`stochastic_cycle_mode="single"` and `"multi_cycle"` are supported by the top-level pipeline. `"multi_peak_single_cycle"` remains a lower-level grid/dispatcher mode and raises `NotImplementedError` in `run_cyclical_fractional_test`. kwargs override individual config fields (using `dataclasses.replace`). `error_model` accepts `"white_noise"` (default), `"ar1"`, or `"ar2"`. The result always contains `top_k_results` (ordered ascending by |TEST| or |TEST* according to `statistic_mode`) and `n_candidates_evaluated`.

The optional `threshold` parameter (a positive float, kept separate from the config so it is never passed through `dataclasses.replace`) turns on `result.under_threshold_results`: a `dict[tuple[int, ...], list[GridCandidateResult]]` keyed by the full frequency tuple, `(R,)` for single-cycle candidates and `(R1, R2, ...)` for multi-cycle candidates. Each key maps to evaluated candidates whose statistic score — `score_candidate(candidate, config.statistic_mode)`, i.e. `|TEST|` or `|TEST*|` — is strictly below the threshold. Keys are sorted ascending and each list is ordered best-first (smallest score). It draws from **every** evaluated candidate: all fixed-grid points, or all coarse+fine candidates in adaptive mode. When `threshold` is `None` (default) the object stays `None`. It is descriptive only and never feeds back into `best_result`, `top_k_results`, or any statistic.

### Adaptive D search

This is a **search strategy only**; the test statistic, ψ, XAA, XA, VAR/VAR*, TEST/TEST* are all unchanged. For a fixed R the statistic is asymptotically normal, so the objective in D is locally well-behaved enough that a coarse-to-fine search approximates the dense-grid minimiser at a fraction of the cost.

`config.d_search_strategy` selects the behavior (default `"adaptive"`):

- `"adaptive"` — for each R, `evaluate_r_with_adaptive_d` evaluates a coarse D grid (`d_coarse_grid` or the default `[0.0, 0.1, …, 1.0]`), picks the best coarse D via `score_candidate`, then evaluates a local fine grid `build_d_fine_grid(best, d_fine_radius, d_fine_step)` (default step `0.01`, radius `0.09`, clipped to `[0,1]`). The best of all evaluated candidates for that R is kept. Each R contributes one candidate to the `TopKSelector`, so `top_k` ranks across frequencies. `config.d_grid` is **ignored** in this mode.
- `"fixed_grid"` — the original behavior: a Cartesian `(R, D)` grid built from `config.d_grid` (or the default 11-point grid), every candidate evaluated and ranked.

Reuse: the fine grid is keyed by D rounded to 12 decimals, so the best coarse D that reappears in the fine window is not recomputed. `n_candidates_evaluated` counts the distinct `(R, D)` pairs actually evaluated (coarse + fine − reused), summed over R. Grid values are rounded to 12 decimals for stable floating point.

`run_cyclical_fractional_test(...).d_grid` and `diagnostics.d_grid_count` report the coarse grid in adaptive mode and the fixed grid in fixed-grid mode. Adaptive-only diagnostics (`d_search_strategy`, `d_fine_step`, `d_fine_radius`, `best_coarse_d_per_r`, `final_d_per_r`, `n_coarse_evaluations`, `n_fine_evaluations`) are descriptive and never feed back into the computation.

### Residual error models (Wave 16)

`error_model="white_noise"` preserves the original formulas and remains the default. For `"ar1"` and `"ar2"`, `estimate_ar_ols` estimates nuisance coefficients from each candidate's filtered-regression residuals. `compute_ar_spectral_adjustment`, `compute_xaa_error_model`, and `compute_xa_error_model` then apply the corresponding spectral weighting before TEST / TEST* are computed.

The residual error model and `stochastic_cycle_mode` are independent axes. AR(1) and AR(2) therefore dispatch across both single-cycle and aggregate multi-cycle paths. The multi-cycle paths use the aggregate ψ/XAA/XA implementations before applying the residual spectral adjustment.

Each `GridCandidateResult` exposes `error_model` and `ar_coefficients`. For white noise, `ar_coefficients == ()`.

### Chebyshev polynomials (Wave 2)

`P_0(t) = 1`, `P_k(t) = 2cos(kπ(t−0.5)/T)` for k ≥ 1, t = 1,...,T.

`build_chebyshev_design` generates exactly `n_cycles` columns (no zero-padded extras).

### Periodogram convention (Wave 3)

`compute_document_periodogram` uses **NumPy 0-based indexing** (j = 0,...,T-1) internally, matching `np.fft.fft`. The formula `I(λ_j) = |FFT(x)_j|² / (2πT)` is mathematically equivalent to the document's sin/cos sum because `|·|²` eliminates the phase shift from index offset. No explicit sin/cos loops.

### ψ singularity handling (Wave 5)

`ψ(λ_j, R) = log(|2(cos(λ_j) − cos(λ_R))|)` is singular at j = R and at j = T−R.
- `j = R` is *exactly* singular (same arithmetic expression → exact cancellation → log(0) = −∞).
- `j = T−R` is *approximately* singular (floating-point cos imprecision makes the argument ≈ 0 but not exactly 0).

With `drop_singular_frequency=True` (default), both positions are explicitly set to 0.0.

### Stochastic cycles design

Cycles are always represented as `tuple[StochasticCycle, ...]`, even in the single-cycle case. Each `StochasticCycle(R, D)` represents the filter factor `(1 − 2cos(2πR/T)L + L²)^D`. The three `stochastic_cycle_mode` values gate multi-cycle logic:

- `"single"` — one cycle
- `"multi_peak_single_cycle"` — multiple periodogram peaks fed as separate single-cycle candidates; `candidate_iterator` and dispatchers treat this identically to `"single"` at the per-candidate level
- `"multi_cycle"` — multiple simultaneous cycles represented by one candidate tuple and evaluated with aggregate ψ/XAA/XA quantities

### Dispatcher pattern (Wave 5)

`compute_psi_dynamic`, `compute_xaa_dynamic`, `compute_xa_dynamic`, and `compute_frequency_variance_dynamic` select the correct implementation based on `stochastic_cycle_mode` / `mode`. `compute_xaa_error_model` and `compute_xa_error_model` dispatch across both the residual error specification and cycle mode. Their AR branches delegate to `compute_xaa_ar*_dynamic` and `compute_xa_ar*_dynamic`, which distinguish single-cycle implementations from multi-cycle implementations. Add new implementations in `spectral.py` and register them in these dispatchers; do not add `if/else` chains elsewhere.

### Scoring and ranking (Wave 11)

- `compute_test_statistic` and `compute_test_star_statistic` preserve the sign of XA. The absolute value is taken only in `score_candidate`.
- `score_candidate` accepts `statistic_mode` aliases: `"standard"`/`"test"` → `abs(test_value)`, `"frequency"`/`"test_star"` → `abs(test_star_value)`.
- `TopKSelector` orders candidates by ascending score (closest to zero is best); `get_best()` returns `None` when no candidate has been considered.

### Validation style rule

Each function in a math/logic module (`chebyshev.py`, `spectral.py`, `grid.py`, and any future wave module) must follow this layout:

```python
def my_function(args):
    _validate_my_function(args)   # ← one call, nothing else
    # actual logic here
```

All error-checking for that function is extracted into a single private helper `_validate_<function_name>(args)`. These helpers are grouped together at the **end of the file**, separated by a comment block:

```python
# ---------------------------------------------------------------------------
# Validators
# In this section we define the input validation for each of the functions of
# this script, this way we ensure that in case of error, we know  the exact
# reason why the process failed.
# ---------------------------------------------------------------------------
```

**Rationale:** this is an academic codebase whose primary goal is to demonstrate the statistical test. Keeping the logic visible and uncluttered is more important than defensive robustness.

**Exceptions:** `validation.py` (it is itself a pure validation layer), `api.py` (already a thin entry point), dispatcher functions where the mode check IS the logic, and placeholder functions that only raise `NotImplementedError`.

### Validation conventions

- `validate_series` / `validate_cycle` / `validate_cycles` raise their own typed exceptions directly.
- `validate_config` is the aggregate entry point: calls every field validator, raises `InvalidConfigurationError` on the first failure.
- `validate_cycle` and `validate_cycles` use a **local import** of `StochasticCycle` from `results.py` to avoid the circular dependency `results → config → validation → results`.
- `bool` is a subclass of `int` in Python; all integer validators explicitly reject `bool` with `isinstance(n, bool)` before the `isinstance(n, int)` check.
- `d_grid=None` is valid — means "use the default grid `[0.0, 0.1, ..., 1.0]` at runtime."

## Comment Style

Avoid long docstrings or comments that feel overly templated. For classes, the docstring should only describe the general purpose of the class and keep any relevant formulas if needed.

Do not list every attribute inside the class docstring. Instead, add a short inline comment next to each attribute declaration, for example:

```python
R: int  # Candidate index for the cyclic frequency.
D: float  # Fractional integration parameter for this cycle.
```

### README update rule

**Whenever a new wave is introduced, update `README.md` before closing the task.** Specifically:

- Advance the wave's row in the status table from 🔜 to ✅.
- Add any new public building blocks to the usage example if they are meaningful to show standalone.
- Update the package structure tree if new source files were added.
- Keep the "Current status" headline in sync with the highest completed wave number.

### Diagnostics (Wave 14)

`diagnostics.py` is computed **after** all candidate evaluations; it reads from already-computed values and never feeds back into the statistical computation. Adding or removing diagnostics cannot change `best_result` or `top_k_results`. The module is exempt from the validation-style rule because it contains no mathematical logic.

Key invariant: `TestDiagnostics.n_failed_candidates` is always 0 for the current implementation (Policy A — numerical errors surface as exceptions, not silent skips).

### Documentation and publishing

| File | Contents |
|------|----------|
| `docs/index.md` | Documentation landing page for MkDocs |
| `docs/quickstart.md` | Install and first-use examples |
| `docs/api_reference.md` | Public API overview |
| `docs/mathematical_background.md` | Full derivation: Chebyshev basis, periodogram, ψ, XAA, filter, regression, VAR / VAR*, XA, AR residual adjustments, TEST / TEST* |
| `docs/original_test_mapping.md` | Table mapping each step of the original statistical document to source files and functions |
| `docs/data_flow_diagram.md` | Mermaid diagrams of the global pipeline and the per-candidate pipeline |
| `docs/implementation_notes.md` | 14 non-obvious implementation decisions with rationale |
| `docs/development.md` | Local setup, tests, docs, and build checks |
| `docs/publishing.md` | PyPI release workflow and checklist |

Packaging files now include `pyproject.toml` metadata, `LICENSE`, `MANIFEST.in`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff`, `mkdocs.yml`, `src/cyclical_fractional_test/py.typed`, and GitHub Actions workflows for CI and PyPI publishing. `MANIFEST.in` intentionally excludes local data, figures, models, and notebooks from source distributions.

### Sklearn-style model & prediction (Wave 18)

`CyclicalFractionalModel` (`model.py`) is a thin scikit-learn-style wrapper. `fit(y)` runs `run_cyclical_fractional_test`, then stores the selected model on `_`-suffixed attributes (`cycles_`, `R_`, `D_`, `betas_`, `error_model_`, `ar_coefficients_`, `innovation_variance_`, `residuals_`, `X_train_`, `y_train_`, `n_train_`, `result_`). `predict(n)` returns ŷ for t = 1, …, n; `predict_interval(n, alpha)` returns `(lower, upper)`. `predict` before `fit` raises `NotFittedError`.

The prediction math lives in `prediction.py` (pure arrays, no config, no grids) and rests on the generative model

```
Y_t = X_t·β̂ + S_t,   filter_{R,D}(S)_t = ε̂_t,   ε̂_t = Σ_i φ_i·ε̂_{t-i} + e_t
```

where ε̂ are exactly `best_result.residuals` (because the filter is linear, `Y_D − X_D β̂ = filter(Y − X β̂)`), φ are `best_result.ar_coefficients`, and β̂ are `best_result.betas`.

```
model.fit(y)
    └── run_cyclical_fractional_test(y, config) → best_result
    └── estimate_innovation_variance(residuals, ar_coefficients) → σ̂²
model.predict(n)
    └── reconstruct_in_sample(...)                         → ŷ_{1..T} (one-step conditional mean)
    └── if n > T: build_chebyshev_design_at(t>T, T_ref=T)  → X_future
                  forecast_out_of_sample(...)              → ŷ_{T+1..n}
model.predict_interval(n, alpha)
    └── compute_ma_weights(...) → forecast-error growth → σ̂·√Σψ²
```

Key facts / invariants:
- **In-sample reconstruction is one-step-ahead.** `Y_t − ŷ_t` equals the AR innovation `e_t` exactly (machine precision); this is the verification test in `test_prediction.py`.
- **AR coefficients are already estimated** by `estimate_ar_ols` (nuisance for the spectral statistic) and reused for prediction; Wave 18 only adds the innovation variance σ̂² and the forecast recursions that consume φ. No new AR estimator.
- **Chebyshev extrapolation** for t > T evaluates the same `P_k(t) = 2cos(kπ(t−0.5)/T_ref)` with `T_ref` fixed at the training length. This is documented extrapolation of a basis designed for [1, T].
- **Multi-cycle prediction** reuses the same path: `_combined_coefficients` convolves the per-cycle inverse coefficients (via `compute_fractional_coefficients_from_mu`, fixed mu) and `residuals` are already the full multi-cycle filtered residuals.
- `prediction.py` follows the validation-style rule (one `_validate_*` call per function, validators grouped at the end). `model.py` is a thin entry point (like `api.py`) and is exempt.
- Prediction is read-only with respect to the test: it never changes `best_result`, `top_k_results`, or any statistic.

### What NOT to add in Wave 19+

Keep `prediction.py` limited to reconstruction/forecast array math and `model.py` to the sklearn-style entry point — no statistic, grid, or scoring logic in either. Do not put mathematical computation into `validation.py` or `config.py`. Keep `spectral.py` limited to periodogram + ψ + XAA + VAR* + XA + AR spectral adjustments; keep `filters.py` limited to coefficients and filter application; keep `regression.py` limited to OLS + residuals + time variance + AR nuisance estimation + innovation variance; keep `scoring.py` limited to TEST/TEST* + ranking; keep `evaluation.py` limited to per-candidate orchestration; keep `diagnostics.py` limited to diagnostics summaries; keep `api.py` limited to top-level orchestration.
