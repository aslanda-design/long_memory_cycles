# Mapping: Original Test Document → Implementation

This table maps each step of the original statistical test to the concrete files and functions in `cyclical_fractional_test`.

---

| # | Step description | Math | File | Functions / Classes | Status |
|---|---|---|---|---|---|
| 1 | Import / validate Y(t) | Y(t), t=1,…,T | `validation.py`, `api.py` | `validate_series`, `run_cyclical_fractional_test` | ✅ Implemented |
| 2 | Build Chebyshev basis | P_k(t) = 2cos(kπ(t−0.5)/T) | `chebyshev.py` | `build_chebyshev_design`, `build_single_chebyshev_polynomial` | ✅ Implemented |
| 3 | Compute periodogram of Y(t) | I(λ_j) = \|FFT(Y)_j\|²/(2πT) | `spectral.py` | `compute_document_periodogram` | ✅ Implemented |
| 3b | Locate R* | R* = argmax I(λ_j), usually j≥1 unless zero frequency is enabled | `spectral.py` | `find_periodogram_peak`, `find_top_periodogram_peaks` | ✅ Implemented |
| 4 | Build R candidate grid | R ∈ [R*−w, R*+w] ∩ [R_min, T−1] | `grid.py` | `build_r_grid_around_peak` | ✅ Implemented |
| 5a | Compute ψ(λ_j, R) | ψ = log\|2(cosλ_j − cosλ_R)\| | `spectral.py` | `compute_psi_single_cycle`, `compute_psi_dynamic` | ✅ Implemented |
| 5b | Compute XAA(R) | XAA = (2/T) Σ ψ² | `spectral.py` | `compute_xaa_single_cycle`, `compute_xaa_dynamic` | ✅ Implemented |
| 5c | Adjust XAA for residual AR errors | XAA_AR1 or XAA_AR2 projection correction | `spectral.py` | `compute_xaa_error_model`, `compute_xaa_ar1_dynamic`, `compute_xaa_ar1_single_cycle`, `compute_xaa_ar1_multi_cycle`, `compute_xaa_ar2_dynamic`, `compute_xaa_ar2_single_cycle`, `compute_xaa_ar2_multi_cycle` | ✅ Implemented |
| 6 | Build D candidate grid | D ∈ [0, 1] | `grid.py` | `build_d_grid`, `build_default_d_coarse_grid`, `build_d_fine_grid`, `build_d_grid_for_strategy` | ✅ Implemented |
| 6b | Iterate candidates | Cartesian (R, D), or per-R adaptive coarse→fine D | `grid.py`, `evaluation.py`, `api.py` | `build_single_cycle_candidate_grid`, `candidate_iterator`, `evaluate_r_with_adaptive_d` | ✅ Implemented |
| 7 | Apply fractional cyclic filter | (1−2μL+L²)^D applied to Y and X | `filters.py` | `compute_mu`, `compute_fractional_coefficients_single_cycle`, `apply_fractional_filter_single_series`, `apply_single_cycle_filter`, `filter_response_and_design` | ✅ Implemented |
| 8 | Fit filtered regression | Y_D = X_D β + ε via OLS | `regression.py` | `fit_filtered_regression` | ✅ Implemented |
| 9 | Extract betas and residuals | β̂, ε̂ = Y_D − X_D β̂ | `regression.py` | `RegressionResult`, `compute_residuals`, `compute_residual_sum_squares` | ✅ Implemented |
| 10 | Residual periodogram | I_resid(λ_j) = \|FFT(ε̂)_j\|²/(2πT) | `spectral.py` | `compute_residual_periodogram` | ✅ Implemented |
| 10b | Estimate AR nuisance coefficients | ε̂_t = Σ_k φ_k ε̂_{t−k} + e_t | `regression.py` | `estimate_ar_ols` | ✅ Implemented |
| 10c | Build AR spectral adjustment | g(λ_j; φ̂) = \|1 − Σ_k φ̂_k exp(i k λ_j)\|^(−2) | `spectral.py` | `compute_ar_spectral_adjustment` | ✅ Implemented |
| 11a | Time-domain variance VAR | VAR = (1/T) Σ ε̂(t)² | `regression.py` | `compute_time_variance` | ✅ Implemented |
| 11b | Frequency-domain variance VAR* | VAR* = (2π/T) Σ I_resid(λ_j) | `spectral.py` | `compute_frequency_variance_single_cycle`, `compute_frequency_variance_dynamic` | ✅ Implemented |
| 12 | Compute XA(R,D) | XA = −(2π/T) Σ ψ · I_resid, divided by g for AR errors | `spectral.py` | `compute_xa_single_cycle`, `compute_xa_multi_cycle`, `compute_xa_ar_adjusted`, `compute_xa_error_model`, `compute_xa_ar1_dynamic`, `compute_xa_ar1_single_cycle`, `compute_xa_ar1_multi_cycle`, `compute_xa_ar2_dynamic`, `compute_xa_ar2_single_cycle`, `compute_xa_ar2_multi_cycle`, `compute_xa_dynamic` | ✅ Implemented |
| 13a | Compute TEST | TEST = √T / √XAA · XA / VAR | `scoring.py` | `compute_test_statistic` | ✅ Implemented |
| 13b | Compute TEST* | TEST* = √T / √XAA · XA / VAR* | `scoring.py` | `compute_test_star_statistic` | ✅ Implemented |
| PS | Select closest to zero | min \|TEST\| or \|TEST*\| over (R,D) | `scoring.py`, `evaluation.py`, `api.py` | `score_candidate`, `TopKSelector`, `evaluate_candidate`, `run_cyclical_fractional_test` | ✅ Implemented |

---

## Per-step detail

### Point 1 — Import / validate Y(t)

```python
# validation.py
validate_series(y)              # raises InvalidSeriesError on NaN, inf, wrong shape, too short

# api.py
arr = validate_series(y)        # returns clean np.ndarray[float, 1-D]
```

### Point 2 — Chebyshev basis

```python
# chebyshev.py
build_single_chebyshev_polynomial(T, k)  # single column
build_chebyshev_design(T, n_cycles, include_intercept)  # full (T, m) matrix
```

### Point 3 — Original periodogram

```python
# spectral.py
lambdas, I_y = compute_document_periodogram(y)
r_peak = find_periodogram_peak(I_y, exclude_zero=True)   # use False to allow R=0
```

### Points 4 & 6 — Candidate grids

```python
# grid.py
r_candidates = build_r_grid_around_peak(
    r_peak, r_window, T, include_zero=not config.exclude_zero_frequency
)

# Fixed-grid strategy (config.d_search_strategy == "fixed_grid"):
d_grid     = build_d_grid(config.d_grid)        # default [0.0, 0.1, ..., 1.0]
candidates = build_single_cycle_candidate_grid(r_candidates, d_grid)

# Adaptive strategy (default): coarse grid + local refinement, per R
# evaluation.py
search = evaluate_r_with_adaptive_d(y, X, R, config)   # AdaptiveDSearchResult
best_for_R = search.best_result
```

The adaptive search seeds from the coarse grid (`build_d_grid_for_strategy` → `build_default_d_coarse_grid` or `config.d_coarse_grid`), then refines with `build_d_fine_grid(best_coarse_D, config.d_fine_radius, config.d_fine_step)`. It only affects **which** (R, D) candidates are evaluated; each candidate is still scored by the same `evaluate_candidate` machinery (Points 5–13).

### Point 5 — ψ and XAA

```python
# spectral.py
psi = compute_psi_single_cycle(T, R)        # shape (T,), singular positions zeroed
xaa = compute_xaa_single_cycle(psi)         # float
```

For AR(1) and AR(2) residual errors, final XAA is computed after nuisance-parameter estimation:

```python
xaa = compute_xaa_error_model(
    psi, lambdas_r, error_model, ar_coefficients, stochastic_cycle_mode
)
```

### Point 7 — Fractional filter

```python
# filters.py
mu     = compute_mu(T, R)
coeffs = compute_fractional_coefficients_single_cycle(T, R, D)   # shape (T,)
y_f, X_f = filter_response_and_design(y, X, cycles, mode="single")
```

### Points 8 & 9 — Regression

```python
# regression.py
reg = fit_filtered_regression(y_f, X_f)
# reg.betas, reg.residuals, reg.residual_sum_squares
```

### Point 10 — Residual periodogram

```python
# spectral.py
lambdas_r, I_resid = compute_residual_periodogram(reg.residuals)
```

For AR-adjusted runs, estimate nuisance parameters and build the spectral weighting:

```python
# regression.py
ar_coefficients = estimate_ar_ols(reg.residuals, order)

# spectral.py
g = compute_ar_spectral_adjustment(lambdas_r, ar_coefficients)
```

### Point 11 — VAR and VAR*

```python
# regression.py
var_t = compute_time_variance(reg.residuals)

# spectral.py
var_f = compute_frequency_variance_dynamic(I_resid, cycles, mode="single", drop_frequency=True)
```

### Point 12 — XA

```python
# spectral.py
xa = compute_xa_error_model(
    psi, I_resid, error_model, g, stochastic_cycle_mode
)
```

With `error_model="white_noise"`, `g` is an array of ones and the dispatcher delegates to `compute_xa_single_cycle`. AR coefficients are nuisance parameters, not the primary output of the long-memory test.

### Point 13 — TEST and TEST*

```python
# scoring.py
test      = compute_test_statistic(T, xa, xaa, var_t)
test_star = compute_test_star_statistic(T, xa, xaa, var_f)
```

### PS — Select top-k

```python
# scoring.py, evaluation.py, api.py
result = evaluate_candidate(y, X, cycles, config)   # one GridCandidateResult

selector = TopKSelector(k=top_k, statistic_mode=config.statistic_mode)
selector.consider(result)

best   = selector.get_best()
top_k  = selector.get_top_k()
```
