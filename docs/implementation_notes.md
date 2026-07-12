# Implementation Notes

This document records non-obvious implementation decisions and their rationale.

---

## 1. Index convention: mathematics vs Python

The document defining the test uses 1-based indices for time (t = 1, …, T) and frequency (j = 1, …, T). Python and NumPy use 0-based indices (j = 0, …, T-1). The code adopts 0-based indexing throughout to match NumPy's FFT output.

**Consequence for R.** The dominant frequency R* is the 0-based index into the periodogram array. `I_y[R]` is the periodogram value at frequency `λ_R = 2π R / T`. This is consistent with the mathematical formula once the phase offset is resolved via `|FFT|²`, which eliminates the 1-based vs 0-based phase difference. R=0 is the zero-frequency component and is only included in the API search when `exclude_zero_frequency=False`.

**Consequence for ψ.** The singular positions in ψ are `j = R` (exact zero) and `j = T−R` (near-zero due to floating-point symmetry). For R=0, `T−R` is outside the array, so only `j=0` is zeroed when `drop_singular_frequency=True`.

---

## 2. Periodogram computation

The periodogram is computed via `np.fft.fft` rather than explicit sin/cos summation:

```python
fft_vals = np.fft.fft(arr)
I = np.abs(fft_vals) ** 2 / (2.0 * np.pi * T)
```

This is mathematically equivalent to the direct formula because `|·|²` eliminates the phase difference caused by 0-based vs 1-based indexing. The advantage is O(T log T) complexity.

---

## 3. Treatment of j = R in ψ and VAR*

**In ψ:** position j = R gives `cos(λ_j) − cos(λ_R) = 0`, so `log(0) = −∞`. The mirror position j = T−R gives a value near 0 due to floating-point cos symmetry, except for R=0 where the mirror index is T and is not part of the FFT grid.

With `drop_singular_frequency=True` (default), singular positions are explicitly set to 0.0 after computing the log. This prevents −∞ from propagating into XAA and XA.

**In VAR*:** when `drop_frequency=True` (default), the term at `j = R` is excluded from the sum `(2π/T) Σ I_resid[j]`. This reflects the fact that the residual periodogram at the estimated frequency is biased by the fractional filtering.

---

## 4. D = 0: identity filter

When D = 0 the filter coefficients reduce to:

```
C_0 = 1,  C_j = 0  for j ≥ 1
```

The causal convolution `Σ C_j x[t−j]` then returns `x[t]` exactly. In the code, `filter_response_and_design` with D=0 returns arrays numerically identical to the inputs, verified by `np.allclose(y_filtered, y)`.

---

## 5. D = 1: second-order MA filter

When D = 1 the filter is exactly `(1 − 2μL + L²)`, which has a finite MA representation:

```
C_0 = 1,  C_1 = −2μ,  C_2 = 1,  C_j = 0  for j ≥ 3
```

The recurrence formula produces these values without special-casing D = 1.

---

## 6. Causal convolution implementation

The filter is applied by:

```python
np.convolve(x, coefficients, mode="full")[:T]
```

This computes the full linear convolution and truncates to the causal part (length T). Commutativity of convolution means `(x * c)[t] = Σ_j c[j] x[t−j]`, which matches the required formula.

---

## 7. top_k instead of full grid

All (R, D) candidates are evaluated, but `TopKSelector` retains only the `k` best. This is implemented as a sorted list that is trimmed after each insertion — a simple O(k) approach sufficient for typical grid sizes (< 1000 candidates). The full grid is never stored.

`CyclicalFractionalTestResult.n_candidates_evaluated` records how many candidates were processed.

---

## 8. Time-domain variance: second moment, not centered

`compute_time_variance` returns:

```python
np.mean(residuals ** 2)
```

This is the **second moment** (not the sample variance `np.var`, which subtracts the mean). The distinction matters for residuals that do not have zero mean. The document's VAR definition is the second moment.

---

## 9. Multi-cycle architecture

The code is structured to support multi-cycle filters of the form `Π_q (1−2μ_q L + L²)^{D_q}` via the dispatcher pattern. The current status of each multi-cycle path is:

| Function | Status |
|---|---|
| `compute_psi_multi_cycle` | ✅ Implemented as aggregate `ψ_multi(λ_j) = Σ_q ψ(λ_j, R_q)` |
| `compute_xaa_multi_cycle` | ✅ Implemented as scalar `(2/T) Σ_j ψ_multi(λ_j)^2` |
| `apply_multi_cycle_filter` | ✅ Implemented (sequential chaining) |
| `compute_frequency_variance_multi_cycle` | ✅ Implemented (excludes all R_q) |
| `compute_xa_multi_cycle` | ✅ Implemented as scalar `-(2π/T) Σ_j ψ_multi(λ_j) I_resid(λ_j)` |
| `compute_xaa_ar1_multi_cycle` | ✅ Implemented with the AR(1) projection formula using `ψ_multi` |
| `compute_xaa_ar2_multi_cycle` | ✅ Implemented with the AR(2) projection formula using `ψ_multi` |
| `compute_xa_ar1_multi_cycle` | ✅ Implemented with `ψ_multi / g_AR1` |
| `compute_xa_ar2_multi_cycle` | ✅ Implemented with `ψ_multi / g_AR2` |

In `run_cyclical_fractional_test`, `stochastic_cycle_mode="multi_cycle"` selects the top `n_stochastic_cycles = k` periodogram peaks and always evaluates complete joint candidates of the form `((R_1,D_1), ..., (R_k,D_k))`. Fixed-grid search evaluates the full Cartesian product `d_grid^k`. Adaptive search evaluates the coarse Cartesian product, chooses the best joint D vector, then evaluates the local fine Cartesian product around that vector.

---

## 10. Diagnostics do not alter statistics

The `TestDiagnostics` object and related functions in `diagnostics.py` are computed **after** all candidate evaluations complete. They read from already-computed values (counters, lambdas, periodogram) and never feed back into the statistical computation. Adding or removing diagnostics cannot change `best_result` or `top_k_results`.

---

## 11. Validation style

Each function in a math/logic module calls a single private `_validate_<function_name>` helper as its first line. All such helpers are grouped at the bottom of their file, separated by a standard comment block. This keeps the mathematical logic visible and uncluttered while still providing defensive input checks.

Dispatcher functions (where the mode check IS the logic), `api.py` (thin orchestration entry point), and `validation.py` itself are exempt from this pattern.

---

## 12. Circular import avoidance

`validation.py` needs to reference `StochasticCycle` from `results.py`, but `results.py` imports `CyclicalTestConfig` from `config.py`, and `config.py` might indirectly re-trigger imports. The solution is a **local import** inside `validate_cycle` and `validate_cycles`:

```python
def validate_cycle(cycle, T=None):
    from .results import StochasticCycle  # local import avoids circular dependency
    ...
```

Similarly, `validate_config` uses `from .config import CyclicalTestConfig` locally.

---

## 13. AR coefficients are per-candidate nuisance parameters

`error_model="white_noise"` remains the default and uses the original formulas unchanged. With `error_model="ar1"` or `"ar2"`, each candidate first fits the filtered deterministic regression, then estimates AR coefficients from that candidate's residuals via `estimate_ar_ols`.

The estimated coefficients feed only the spectral adjustment of XAA and XA. They are exposed as `GridCandidateResult.ar_coefficients` for diagnostics, but they are nuisance parameters rather than the main target of the fractional cyclic long-memory test.

`compute_ar_spectral_adjustment` clips zero and near-zero spectral denominators to machine epsilon before inversion. This keeps the adjustment finite at frequencies where the estimated AR polynomial is numerically zero.

The residual error specification and stochastic-cycle mode are dispatched
independently. AR(1) and AR(2) each have explicit single-cycle functions,
multi-cycle functions, and mode dispatchers for XAA and XA. The multi-cycle AR
paths use the same projection formulas as the single-cycle paths with
`ψ_multi` replacing `ψ`.

---

## 14. Adaptive D search is a search strategy, not a new test

`d_search_strategy="adaptive"` is the default. It changes only **which** D values are evaluated, never how a candidate is scored: every (R, D) still goes through the same `evaluate_candidate` machinery. The justification is that for fixed R the statistic is asymptotically normal, so `|TEST(R, ·)|` is locally smooth and a coarse-to-fine search approximates the dense-grid minimiser cheaply.

Design decisions:

- **`d_grid` is ignored in adaptive mode.** The preferred-behavior option from the task: `config.d_grid` only matters when `d_search_strategy="fixed_grid"`. Adaptive seeds from `d_coarse_grid` (or the default 11-point grid). Keeping the two knobs separate avoids ambiguity about whether a user-supplied grid is "the whole search" or "just the coarse seed."
- **One candidate per R reaches the ranker.** `evaluate_r_with_adaptive_d` returns the best candidate for its R; `run_cyclical_fractional_test` feeds that single result to `TopKSelector`. So `top_k` ranks across frequencies, not across (R, D) pairs. This is why the two count-oriented API tests were switched to `fixed_grid` — their assertions are about Cartesian-grid candidate counts.
- **Reuse via rounded-D keys.** Coarse and fine results are stored in a dict keyed by D rounded to 12 decimals. The best coarse D that reappears in the fine window is reused, not recomputed, so `n_candidates_evaluated` counts distinct (R, D) pairs (coarse + fine − overlap).
- **Stable rounding.** Both `build_default_d_coarse_grid` and `build_d_fine_grid` round to 12 decimals to avoid artifacts like `0.30000000000000004`, and `build_d_fine_grid` clips to `[0, 1]` and de-duplicates so boundary centers yield a one-sided window.
- **Diagnostics are descriptive only.** The adaptive fields on `TestDiagnostics` (`d_search_strategy`, `d_fine_step`, `d_fine_radius`, `best_coarse_d_per_r`, `final_d_per_r`, `n_coarse_evaluations`, `n_fine_evaluations`) record what the search did and never feed back into the computation.
