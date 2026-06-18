# Mathematical Background

This document describes the statistical test for **fractional cyclic long memory** implemented in `cyclical_fractional_test`.

---

## 1. Objective

The procedure tests whether a time series Y(t) exhibits fractional cyclic long memory at a dominant frequency. The approach combines:

- A **deterministic component** — Chebyshev polynomials that capture smooth trends.
- A **stochastic fractional cyclic component** — the filter `(1 − 2cos(2πR/T)L + L²)^D`, where R is the cyclic frequency index and D is the fractional integration parameter.
- A configurable **residual error specification** — white noise, AR(1), or AR(2).
- A **grid search** over candidate pairs (R, D).
- A **spectral statistic** TEST (or TEST*) that measures how close XA is to zero after normalisation.

---

## 2. Series

```
Y(t),  t = 1, ..., T
```

A real-valued time series of length T. The test is designed to detect long memory at a cyclic frequency.

---

## 3. Deterministic basis (Chebyshev polynomials)

The deterministic part is represented by `m` Chebyshev polynomials:

```
P_0(t) = 1

P_k(t) = 2 cos(k π (t − 0.5) / T)   for k = 1, ..., m
```

The parameter `n_deterministic_cycles` in `CyclicalTestConfig` controls `m`. With `include_intercept=False` (default), the design matrix X has columns `[P_1, ..., P_m]`. With `include_intercept=True`, column `P_0` is prepended.

---

## 4. Periodogram

The discrete periodogram is:

```
I(λ_j) = (1 / (2π T)) [ (Σ_t Y(t) sin(λ_j t))²  +  (Σ_t Y(t) cos(λ_j t))² ]
```

with frequencies:

```
λ_j = 2π j / T,   j = 0, 1, ..., T−1
```

In practice this is computed via the Fast Fourier Transform:

```
I(λ_j) = |FFT(Y)_j|² / (2π T)
```

The dominant frequency R* is the index j that maximises I(λ_j), typically
excluding j=0 (the mean contribution). Set `exclude_zero_frequency=False` to
allow R=0 to enter the search.

---

## 5. Candidate grid for R

A symmetric window of width `r_window` is built around R*:

```
R ∈ { max(R_min, R* − r_window), ..., min(T−1, R* + r_window) }
```

The default is `R_min=1`, so frequency zero is excluded. When
`exclude_zero_frequency=False`, `R_min=0` and R=0 may be a candidate.

---

## 6. Function ψ

For each candidate R, define:

```
ψ(λ_j, R) = log( |2 ( cos(λ_j) − cos(λ_R) )| )
```

This is the score function of the fractional spectral model. For R=0 the
frequency is zero rather than oscillatory, and the filter becomes
`(1 − 2L + L²)^D = (1 − L)^(2D)`.

**Singularity handling.** The expression is singular at j = R and at j = T−R
(the mirror frequency). For R=0, the only in-array singularity is j=0. Singular
positions are set to 0 when `drop_singular_frequency=True` (the default). See
[implementation_notes.md](implementation_notes.md) for details.

---

## 7. White-noise XAA

The asymptotic variance of the score:

```
XAA(R) = (2 / T) Σ_{j=0}^{T−1} ψ(λ_j, R)²
```

This is the XAA formula for `error_model="white_noise"`. XAA is always positive for valid (non-degenerate) R. AR-adjusted definitions are given in Section 12a.

---

## 8. Fractional integration parameter D

D ∈ [0, 1] controls the strength of the fractional cyclic memory. The default search grid is `[0.0, 0.1, …, 1.0]`.

- D = 0 → no filtering (identity operator).
- D = 1 → second-order moving-average filter.
- Intermediate D → genuine fractional cyclic integration.

---

## 9. Fractional cyclic filter

The filter operator is:

```
(1 − 2 cos(2πR/T) L + L²)^D
```

where L is the lag operator and:

```
μ = cos(2π R / T)
```

Its moving-average representation has coefficients `C_0, C_1, C_2, …` given by:

```
C_0 = 1
C_1 = −2 D μ
C_j = [ 2μ (j − 1 − D) C_{j−1}  +  (2D − j + 2) C_{j−2} ] / j   for j ≥ 2
```

Special cases:
- D = 0: `C_0 = 1`, `C_j = 0` for j ≥ 1 (identity).
- D = 1: `C_0 = 1`, `C_1 = −2μ`, `C_2 = 1`, `C_j = 0` for j ≥ 3.

---

## 10. Filtered series

The filter is applied by causal convolution:

```
Y_D(t) = Σ_{j=0}^{t−1} C_j  Y(t − j)
```

The same convolution is applied to each column of the design matrix X, yielding `X_D`.

---

## 11. Filtered regression

After filtering, the ordinary least-squares model is:

```
Y_D = X_D β + ε
```

Estimated via `np.linalg.lstsq`, giving coefficients `β̂` and residuals `ε̂ = Y_D − X_D β̂`.

---

## 12. Residual periodogram

The periodogram of the regression residuals uses the identical formula as Section 4:

```
I_resid(λ_j) = |FFT(ε̂)_j|² / (2π T)
```

## 12a. Residual error specifications

The residual error model is selected with `CyclicalTestConfig.error_model`.

### White noise

With `error_model="white_noise"`, the original XAA and XA formulas are used unchanged. No nuisance coefficients are estimated.

### AR(1)

For `error_model="ar1"`, the filtered-regression residuals are used to estimate:

```
ε̂_t = φ ε̂_{t−1} + e_t
```

by OLS. The spectral adjustment is:

```
g(λ_j; φ̂) = |1 − φ̂ exp(i λ_j)|^(−2)
           = (1 + φ̂² − 2 φ̂ cos(λ_j))^(−1)
```

Define:

```
epsilon(λ_j; φ̂) = 2 (cos(λ_j) − φ̂) g(λ_j; φ̂)
```

Then:

```
XAA_AR1 = (2/T) [
    Σ_j ψ(λ_j)²
    − (Σ_j ψ(λ_j) epsilon(λ_j; φ̂))² / Σ_j epsilon(λ_j; φ̂)²
]

XA_AR1 = −(2π/T) Σ_j ψ(λ_j) I_resid(λ_j) / g(λ_j; φ̂)
```

### AR(2)

For `error_model="ar2"`, OLS estimates:

```
ε̂_t = φ_1 ε̂_{t−1} + φ_2 ε̂_{t−2} + e_t
```

The spectral adjustment is:

```
g(λ_j; φ̂) =
|1 − φ̂_1 exp(i λ_j) − φ̂_2 exp(2 i λ_j)|^(−2)
```

Define:

```
epsilon_1(λ_j; φ̂) =
    2 (cos(λ_j) − φ̂_1 − φ̂_2 cos(λ_j)) g(λ_j; φ̂)

epsilon_2(λ_j; φ̂) =
    2 (cos(2λ_j) − φ̂_1 cos(λ_j) − φ̂_2) g(λ_j; φ̂)
```

Let `epsilon(λ_j; φ̂) = [epsilon_1(λ_j; φ̂), epsilon_2(λ_j; φ̂)]`. Then:

```
S_psi_epsilon = Σ_j epsilon(λ_j; φ̂) ψ(λ_j)
S_epsilon_epsilon = Σ_j epsilon(λ_j; φ̂) epsilon(λ_j; φ̂).T

XAA_AR2 = (2/T) [
    Σ_j ψ(λ_j)²
    − S_psi_epsilon.T inv(S_epsilon_epsilon) S_psi_epsilon
]

XA_AR2 = −(2π/T) Σ_j ψ(λ_j) I_resid(λ_j) / g(λ_j; φ̂)
```

The estimated AR coefficients are nuisance parameters used to adjust the score statistic. They are exposed in each `GridCandidateResult` for diagnostics, but they are not the target of the long-memory test.

The AR(1)/AR(2) specification is independent of the number of stochastic
cycles. The code therefore exposes separate single-cycle implementations and
multi-cycle implementations for adjusted XAA and XA. The multi-cycle AR
formulas use the same projection adjustments, replacing `ψ(λ_j)` with the
aggregate `ψ_multi(λ_j)`.

---

## 13. Variance estimators

**Time-domain variance (VAR):**

```
VAR(R, D) = (1/T) Σ_{t=1}^{T} ε̂(t)²
```

This is the second moment of the residuals (not the centered variance).

**Frequency-domain variance (VAR*):**

```
VAR*(R, D) = (2π / T) Σ_{j=0}^{T−1} I_resid(λ_j)
```

When `drop_singular_frequency=True`, the term at j = R is excluded from the sum,
including j=0 for R=0. Under Parseval's theorem, VAR ≈ VAR* for well-specified
models.

---

## 14. White-noise XA

The cross-product between the score function and the residual periodogram:

```
XA(R, D) = − (2π / T) Σ_{j=0}^{T−1} ψ(λ_j, R) I_resid(λ_j)
```

This is the XA formula for `error_model="white_noise"`. AR-adjusted XA formulas are given in Section 12a. XA is the numerator of the score test statistic. It equals zero when the model is correctly specified at (R, D).

---

## 15. TEST and TEST*

The standardised statistics are:

```
TEST(R, D)  = √T / √XAA(R)  ×  XA(R, D) / VAR(R, D)

TEST*(R, D) = √T / √XAA(R)  ×  XA(R, D) / VAR*(R, D)
```

Both statistics are signed. The absolute value is used only when ranking candidates.

---

## 16. Selection criterion

The best estimate of (R, D) is the combination that makes |TEST| (or |TEST*|) closest to zero, i.e., the value where XA ≈ 0 and the model is most consistent with the data.

---

## 16b. Search strategy for D (adaptive coarse-to-fine)

This section concerns **how candidate D values are chosen**, not the test itself. The statistic in Sections 1–16 is unchanged: ψ, XAA, XA, VAR, VAR*, TEST, and TEST* are computed exactly as above for whichever (R, D) the search decides to evaluate.

For a fixed frequency index R, the test statistic is asymptotically normal in D. The objective `|TEST(R, ·)|` is therefore locally smooth enough that its minimiser can be located by a two-stage search instead of a dense grid over the whole interval `[0, 1]`:

1. **Coarse stage.** Evaluate D on a coarse grid, by default `{0.0, 0.1, …, 1.0}`. Select the coarse minimiser

   ```
   D_coarse* = argmin_D |TEST(R, D)|        (or |TEST*(R, D)|)
   ```

2. **Fine stage.** Evaluate a local grid around `D_coarse*` with step `h` and radius `ρ` (defaults `h = 0.01`, `ρ = 0.09`), clipped to `[0, 1]`:

   ```
   { clip(D_coarse* + k·h, 0, 1) : k = -⌊ρ/h⌋, …, ⌊ρ/h⌋ }
   ```

   For `D_coarse* = 0.3` this is `{0.21, 0.22, …, 0.39}`. At a boundary (`0.0` or `1.0`) the window is one-sided after clipping.

3. **Selection.** The chosen D for this R is the minimiser of `|TEST|` (or `|TEST*|`) over the union of the coarse and fine evaluations.

The same scoring rule used for ranking (`abs(TEST)` for `statistic_mode="test"`, `abs(TEST*)` for `"test_star"`) drives both stages. The strategy reduces the number of D evaluations from a dense grid to roughly `(#coarse) + (2ρ/h + 1)` per R while still resolving D to the fine step `h`. Setting `d_search_strategy="fixed_grid"` evaluates the full Cartesian `(R, D)` grid instead.

---

## 17. top_k

Rather than storing the full grid, the algorithm retains only the `top_k` candidates with the smallest |TEST| (or |TEST*|). This is controlled by `CyclicalTestConfig.top_k`.

---

## 18. Multi-cycle architecture

The codebase separates single-cycle and multi-cycle paths via dispatcher functions (`compute_psi_dynamic`, `compute_xa_dynamic`, etc.).

With `stochastic_cycle_mode="multi_cycle"`, the API selects the top `n_stochastic_cycles = k` periodogram peaks:

```
R_1, ..., R_k = top-k indices of I(λ_j)
```

Multi-cycle search evaluates joint D vectors. A candidate is the tuple:

```
((R_1, D_1), ..., (R_k, D_k))
```

Fixed-grid search evaluates every tuple in the Cartesian product `D_grid^k`.
For `m = len(D_grid)`, this gives `m^k` multi-cycle candidates. Adaptive search
evaluates the coarse Cartesian product, chooses the best coarse D vector, and
then evaluates the local fine Cartesian product around that vector. The
residuals, XA, XAA, VAR, VAR*, and AR nuisance coefficients are computed from
each complete multi-cycle filter, so the selected `D_q` values are not marginal
per-frequency optima.

The scalar multi-cycle score uses the aggregate function:

```
ψ_multi(λ_j) = Σ_q log( |2 (cos(λ_j) − cos(λ_Rq))| )
```

In-array singular positions at every `R_q` and `T−R_q` are set to zero when
`drop_singular_frequency=True`; for `R_q=0`, this is only `j=0`.

The white-noise multi-cycle quantities are:

```
XAA_multi = (2/T) Σ_j ψ_multi(λ_j)^2

XA_multi = −(2π/T) Σ_j ψ_multi(λ_j) I_resid(λ_j)
```

AR(1) and AR(2) multi-cycle adjustments use the same projection formulas as Section 12a, replacing `ψ(λ_j)` with `ψ_multi(λ_j)`.

---

## 19. Prediction (scikit-learn-style model)

Once a candidate `(R, D)` is selected, the fitted objects define a complete generative model that can reconstruct and forecast the series. With `β̂` the OLS coefficients, `ε̂` the filtered-regression residuals, and `φ̂` the AR error coefficients:

```
Y_t            = X_t·β̂ + S_t                       (deterministic + stochastic)
filter_{R,D}(S)_t = ε̂_t                             (cyclic long memory)
ε̂_t           = Σ_{i=1}^{p} φ̂_i·ε̂_{t-i} + e_t      (AR(p) error, p ∈ {0,1,2})
```

A key identity makes this exact and cheap to compute: because the filter is linear,

```
ε̂ = Y_D − X_D β̂ = filter_{R,D}(Y − X β̂) = filter_{R,D}(S),
```

so the regression residuals **are** the filtered stochastic part — no refiltering is needed.

### 19.1 In-sample reconstruction (one-step conditional mean)

For t = 1, …, T the predicted value is the one-step-ahead conditional mean

```
ŷ_t = X_t·β̂ − Σ_{j=1}^{t-1} C_j·S_{t-j} + Σ_{i=1}^{p} φ̂_i·ε̂_{t-i},
```

where `C_j` are the fractional-filter coefficients of Section 9 and `S_t = Y_t − X_t·β̂`. The prediction error equals the AR innovation exactly:

```
Y_t − ŷ_t = ε̂_t − Σ_i φ̂_i·ε̂_{t-i} = e_t.
```

For `D = 0` (identity filter, white noise) this collapses to the deterministic OLS fit `ŷ_t = X_t·β̂`.

### 19.2 Out-of-sample forecast

For t = T+1, …, T+h the error is forecast first by the AR recursion (expected innovation zero),

```
ε̂_{T+k} = Σ_{i=1}^{p} φ̂_i·ε̂_{T+k-i},
```

then the stochastic component is propagated by inverting the cyclic filter,

```
Ŝ_{T+k} = ε̂_{T+k} − Σ_{j=1}^{T+k-1} C_j·Ŝ_{T+k-j},
```

reusing the in-sample `S` values for the known indices, and finally

```
ŷ_{T+k} = X_{T+k}·β̂ + Ŝ_{T+k}.
```

The deterministic term extrapolates the Chebyshev basis `P_k(t) = 2cos(kπ(t−0.5)/T_ref)` with the **training length `T_ref = T` held fixed**, so the basis is evaluated consistently beyond the sample. The filter coefficients `C_j` are extended to length `T+h` using the recursion of Section 9 with `μ = cos(2πR/T)` fixed at the training frequency.

### 19.3 AR coefficient estimation

The AR coefficients `φ̂` are the same nuisance parameters estimated by OLS from `ε̂` (Section 12a); prediction reuses them rather than re-estimating. The only additional quantity is the innovation variance

```
σ̂² = mean(e_t²),   e_t = ε̂_t − Σ_i φ̂_i·ε̂_{t-i}.
```

### 19.4 Prediction intervals

In-sample bounds use the one-step innovation standard deviation `σ̂`. Forecast bounds widen with the MA(∞) weights `ψ_l` of the inverse system `S = filter_{R,D}^{-1}(ε)` with `ε = AR(p)^{-1}(e)` — the convolution of the inverse-filter coefficients (params `−D`) with the AR inverse MA weights. The horizon-k forecast-error variance is

```
Var(ŷ_{T+k}) = σ̂² · Σ_{l=0}^{k-1} ψ_l².
```

Parameter-estimation uncertainty is not included.
