# Mathematical Background

This document describes the statistical test for **fractional cyclic long memory** implemented in `cyclical_fractional_test`.

---

## 1. Objective

The procedure tests whether a time series Y(t) exhibits fractional cyclic long memory at a dominant frequency. The approach combines:

- A **deterministic component** — Chebyshev polynomials that capture smooth trends.
- A **stochastic fractional cyclic component** — the filter `(1 − 2cos(2πR/T)L + L²)^D`, where R is the cyclic frequency index and D is the fractional integration parameter.
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

The dominant cyclic frequency R* is the index j that maximises I(λ_j), typically excluding j=0 (the mean contribution).

---

## 5. Candidate grid for R

A symmetric window of width `r_window` is built around R*:

```
R ∈ { max(1, R* − r_window), ..., min(T−1, R* + r_window) }
```

The default `r_window = 10` gives at most 21 R candidates.

---

## 6. Function ψ

For each candidate R, define:

```
ψ(λ_j, R) = log( |2 ( cos(λ_j) − cos(λ_R) )| )
```

This is the score function of the cyclic fractional spectral model.

**Singularity handling.** The expression is singular at j = R and at j = T−R (the mirror frequency). Both positions are set to 0 when `drop_singular_frequency=True` (the default). See [implementation_notes.md](implementation_notes.md) for details.

---

## 7. XAA

The asymptotic variance of the score:

```
XAA(R) = (2 / T) Σ_{j=0}^{T−1} ψ(λ_j, R)²
```

XAA is always positive for valid (non-degenerate) R.

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

When `drop_singular_frequency=True`, the term at j = R is excluded from the sum. Under Parseval's theorem, VAR ≈ VAR* for well-specified models.

---

## 14. XA

The cross-product between the score function and the residual periodogram:

```
XA(R, D) = − (2π / T) Σ_{j=0}^{T−1} ψ(λ_j, R) I_resid(λ_j)
```

XA is the numerator of the score test statistic. It equals zero when the model is correctly specified at (R, D).

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

## 17. top_k

Rather than storing the full grid, the algorithm retains only the `top_k` candidates with the smallest |TEST| (or |TEST*|). This is controlled by `CyclicalTestConfig.top_k`.

---

## 18. Multi-cycle architecture

The codebase separates single-cycle and multi-cycle paths via dispatcher functions (`compute_psi_dynamic`, `compute_xa_dynamic`, etc.). Multi-cycle support (`stochastic_cycle_mode="multi_cycle"`) is architecturally prepared but not yet numerically implemented; calling it raises `NotImplementedError`.
