# Quickstart

## Run the Test

```python
import numpy as np

from cyclical_fractional_test import CyclicalTestConfig, run_cyclical_fractional_test

rng = np.random.default_rng(42)
T = 240
t = np.arange(T, dtype=float)
y = np.cos(2.0 * np.pi * 12 * t / T) + 0.25 * rng.standard_normal(T)

result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        n_deterministic_cycles=4,
        r_window=5,
        top_k=3,
        error_model="ar1",
    ),
)

best = result.best_result
print(best.cycles)
print(best.test_value)
print(best.ar_coefficients)
```

The default `d_search_strategy="adaptive"` evaluates a coarse grid of `D`
values and refines locally around the best coarse value for each candidate
frequency.

## Fixed D Grid

```python
result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        r_window=5,
        top_k=3,
    ),
)
```

## Exclude Known Frequencies

Use `ignored_stochastic_rs` when a known deterministic or nuisance frequency
should not be selected as stochastic memory.

```python
result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        chebyshev_orders=(2, 10),
        include_intercept=True,
        ignored_stochastic_rs=(5,),
    ),
)
```

## Multi-Cycle Mode

```python
result = run_cyclical_fractional_test(
    y,
    config=CyclicalTestConfig(
        stochastic_cycle_mode="multi_cycle",
        n_stochastic_cycles=3,
        d_search_strategy="adaptive",
        error_model="ar1",
    ),
)

print(result.r_candidates)
print(result.best_result.cycles)
```

## Model Wrapper

```python
from cyclical_fractional_test import CyclicalFractionalModel

model = CyclicalFractionalModel(
    n_deterministic_cycles=4,
    error_model="ar1",
).fit(y)

forecast = model.predict(len(y) + 20)
lower, upper = model.predict_interval(len(y) + 20, alpha=0.05)
```

## Thresholded Candidate Collection

`threshold` keeps every evaluated candidate whose selected absolute statistic is
below the threshold. These candidates are descriptive: they do not affect
`best_result` or `top_k_results`.

```python
model = CyclicalFractionalModel(threshold=2.0).fit(y)
candidates = model.get_under_threshold_candidates()
```
