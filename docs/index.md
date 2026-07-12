# cyclical-fractional-test

`cyclical-fractional-test` implements a statistical workflow for detecting
fractional cyclic long memory in time series.

The library starts from a univariate series, identifies candidate periodogram
frequencies, evaluates fractional cyclic filters over candidate `D` values, fits
the filtered regression on deterministic Chebyshev terms, and ranks candidates
with `TEST` or `TEST*`.

## What You Get

- A full `run_cyclical_fractional_test` pipeline.
- Low-level helpers for periodograms, grids, filters, regression, scoring, and
  diagnostics.
- Adaptive and fixed-grid `D` search.
- White-noise, AR(1), and AR(2) residual error models.
- A fitted `CyclicalFractionalModel` wrapper for reconstruction and forecasting.

## Install

```bash
python3 -m pip install cyclical-fractional-test
```

For development:

```bash
python3 -m pip install -e ".[dev,docs]"
```

## Next Steps

- Start with the [quickstart](quickstart.md).
- Use the [API reference](api_reference.md) while integrating the package.
- Read the [mathematical background](mathematical_background.md) for the test
  derivation.
- Follow the [publishing guide](publishing.md) before cutting a release.
