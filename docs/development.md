# Development

## Environment

```bash
python3 -m pip install -e ".[dev,docs]"
```

The core runtime dependency is NumPy. Development extras add tests, coverage,
build tools, Twine, and documentation tooling.

## Tests

```bash
python3 -m pytest
```

With coverage:

```bash
python3 -m pytest --cov=cyclical_fractional_test --cov-report=term-missing
```

## Documentation

Preview docs locally:

```bash
python3 -m mkdocs serve
```

Build static docs:

```bash
python3 -m mkdocs build --strict
```

## Distribution Checks

```bash
rm -rf dist build *.egg-info
python3 -m build
python3 -m twine check dist/*
```

`MANIFEST.in` intentionally excludes local datasets, generated figures, model
artifacts, and notebooks from source distributions.

## Versioning

For a release:

1. Update `version` in `pyproject.toml`.
2. Update `__version__` fallback in `src/cyclical_fractional_test/__init__.py`.
3. Update `CHANGELOG.md`.
4. Update `CITATION.cff`.
5. Run tests and distribution checks.
6. Tag the release and publish from GitHub.
