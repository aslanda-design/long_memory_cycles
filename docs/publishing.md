# Publishing

This project is configured for PyPI publication using standard Python
distribution artifacts and a GitHub Actions publish workflow.

## Local Preflight

Start from a clean working tree, then run:

```bash
python3 -m pip install -e ".[dev,docs]"
python3 -m pytest
python3 -m mkdocs build --strict
rm -rf dist build *.egg-info
python3 -m build
python3 -m twine check dist/*
```

Inspect the built source distribution before upload:

```bash
tar -tzf dist/*.tar.gz | sort
```

The archive should include `src/`, `tests/`, `docs/`, `README.md`, `LICENSE`,
`CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff`, `MANIFEST.in`, and
`pyproject.toml`. It should not include local `data/`, `figures/`, `models/`,
or notebook directories.

## PyPI Trusted Publishing

The workflow in `.github/workflows/publish.yml` is designed for PyPI trusted
publishing.

1. Create the project on PyPI if it does not exist.
2. In the PyPI project settings, add a trusted publisher for:
   - Owner: `aslanda-design`
   - Repository: `log_memory_cycles`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. Create a GitHub environment named `pypi`.
4. Publish a GitHub release. The workflow builds, checks, and uploads the
   distributions.

The workflow also supports manual dispatch from GitHub Actions.

## Manual Upload

If trusted publishing is not configured, use an API token:

```bash
python3 -m twine upload dist/*
```

## Release Checklist

- Confirm the license choice in `LICENSE` and `pyproject.toml`.
- Confirm the package name `cyclical-fractional-test` is available on PyPI.
- Update `pyproject.toml`, `CHANGELOG.md`, and `CITATION.cff` to the same
  version.
- Run tests, docs build, package build, and `twine check`.
- Tag the release, for example `v0.1.0`.
