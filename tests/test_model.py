import numpy as np
import pytest

from cyclical_fractional_test import (
    CyclicalFractionalModel,
    CyclicalTestConfig,
    NotFittedError,
)
from cyclical_fractional_test.exceptions import InvalidConfigurationError


def _series(n=120, freq=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    return np.cos(2 * np.pi * freq * t / n) + 0.3 * rng.standard_normal(n)


def test_fit_returns_self_and_sets_attributes():
    y = _series()
    model = CyclicalFractionalModel()
    out = model.fit(y)
    assert out is model
    assert model.n_train_ == len(y)
    assert isinstance(model.R_, int)
    assert 0.0 <= model.D_ <= 1.0
    assert model.betas_.ndim == 1
    assert model.residuals_.shape == (len(y),)
    assert model.innovation_variance_ > 0.0


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        CyclicalFractionalModel().predict(5)


def test_predict_interval_before_fit_raises():
    with pytest.raises(NotFittedError):
        CyclicalFractionalModel().predict_interval(5)


def test_predict_in_sample_length():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    assert model.predict(len(y)).shape == (len(y),)
    assert model.predict(10).shape == (10,)


def test_predict_out_of_sample_length():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    out = model.predict(len(y) + 15)
    assert out.shape == (len(y) + 15,)
    assert np.all(np.isfinite(out))


def test_predict_prefix_consistency():
    # predict(n) for n <= T must be the prefix of the full in-sample reconstruction.
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    full = model.predict(len(y))
    np.testing.assert_allclose(model.predict(30), full[:30])


def test_predict_extends_in_sample_prefix():
    # The first T entries of an out-of-sample prediction equal the in-sample ones.
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    in_sample = model.predict(len(y))
    extended = model.predict(len(y) + 8)
    np.testing.assert_allclose(extended[: len(y)], in_sample)


@pytest.mark.parametrize("error_model", ["white_noise", "ar1", "ar2"])
def test_predict_all_error_models(error_model):
    y = _series(seed=2)
    model = CyclicalFractionalModel(error_model=error_model).fit(y)
    expected_order = {"white_noise": 0, "ar1": 1, "ar2": 2}[error_model]
    assert len(model.ar_coefficients_) == expected_order
    out = model.predict(len(y) + 5)
    assert np.all(np.isfinite(out))


def test_kwargs_override_config():
    model = CyclicalFractionalModel(error_model="ar1", n_deterministic_cycles=3)
    assert model.config.error_model == "ar1"
    assert model.config.n_deterministic_cycles == 3


@pytest.mark.parametrize("include_intercept, expected_betas", [(True, 1), (False, 0)])
def test_fit_predict_supports_zero_deterministic_cycles(
    include_intercept, expected_betas
):
    y = _series(seed=8)
    model = CyclicalFractionalModel(
        n_deterministic_cycles=0,
        include_intercept=include_intercept,
        d_search_strategy="fixed_grid",
        d_grid=np.array([0.0]),
        r_window=0,
    ).fit(y)
    assert model.betas_.shape == (expected_betas,)
    out = model.predict(len(y) + 5)
    assert out.shape == (len(y) + 5,)
    assert np.all(np.isfinite(out))


def test_explicit_config_respected():
    config = CyclicalTestConfig(error_model="ar2")
    model = CyclicalFractionalModel(config=config)
    assert model.config.error_model == "ar2"


def test_predict_interval_brackets_prediction_and_widens():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    n = len(y) + 12
    center = model.predict(n)
    lower, upper = model.predict_interval(n, alpha=0.05)
    assert np.all(lower < center) and np.all(center < upper)
    # Forecast bounds should not shrink as the horizon grows.
    widths = upper[len(y):] - lower[len(y):]
    assert np.all(np.diff(widths) >= -1e-9)


def test_predict_interval_rejects_unknown_alpha():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    with pytest.raises(InvalidConfigurationError):
        model.predict_interval(5, alpha=0.5)


def test_predict_rejects_non_positive_n():
    y = _series()
    model = CyclicalFractionalModel().fit(y)
    with pytest.raises(InvalidConfigurationError):
        model.predict(0)


def test_multi_cycle_predict():
    y = _series(seed=4)
    model = CyclicalFractionalModel(
        stochastic_cycle_mode="multi_cycle", n_stochastic_cycles=2
    ).fit(y)
    out = model.predict(len(y) + 6)
    assert out.shape == (len(y) + 6,)
    assert np.all(np.isfinite(out))
