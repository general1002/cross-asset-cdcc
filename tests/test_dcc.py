import numpy as np

from src.dcc import (
    covariance_to_correlation,
    filter_correlations,
    forecast_next_correlation,
    parameters_from_unconstrained,
)


def test_parameter_constraints() -> None:
    a, b = parameters_from_unconstrained(
        np.array([1.0, 3.0])
    )

    assert a >= 0
    assert b >= 0
    assert a + b < 0.999


def test_covariance_to_correlation() -> None:
    covariance = np.array(
        [
            [4.0, 1.0],
            [1.0, 9.0],
        ]
    )

    correlation = covariance_to_correlation(
        covariance
    )

    assert np.allclose(
        np.diag(correlation),
        1.0,
    )

    assert np.isclose(
        correlation[0, 1],
        1.0 / 6.0,
    )


def test_dcc_filter_shapes() -> None:
    rng = np.random.default_rng(42)

    residuals = rng.normal(
        size=(100, 3)
    )

    target = np.eye(3)

    q_history, r_history = filter_correlations(
        residuals,
        a=0.03,
        b=0.95,
        target=target,
        corrected=True,
    )

    assert q_history.shape == (100, 3, 3)
    assert r_history.shape == (100, 3, 3)

    assert np.allclose(
        np.diagonal(
            r_history,
            axis1=1,
            axis2=2,
        ),
        1.0,
    )


def test_forecast_next_correlation() -> None:
    rng = np.random.default_rng(123)

    residuals = rng.normal(
        size=(120, 4)
    )

    target = np.corrcoef(
        residuals,
        rowvar=False,
    )

    a = 0.03
    b = 0.94

    q_history, r_history = filter_correlations(
        residuals,
        a=a,
        b=b,
        target=target,
        corrected=True,
    )

    result = {
        "a": a,
        "b": b,
        "target": target,
        "q_history": q_history,
        "r_history": r_history,
    }

    forecast = forecast_next_correlation(
        result=result,
        latest_standardized_residual=residuals[-1],
        corrected=True,
    )

    assert forecast.shape == (4, 4)

    assert np.allclose(
        forecast,
        forecast.T,
    )

    assert np.allclose(
        np.diag(forecast),
        1.0,
    )

    assert np.all(
        np.linalg.eigvalsh(forecast) > 0
    )

    assert np.all(
        forecast <= 1.0 + 1e-10
    )

    assert np.all(
        forecast >= -1.0 - 1e-10
    )