import numpy as np

from src.portfolio import (
    equal_weight,
    inverse_volatility_weight,
    minimum_variance_weight,
    risk_contributions,
    risk_parity_weight,
)


def test_equal_weight() -> None:
    weights = equal_weight(4)

    assert np.allclose(
        weights,
        0.25,
    )


def test_inverse_volatility_weight() -> None:
    covariance = np.diag(
        [0.01, 0.04]
    )

    weights = inverse_volatility_weight(
        covariance
    )

    assert np.allclose(
        weights,
        [2.0 / 3.0, 1.0 / 3.0],
    )


def test_minimum_variance_constraints() -> None:
    covariance = np.diag(
        [0.01, 0.02, 0.03]
    )

    weights = minimum_variance_weight(
        covariance,
        max_weight=0.60,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )

    assert np.all(weights >= 0)
    assert np.all(
        weights <= 0.60 + 1e-8
    )


def test_risk_parity_contributions() -> None:
    covariance = np.diag(
        [0.01, 0.04, 0.09]
    )

    weights = risk_parity_weight(
        covariance,
        max_weight=0.60,
    )

    contributions = risk_contributions(
        weights,
        covariance,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )

    assert np.allclose(
        contributions,
        contributions.mean(),
        atol=1e-5,
    )