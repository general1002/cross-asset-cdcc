from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def validate_covariance(
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if covariance.ndim != 2:
        raise ValueError(
            "Covariance must be a matrix."
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Covariance matrix must be square."
        )

    if not np.isfinite(covariance).all():
        raise ValueError(
            "Covariance contains non-finite values."
        )

    covariance = 0.5 * (
        covariance + covariance.T
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    if eigenvalues.min() <= 0:
        jitter = (
            abs(eigenvalues.min())
            + 1e-10
        )

        covariance = (
            covariance
            + jitter
            * np.eye(
                covariance.shape[0]
            )
        )

    return covariance


def equal_weight(
    asset_count: int,
) -> np.ndarray:
    if asset_count <= 0:
        raise ValueError(
            "Asset count must be positive."
        )

    return np.full(
        asset_count,
        1.0 / asset_count,
    )


def inverse_volatility_weight(
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = validate_covariance(
        covariance
    )

    volatility = np.sqrt(
        np.diag(covariance)
    )

    inverse_volatility = (
        1.0
        / np.maximum(
            volatility,
            1e-12,
        )
    )

    return (
        inverse_volatility
        / inverse_volatility.sum()
    )


def portfolio_variance(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    return float(
        weights
        @ covariance
        @ weights
    )


def portfolio_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            max(
                portfolio_variance(
                    weights,
                    covariance,
                ),
                0.0,
            )
        )
    )


def risk_contributions(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = validate_covariance(
        covariance
    )

    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    if volatility <= 0:
        raise ValueError(
            "Portfolio volatility is zero."
        )

    marginal_risk = (
        covariance @ weights
    ) / volatility

    return weights * marginal_risk


def minimum_variance_weight(
    covariance: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    covariance = validate_covariance(
        covariance
    )

    asset_count = covariance.shape[0]

    if max_weight * asset_count < 1.0:
        raise ValueError(
            "Maximum weight makes the "
            "constraints infeasible."
        )

    initial = equal_weight(
        asset_count
    )

    result = minimize(
        fun=lambda weights: (
            portfolio_variance(
                weights,
                covariance,
            )
        ),
        x0=initial,
        method="SLSQP",
        bounds=[
            (0.0, max_weight)
            for _ in range(asset_count)
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: (
                    weights.sum() - 1.0
                ),
            }
        ],
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum variance optimization "
            f"failed: {result.message}"
        )

    weights = np.maximum(
        result.x,
        0.0,
    )

    return weights / weights.sum()


def risk_parity_weight(
    covariance: np.ndarray,
    max_weight: float = 0.20,
) -> np.ndarray:
    covariance = validate_covariance(
        covariance
    )

    asset_count = covariance.shape[0]

    if max_weight * asset_count < 1.0:
        raise ValueError(
            "Maximum weight makes the "
            "constraints infeasible."
        )

    initial = inverse_volatility_weight(
        covariance
    )

    initial = np.minimum(
        initial,
        max_weight,
    )

    initial = (
        initial
        / initial.sum()
    )

    def objective(
        weights: np.ndarray,
    ) -> float:
        contributions = risk_contributions(
            weights,
            covariance,
        )

        target = (
            contributions.sum()
            / asset_count
        )

        return float(
            np.sum(
                (
                    contributions
                    - target
                )
                ** 2
            )
        )

    result = minimize(
        fun=objective,
        x0=initial,
        method="SLSQP",
        bounds=[
            (1e-8, max_weight)
            for _ in range(asset_count)
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: (
                    weights.sum() - 1.0
                ),
            }
        ],
        options={
            "maxiter": 3000,
            "ftol": 1e-14,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Risk parity optimization failed: "
            f"{result.message}"
        )

    weights = np.maximum(
        result.x,
        0.0,
    )

    return weights / weights.sum()