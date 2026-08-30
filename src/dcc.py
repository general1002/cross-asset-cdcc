from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


PARAMETER_CAP = 0.999


def parameters_from_unconstrained(
    theta: np.ndarray,
) -> tuple[float, float]:
    """
    Convert unconstrained parameters into a >= 0, b >= 0,
    and a + b < PARAMETER_CAP.
    """
    theta = np.asarray(theta, dtype=float)

    shifted = theta - max(
        0.0,
        float(np.max(theta)),
    )

    exp_theta = np.exp(shifted)
    baseline = np.exp(
        -max(0.0, float(np.max(theta)))
    )

    denominator = baseline + exp_theta.sum()

    a = (
        PARAMETER_CAP
        * exp_theta[0]
        / denominator
    )

    b = (
        PARAMETER_CAP
        * exp_theta[1]
        / denominator
    )

    return float(a), float(b)


def unconstrained_from_parameters(
    a: float,
    b: float,
) -> np.ndarray:
    remainder = PARAMETER_CAP - a - b

    if a <= 0 or b <= 0 or remainder <= 0:
        raise ValueError(
            "Initial parameters must satisfy "
            "a > 0, b > 0 and a + b < 0.999."
        )

    return np.array(
        [
            np.log(a / remainder),
            np.log(b / remainder),
        ],
        dtype=float,
    )


def covariance_to_correlation(
    matrix: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    diagonal = np.maximum(
        np.diag(matrix),
        epsilon,
    )

    scale = np.sqrt(diagonal)

    correlation = (
        matrix
        / np.outer(scale, scale)
    )

    correlation = 0.5 * (
        correlation + correlation.T
    )

    np.fill_diagonal(
        correlation,
        1.0,
    )

    return correlation


def ledoit_wolf_target(
    standardized_residuals: np.ndarray,
) -> tuple[np.ndarray, float]:
    estimator = LedoitWolf().fit(
        standardized_residuals
    )

    covariance = estimator.covariance_

    correlation = covariance_to_correlation(
        covariance
    )

    return (
        correlation,
        float(estimator.shrinkage_),
    )


def filter_correlations(
    standardized_residuals: np.ndarray,
    a: float,
    b: float,
    target: np.ndarray,
    corrected: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run standard DCC or corrected-DCC recursion.

    corrected=False:
        innovation = z[t-1]

    corrected=True:
        innovation = sqrt(diag(Q[t-1])) * z[t-1]
    """
    z = np.asarray(
        standardized_residuals,
        dtype=float,
    )

    observations, assets = z.shape

    q_history = np.empty(
        (observations, assets, assets),
        dtype=float,
    )

    r_history = np.empty_like(
        q_history
    )

    q = np.asarray(
        target,
        dtype=float,
    ).copy()

    for t in range(observations):
        if t > 0:
            innovation = z[t - 1]

            if corrected:
                q_scale = np.sqrt(
                    np.maximum(
                        np.diag(q),
                        1e-12,
                    )
                )

                innovation = (
                    q_scale * innovation
                )

            q = (
                (1.0 - a - b) * target
                + a
                * np.outer(
                    innovation,
                    innovation,
                )
                + b * q
            )

            q = 0.5 * (
                q + q.T
            )

        correlation = covariance_to_correlation(
            q
        )

        q_history[t] = q
        r_history[t] = correlation

    return q_history, r_history


def correlation_negative_loglikelihood(
    theta: np.ndarray,
    standardized_residuals: np.ndarray,
    target: np.ndarray,
    corrected: bool,
) -> float:
    a, b = parameters_from_unconstrained(
        theta
    )

    _, correlations = filter_correlations(
        standardized_residuals,
        a=a,
        b=b,
        target=target,
        corrected=corrected,
    )

    objective = 0.0

    for t, correlation in enumerate(
        correlations
    ):
        try:
            cholesky = np.linalg.cholesky(
                correlation
            )
        except np.linalg.LinAlgError:
            return 1e12

        log_determinant = (
            2.0
            * np.log(
                np.diag(cholesky)
            ).sum()
        )

        solved = np.linalg.solve(
            cholesky,
            standardized_residuals[t],
        )

        quadratic = float(
            solved @ solved
        )

        objective += 0.5 * (
            log_determinant
            + quadratic
        )

    if not np.isfinite(objective):
        return 1e12

    return float(objective)


def estimate_dcc(
    standardized_residuals: np.ndarray,
    corrected: bool,
    initial_a: float = 0.02,
    initial_b: float = 0.95,
) -> dict:
    z = np.asarray(
        standardized_residuals,
        dtype=float,
    )

    if z.ndim != 2:
        raise ValueError(
            "Standardized residuals must be a 2D matrix."
        )

    if not np.isfinite(z).all():
        raise ValueError(
            "Standardized residuals contain missing "
            "or non-finite values."
        )

    target, shrinkage = ledoit_wolf_target(
        z
    )

    initial_theta = (
        unconstrained_from_parameters(
            initial_a,
            initial_b,
        )
    )

    optimization = minimize(
        correlation_negative_loglikelihood,
        x0=initial_theta,
        args=(
            z,
            target,
            corrected,
        ),
        method="L-BFGS-B",
        options={
            "maxiter": 500,
            "ftol": 1e-9,
            "gtol": 1e-6,
        },
    )

    a, b = parameters_from_unconstrained(
        optimization.x
    )

    q_history, r_history = (
        filter_correlations(
            z,
            a=a,
            b=b,
            target=target,
            corrected=corrected,
        )
    )

    return {
        "model": (
            "cDCC"
            if corrected
            else "DCC"
        ),
        "a": a,
        "b": b,
        "persistence": a + b,
        "negative_loglikelihood": float(
            optimization.fun
        ),
        "converged": bool(
            optimization.success
        ),
        "optimization_message": str(
            optimization.message
        ),
        "iterations": int(
            optimization.nit
        ),
        "shrinkage": shrinkage,
        "target": target,
        "q_history": q_history,
        "r_history": r_history,
    }
    
def forecast_next_correlation(
    result: dict,
    latest_standardized_residual: np.ndarray,
    corrected: bool,
) -> np.ndarray:
    """
    Produce the one-step-ahead DCC/cDCC correlation forecast.
    """
    latest_standardized_residual = np.asarray(
        latest_standardized_residual,
        dtype=float,
    )

    q = np.asarray(
        result["q_history"][-1],
        dtype=float,
    )

    target = np.asarray(
        result["target"],
        dtype=float,
    )

    a = float(result["a"])
    b = float(result["b"])

    innovation = (
        latest_standardized_residual
    )

    if corrected:
        q_scale = np.sqrt(
            np.maximum(
                np.diag(q),
                1e-12,
            )
        )

        innovation = (
            q_scale * innovation
        )

    q_forecast = (
        (1.0 - a - b) * target
        + a
        * np.outer(
            innovation,
            innovation,
        )
        + b * q
    )

    q_forecast = 0.5 * (
        q_forecast + q_forecast.T
    )

    return covariance_to_correlation(
        q_forecast
    )