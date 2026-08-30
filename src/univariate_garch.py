from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def gjr_persistence(
    alpha: float,
    gamma: float,
    beta: float,
) -> float:
    """
    Approximate GJR-GARCH persistence under symmetric shocks.
    """
    return alpha + 0.5 * gamma + beta


def fit_single_gjr_garch(
    returns: pd.Series,
):
    """
    Fit constant-mean GJR-GARCH(1,1) with Student-t errors.

    Returns are expected to be expressed in percent.
    """
    clean_returns = returns.dropna().astype(float)

    if len(clean_returns) < 500:
        raise ValueError(
            f"{returns.name}: fewer than 500 observations."
        )

    model = arch_model(
        clean_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        power=2.0,
        dist="StudentsT",
        rescale=False,
    )

    result = model.fit(
        disp="off",
        show_warning=False,
        options={"maxiter": 2000},
    )

    return result


def fit_all_gjr_garch(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit one GJR-GARCH model per asset.

    Returns
    -------
    standardized_residuals
        Residual divided by conditional volatility.
    conditional_volatility
        Conditional volatility in daily percentage units.
    summary
        Parameters and convergence diagnostics.
    """
    if returns.empty:
        raise ValueError("Return matrix is empty.")

    if returns.isna().any().any():
        raise ValueError(
            "Return matrix contains missing observations."
        )

    standardized_residuals = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    conditional_volatility = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    summary_rows = []

    for asset in returns.columns:
        print(f"Fitting GJR-GARCH: {asset}")

        result = fit_single_gjr_garch(
            returns[asset]
        )

        params = result.params

        mu = float(params.get("mu", np.nan))
        omega = float(params.get("omega", np.nan))
        alpha = float(params.get("alpha[1]", np.nan))
        gamma = float(params.get("gamma[1]", np.nan))
        beta = float(params.get("beta[1]", np.nan))
        nu = float(params.get("nu", np.nan))

        persistence = gjr_persistence(
            alpha=alpha,
            gamma=gamma,
            beta=beta,
        )

        standardized_residuals.loc[
            result.std_resid.index,
            asset,
        ] = result.std_resid

        conditional_volatility.loc[
            result.conditional_volatility.index,
            asset,
        ] = result.conditional_volatility

        summary_rows.append(
            {
                "asset": asset,
                "mu": mu,
                "omega": omega,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "nu": nu,
                "persistence": persistence,
                "convergence_flag": int(
                    result.convergence_flag
                ),
                "converged": (
                    result.convergence_flag == 0
                ),
                "log_likelihood": float(
                    result.loglikelihood
                ),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "observations": int(result.nobs),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .set_index("asset")
        .sort_index()
    )

    return (
        standardized_residuals,
        conditional_volatility,
        summary,
    )