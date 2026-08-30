from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parent

STATIONARITY_LIMIT = 1.0 - 1e-6
NEAR_INTEGRATED_LIMIT = 0.995


def fit_model(
    returns: pd.Series,
    model_name: str,
):
    asymmetric = model_name == "GJR-GARCH"

    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1 if asymmetric else 0,
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


def extract_diagnostics(
    asset: str,
    model_name: str,
    result,
) -> dict:
    params = result.params

    alpha = float(
        params.get("alpha[1]", 0.0)
    )

    gamma = float(
        params.get("gamma[1]", 0.0)
    )

    beta = float(
        params.get("beta[1]", 0.0)
    )

    if model_name == "GJR-GARCH":
        persistence = (
            alpha
            + 0.5 * gamma
            + beta
        )
    else:
        persistence = alpha + beta

    standardized_residuals = (
        result.std_resid
        .dropna()
        .astype(float)
    )

    lb_residual = acorr_ljungbox(
        standardized_residuals,
        lags=[10],
        return_df=True,
    )

    lb_squared = acorr_ljungbox(
        standardized_residuals**2,
        lags=[10],
        return_df=True,
    )

    converged = result.convergence_flag == 0

    stationary = (
        np.isfinite(persistence)
        and persistence < STATIONARITY_LIMIT
    )

    return {
        "asset": asset,
        "model": model_name,
        "mu": float(
            params.get("mu", np.nan)
        ),
        "omega": float(
            params.get("omega", np.nan)
        ),
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "nu": float(
            params.get("nu", np.nan)
        ),
        "persistence": persistence,
        "convergence_flag": int(
            result.convergence_flag
        ),
        "converged": converged,
        "stationary": stationary,
        "near_integrated": (
            persistence >= NEAR_INTEGRATED_LIMIT
        ),
        "log_likelihood": float(
            result.loglikelihood
        ),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "lb_resid_pvalue_10": float(
            lb_residual["lb_pvalue"].iloc[0]
        ),
        "lb_squared_pvalue_10": float(
            lb_squared["lb_pvalue"].iloc[0]
        ),
        "observations": int(result.nobs),
    }


def choose_model(
    candidate_rows: list[dict],
) -> tuple[dict, str]:
    eligible = [
        row
        for row in candidate_rows
        if row["converged"]
        and row["stationary"]
    ]

    if eligible:
        selected = min(
            eligible,
            key=lambda row: row["bic"],
        )

        return selected, "lowest_bic_stationary"

    converged = [
        row
        for row in candidate_rows
        if row["converged"]
    ]

    if converged:
        selected = min(
            converged,
            key=lambda row: row["persistence"],
        )

        return selected, "boundary_fallback"

    raise RuntimeError(
        f"No model converged for "
        f"{candidate_rows[0]['asset']}."
    )


def main() -> None:
    return_path = (
        ROOT
        / "data"
        / "processed"
        / "common_returns_pct.csv"
    )

    returns = pd.read_csv(
        return_path,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    if returns.isna().any().any():
        raise ValueError(
            "Common return matrix contains missing values."
        )

    selected_residuals = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    selected_volatility = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    comparison_rows = []
    selected_rows = []

    for asset in returns.columns:
        print(f"\nSelecting volatility model: {asset}")

        fitted_results = {}
        asset_rows = []

        for model_name in [
            "GARCH",
            "GJR-GARCH",
        ]:
            print(f"  Fitting {model_name}")

            result = fit_model(
                returns[asset],
                model_name,
            )

            fitted_results[model_name] = result

            row = extract_diagnostics(
                asset,
                model_name,
                result,
            )

            asset_rows.append(row)
            comparison_rows.append(row)

        selected, reason = choose_model(
            asset_rows
        )

        selected_model = selected["model"]
        selected_result = fitted_results[
            selected_model
        ]

        selected_residuals.loc[
            selected_result.std_resid.index,
            asset,
        ] = selected_result.std_resid

        selected_volatility.loc[
            selected_result
            .conditional_volatility
            .index,
            asset,
        ] = (
            selected_result
            .conditional_volatility
        )

        selected_row = selected.copy()
        selected_row["selection_reason"] = reason

        selected_rows.append(selected_row)

        print(
            f"  Selected {selected_model}: "
            f"BIC={selected['bic']:.2f}, "
            f"persistence="
            f"{selected['persistence']:.6f}"
        )

    comparison = (
        pd.DataFrame(comparison_rows)
        .set_index(["asset", "model"])
        .sort_index()
    )

    selected_summary = (
        pd.DataFrame(selected_rows)
        .set_index("asset")
        .sort_index()
    )

    processed = ROOT / "data" / "processed"
    outputs = ROOT / "outputs"

    selected_residuals.to_csv(
        processed
        / "selected_standardized_residuals.csv"
    )

    selected_volatility.to_csv(
        processed
        / "selected_conditional_volatility_pct.csv"
    )

    comparison.to_csv(
        outputs
        / "volatility_model_comparison.csv"
    )

    selected_summary.to_csv(
        outputs
        / "selected_volatility_models.csv"
    )

    columns = [
        "model",
        "persistence",
        "nu",
        "bic",
        "lb_resid_pvalue_10",
        "lb_squared_pvalue_10",
        "near_integrated",
        "selection_reason",
    ]

    print("\nSelected volatility models:")
    print(
        selected_summary[
            columns
        ].to_string()
    )

    diagnostic_failures = selected_summary[
        (
            selected_summary[
                "lb_resid_pvalue_10"
            ] < 0.05
        )
        |
        (
            selected_summary[
                "lb_squared_pvalue_10"
            ] < 0.05
        )
    ]

    if not diagnostic_failures.empty:
        print(
            "\nWARNING: Ljung-Box diagnostic "
            "failed at the 5% level:"
        )

        print(
            diagnostic_failures[
                [
                    "model",
                    "lb_resid_pvalue_10",
                    "lb_squared_pvalue_10",
                ]
            ].to_string()
        )

    near_integrated = selected_summary[
        selected_summary["near_integrated"]
    ]

    if not near_integrated.empty:
        print(
            "\nWARNING: Near-integrated "
            "volatility models:"
        )

        print(
            near_integrated[
                [
                    "model",
                    "persistence",
                ]
            ].to_string()
        )

    print(
        "\nSaved selected standardized residuals."
    )


if __name__ == "__main__":
    main()