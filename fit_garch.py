from pathlib import Path

import pandas as pd

from src.univariate_garch import fit_all_gjr_garch


ROOT = Path(__file__).resolve().parent


def main() -> None:
    return_path = (
        ROOT
        / "data"
        / "processed"
        / "common_returns_pct.csv"
    )

    if not return_path.exists():
        raise FileNotFoundError(
            "common_returns_pct.csv was not found. "
            "Run diagnose_data.py and create the common "
            "return matrix first."
        )

    returns = pd.read_csv(
        return_path,
        index_col=0,
        parse_dates=True,
    )

    returns = returns.sort_index()

    print(
        f"Fitting {returns.shape[1]} assets "
        f"using {returns.shape[0]} observations."
    )

    (
        standardized_residuals,
        conditional_volatility,
        summary,
    ) = fit_all_gjr_garch(returns)

    processed = ROOT / "data" / "processed"
    outputs = ROOT / "outputs"

    processed.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    standardized_residuals.to_csv(
        processed
        / "standardized_residuals.csv"
    )

    conditional_volatility.to_csv(
        processed
        / "conditional_volatility_pct.csv"
    )

    summary.to_csv(
        outputs
        / "garch_summary.csv"
    )

    print("\nGJR-GARCH estimation completed.")
    print("\nParameter summary:")
    print(
        summary[
            [
                "alpha",
                "gamma",
                "beta",
                "nu",
                "persistence",
                "converged",
            ]
        ].to_string()
    )

    unstable = summary[
        summary["persistence"] >= 1.0
    ]

    failed = summary[
        ~summary["converged"]
    ]

    if not unstable.empty:
        print(
            "\nWARNING: Persistence is at least one:"
        )
        print(
            unstable[
                ["persistence"]
            ].to_string()
        )

    if not failed.empty:
        print(
            "\nWARNING: Optimization did not converge:"
        )
        print(
            failed[
                ["convergence_flag"]
            ].to_string()
        )

    print(
        "\nSaved standardized residuals to:"
    )
    print(
        processed
        / "standardized_residuals.csv"
    )

    print(
        "\nSaved conditional volatility to:"
    )
    print(
        processed
        / "conditional_volatility_pct.csv"
    )

    print(
        "\nSaved parameter summary to:"
    )
    print(
        outputs
        / "garch_summary.csv"
    )


if __name__ == "__main__":
    main()