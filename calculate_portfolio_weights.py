from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.portfolio import (
    equal_weight,
    inverse_volatility_weight,
    minimum_variance_weight,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weight,
)


ROOT = Path(__file__).resolve().parent

LOOKBACK = 252
MAX_WEIGHT = 0.20
TRADING_DAYS = 252


def calculate_strategy_summary(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> dict:
    annualized_covariance = (
        covariance * TRADING_DAYS
    )

    annualized_volatility = (
        portfolio_volatility(
            weights,
            annualized_covariance,
        )
    )

    concentration = float(
        np.sum(weights**2)
    )

    effective_assets = (
        1.0 / concentration
    )

    return {
        "annualized_volatility_pct": (
            annualized_volatility
        ),
        "maximum_weight_pct": (
            100.0 * weights.max()
        ),
        "effective_asset_count": (
            effective_assets
        ),
        "concentration_hhi": (
            concentration
        ),
    }


def main() -> None:
    return_path = (
        ROOT
        / "data"
        / "processed"
        / "long_only_returns_pct.csv"
    )

    returns = pd.read_csv(
        return_path,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    if len(returns) < LOOKBACK:
        raise ValueError(
            f"At least {LOOKBACK} observations "
            "are required."
        )

    estimation_sample = returns.tail(
        LOOKBACK
    )

    estimator = LedoitWolf().fit(
        estimation_sample.to_numpy(
            dtype=float
        )
    )

    covariance = estimator.covariance_

    assets = returns.columns.tolist()
    asset_count = len(assets)

    strategies = {
        "Equal Weight": equal_weight(
            asset_count
        ),
        "Inverse Volatility": (
            inverse_volatility_weight(
                covariance
            )
        ),
        "Risk Parity": risk_parity_weight(
            covariance,
            max_weight=MAX_WEIGHT,
        ),
        "Minimum Variance": (
            minimum_variance_weight(
                covariance,
                max_weight=MAX_WEIGHT,
            )
        ),
    }

    weight_table = pd.DataFrame(
        {
            strategy: (
                100.0 * weights
            )
            for strategy, weights
            in strategies.items()
        },
        index=assets,
    )

    annualized_covariance = (
        covariance * TRADING_DAYS
    )

    risk_contribution_table = (
        pd.DataFrame(
            {
                strategy: (
                    100.0
                    * risk_contributions(
                        weights,
                        annualized_covariance,
                    )
                    / portfolio_volatility(
                        weights,
                        annualized_covariance,
                    )
                )
                for strategy, weights
                in strategies.items()
            },
            index=assets,
        )
    )

    summaries = pd.DataFrame(
        {
            strategy: (
                calculate_strategy_summary(
                    weights,
                    covariance,
                )
            )
            for strategy, weights
            in strategies.items()
        }
    ).T

    outputs = ROOT / "outputs"

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    weight_table.to_csv(
        outputs
        / "latest_portfolio_weights_pct.csv"
    )

    risk_contribution_table.to_csv(
        outputs
        / "latest_risk_contributions_pct.csv"
    )

    summaries.to_csv(
        outputs
        / "latest_portfolio_summary.csv"
    )

    pd.DataFrame(
        covariance,
        index=assets,
        columns=assets,
    ).to_csv(
        outputs
        / "latest_shrinkage_covariance.csv"
    )

    print(
        f"Estimation window: "
        f"{estimation_sample.index.min().date()} "
        f"to "
        f"{estimation_sample.index.max().date()}"
    )

    print(
        f"Ledoit-Wolf shrinkage: "
        f"{estimator.shrinkage_:.4f}"
    )

    print(
        "\nPortfolio weights (%):"
    )

    print(
        weight_table.round(2).to_string()
    )

    print(
        "\nPortfolio summary:"
    )

    print(
        summaries.round(4).to_string()
    )

    print(
        "\nRisk contributions (% of total risk):"
    )

    print(
        risk_contribution_table
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()