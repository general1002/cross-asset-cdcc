from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.portfolio import (
    equal_weight,
    inverse_volatility_weight,
    minimum_variance_weight,
    risk_parity_weight,
)


ROOT = Path(__file__).resolve().parent

LOOKBACK = 252
MAX_WEIGHT = 0.20
TRANSACTION_COST_BPS = 5.0
TRADING_DAYS = 252


def calculate_targets(
    estimation_returns: pd.DataFrame,
) -> dict[str, np.ndarray]:
    estimator = LedoitWolf().fit(
        estimation_returns.to_numpy(
            dtype=float
        )
    )

    covariance = estimator.covariance_
    asset_count = covariance.shape[0]

    return {
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


def drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    ending_values = (
        weights
        * (1.0 + asset_returns)
    )

    total_value = ending_values.sum()

    if total_value <= 0:
        raise RuntimeError(
            "Portfolio value became non-positive."
        )

    return ending_values / total_value


def calculate_performance(
    daily_returns: pd.Series,
    turnover: pd.Series,
    transaction_cost: pd.Series,
) -> dict:
    clean_returns = daily_returns.dropna()

    observations = len(clean_returns)

    nav = (
        1.0 + clean_returns
    ).cumprod()

    years = (
        observations / TRADING_DAYS
    )

    annualized_return = (
        nav.iloc[-1] ** (1.0 / years)
        - 1.0
    )

    annualized_volatility = (
        clean_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    annualized_mean_return = (
        clean_returns.mean()
        * TRADING_DAYS
    )

    sharpe = (
        annualized_mean_return
        / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    drawdown = (
        nav / nav.cummax()
        - 1.0
    )

    maximum_drawdown = (
        drawdown.min()
    )

    calmar = (
        annualized_return
        / abs(maximum_drawdown)
        if maximum_drawdown < 0
        else np.nan
    )

    annualized_turnover = (
        turnover.sum() / years
    )

    return {
        "annualized_return_pct": (
            100.0 * annualized_return
        ),
        "annualized_volatility_pct": (
            100.0 * annualized_volatility
        ),
        "sharpe_ratio": sharpe,
        "maximum_drawdown_pct": (
            100.0 * maximum_drawdown
        ),
        "calmar_ratio": calmar,
        "annualized_turnover_pct": (
            100.0 * annualized_turnover
        ),
        "total_transaction_cost_pct": (
            100.0
            * transaction_cost.sum()
        ),
        "observations": observations,
    }


def main() -> None:
    path = (
        ROOT
        / "data"
        / "processed"
        / "long_only_returns_pct.csv"
    )

    returns_pct = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    if returns_pct.isna().any().any():
        raise ValueError(
            "Return matrix contains missing values."
        )

    # Convert percentage returns to decimal.
    returns = returns_pct / 100.0

    strategies = [
        "Equal Weight",
        "Inverse Volatility",
        "Risk Parity",
        "Minimum Variance",
    ]

    current_weights = {
        strategy: None
        for strategy in strategies
    }

    portfolio_returns = pd.DataFrame(
        index=returns.index,
        columns=strategies,
        dtype=float,
    )

    turnover_history = pd.DataFrame(
        0.0,
        index=returns.index,
        columns=strategies,
    )

    cost_history = pd.DataFrame(
        0.0,
        index=returns.index,
        columns=strategies,
    )

    weight_rows = []

    previous_month = None

    for position in range(
        LOOKBACK,
        len(returns),
    ):
        date = returns.index[position]

        current_month = (
            date.year,
            date.month,
        )

        is_rebalance = (
            previous_month is None
            or current_month != previous_month
        )

        if is_rebalance:
            estimation_returns = (
                returns_pct.iloc[
                    position - LOOKBACK:
                    position
                ]
            )

            # Only information through the previous
            # observation is used.
            targets = calculate_targets(
                estimation_returns
            )

            for strategy in strategies:
                target = targets[strategy]

                existing = current_weights[
                    strategy
                ]

                if existing is None:
                    turnover = 1.0
                else:
                    # One-way traded notional.
                    turnover = (
                        0.5
                        * np.abs(
                            target - existing
                        ).sum()
                    )

                cost = (
                    turnover
                    * TRANSACTION_COST_BPS
                    / 10_000.0
                )

                current_weights[
                    strategy
                ] = target.copy()

                turnover_history.loc[
                    date,
                    strategy,
                ] = turnover

                cost_history.loc[
                    date,
                    strategy,
                ] = cost

                row = {
                    "date": date,
                    "strategy": strategy,
                    "turnover": turnover,
                }

                row.update(
                    {
                        asset: weight
                        for asset, weight
                        in zip(
                            returns.columns,
                            target,
                        )
                    }
                )

                weight_rows.append(row)

            previous_month = current_month

        asset_returns = (
            returns.iloc[position]
            .to_numpy(dtype=float)
        )

        for strategy in strategies:
            weights = current_weights[
                strategy
            ]

            gross_return = float(
                weights @ asset_returns
            )

            net_return = (
                gross_return
                - cost_history.loc[
                    date,
                    strategy,
                ]
            )

            portfolio_returns.loc[
                date,
                strategy,
            ] = net_return

            current_weights[
                strategy
            ] = drift_weights(
                weights,
                asset_returns,
            )

    portfolio_returns = (
        portfolio_returns.dropna(
            how="all"
        )
    )

    turnover_history = (
        turnover_history.loc[
            portfolio_returns.index
        ]
    )

    cost_history = (
        cost_history.loc[
            portfolio_returns.index
        ]
    )

    nav = (
        1.0 + portfolio_returns
    ).cumprod()

    drawdowns = (
        nav / nav.cummax()
        - 1.0
    )

    summary = pd.DataFrame(
        {
            strategy: calculate_performance(
                portfolio_returns[strategy],
                turnover_history[strategy],
                cost_history[strategy],
            )
            for strategy in strategies
        }
    ).T

    weight_history = pd.DataFrame(
        weight_rows
    )

    outputs = ROOT / "outputs"

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    portfolio_returns.to_csv(
        outputs
        / "portfolio_backtest_returns.csv"
    )

    nav.to_csv(
        outputs
        / "portfolio_backtest_nav.csv"
    )

    drawdowns.to_csv(
        outputs
        / "portfolio_backtest_drawdowns.csv"
    )

    turnover_history.to_csv(
        outputs
        / "portfolio_backtest_turnover.csv"
    )

    cost_history.to_csv(
        outputs
        / "portfolio_backtest_costs.csv"
    )

    weight_history.to_csv(
        outputs
        / "portfolio_backtest_weights.csv",
        index=False,
    )

    summary.to_csv(
        outputs
        / "portfolio_backtest_summary.csv"
    )

    print(
        "Backtest period:"
    )

    print(
        portfolio_returns.index.min(),
        "to",
        portfolio_returns.index.max(),
    )

    print(
        "\nPerformance after transaction costs:"
    )

    print(
        summary.round(4).to_string()
    )

    print(
        "\nEnding NAV:"
    )

    print(
        nav.iloc[-1]
        .round(4)
        .to_string()
    )

    print(
        "\nOutputs saved under:"
    )

    print(outputs)


if __name__ == "__main__":
    main()