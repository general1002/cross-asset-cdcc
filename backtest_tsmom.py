from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent

LOOKBACK = 252
VOLATILITY_LOOKBACK = 63
MOMENTUM_WINDOWS = [63, 126, 252]

TARGET_VOLATILITY = 0.10
MAX_GROSS_EXPOSURE = 2.0
BASE_SIDE_EXPOSURE = 0.50
BASE_ASSET_CAP = 0.20

TRADING_DAYS = 252

ETF_COST_BPS = 5.0
FX_COST_BPS = 2.0
COMMODITY_COST_BPS = 10.0

FX_ASSETS = {
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
}

COMMODITY_ASSETS = {
    "Crude Oil",
    "Gold",
    "Copper",
}


def transaction_cost_vector(
    assets: list[str],
) -> np.ndarray:
    costs = []

    for asset in assets:
        if asset in FX_ASSETS:
            costs.append(FX_COST_BPS)
        elif asset in COMMODITY_ASSETS:
            costs.append(COMMODITY_COST_BPS)
        else:
            costs.append(ETF_COST_BPS)

    return np.asarray(
        costs,
        dtype=float,
    )


def momentum_signal(
    log_returns: pd.DataFrame,
) -> pd.Series:
    signals = []

    for window in MOMENTUM_WINDOWS:
        cumulative_log_return = (
            log_returns
            .tail(window)
            .sum()
        )

        signals.append(
            np.sign(
                cumulative_log_return
            )
        )

    return (
        pd.concat(
            signals,
            axis=1,
        )
        .mean(axis=1)
    )


def allocate_with_cap(
    raw_scores: np.ndarray,
    total_exposure: float,
    cap: float,
) -> np.ndarray:
    raw_scores = np.maximum(
        np.asarray(
            raw_scores,
            dtype=float,
        ),
        0.0,
    )

    allocation = np.zeros_like(
        raw_scores
    )

    active = raw_scores > 0

    if not active.any():
        return allocation

    active_count = int(active.sum())

    # Ensure feasibility when only a few assets
    # are on one side of the portfolio.
    effective_cap = max(
        cap,
        total_exposure / active_count,
    )

    remaining = total_exposure

    while active.any():
        active_scores = raw_scores[
            active
        ]

        proposed = (
            remaining
            * active_scores
            / active_scores.sum()
        )

        capped = (
            proposed > effective_cap
        )

        active_indices = np.where(
            active
        )[0]

        if not capped.any():
            allocation[
                active_indices
            ] = proposed
            break

        capped_indices = (
            active_indices[capped]
        )

        allocation[
            capped_indices
        ] = effective_cap

        remaining -= (
            effective_cap
            * len(capped_indices)
        )

        active[
            capped_indices
        ] = False

        if remaining <= 1e-12:
            break

    return allocation


def build_signal_weights(
    signal: pd.Series,
    annualized_volatility: pd.Series,
) -> np.ndarray:
    volatility = np.maximum(
        annualized_volatility.to_numpy(
            dtype=float
        ),
        1e-6,
    )

    signal_values = signal.to_numpy(
        dtype=float
    )

    positive_scores = (
        np.maximum(
            signal_values,
            0.0,
        )
        / volatility
    )

    negative_scores = (
        np.maximum(
            -signal_values,
            0.0,
        )
        / volatility
    )

    # Require both long and short positions
    # to preserve dollar neutrality.
    if (
        positive_scores.sum() <= 0
        or negative_scores.sum() <= 0
    ):
        return np.zeros_like(
            signal_values
        )

    long_weights = allocate_with_cap(
        positive_scores,
        total_exposure=BASE_SIDE_EXPOSURE,
        cap=BASE_ASSET_CAP,
    )

    short_weights = allocate_with_cap(
        negative_scores,
        total_exposure=BASE_SIDE_EXPOSURE,
        cap=BASE_ASSET_CAP,
    )

    return long_weights - short_weights


def scale_to_target_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    annualized_covariance = (
        covariance * TRADING_DAYS
    )

    predicted_volatility = float(
        np.sqrt(
            max(
                weights
                @ annualized_covariance
                @ weights,
                0.0,
            )
        )
    )

    if predicted_volatility <= 0:
        return weights, 0.0

    current_gross = float(
        np.abs(weights).sum()
    )

    volatility_scale = (
        TARGET_VOLATILITY
        / predicted_volatility
    )

    gross_scale_limit = (
        MAX_GROSS_EXPOSURE
        / current_gross
        if current_gross > 0
        else 0.0
    )

    scale = min(
        volatility_scale,
        gross_scale_limit,
    )

    return (
        weights * scale,
        predicted_volatility * scale,
    )


def calculate_performance(
    daily_returns: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
) -> dict:
    nav = (
        1.0 + daily_returns
    ).cumprod()

    observations = len(daily_returns)
    years = observations / TRADING_DAYS

    annualized_return = (
        nav.iloc[-1]
        ** (1.0 / years)
        - 1.0
    )

    annualized_volatility = (
        daily_returns.std()
        * np.sqrt(TRADING_DAYS)
    )

    sharpe = (
        daily_returns.mean()
        * TRADING_DAYS
        / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    drawdown = (
        nav / nav.cummax() - 1.0
    )

    maximum_drawdown = float(
        drawdown.min()
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
        "calmar_ratio": (
            annualized_return
            / abs(maximum_drawdown)
            if maximum_drawdown < 0
            else np.nan
        ),
        "annualized_turnover_pct": (
            100.0 * turnover.sum() / years
        ),
        "total_transaction_cost_pct": (
            100.0 * costs.sum()
        ),
        "final_nav": float(
            nav.iloc[-1]
        ),
        "observations": observations,
    }


def main() -> None:
    path = (
        ROOT
        / "data"
        / "processed"
        / "long_short_returns_pct.csv"
    )

    log_returns_pct = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    if log_returns_pct.isna().any().any():
        raise ValueError(
            "Return matrix contains missing values."
        )

    # Input consists of percentage log returns.
    log_returns = log_returns_pct / 100.0

    # Convert to simple returns for portfolio P&L.
    simple_returns = np.expm1(
        log_returns
    )

    assets = log_returns.columns.tolist()

    cost_bps = transaction_cost_vector(
        assets
    )

    dates = log_returns.index[
        LOOKBACK:
    ]

    current_weights = np.zeros(
        len(assets),
        dtype=float,
    )

    daily_portfolio_returns = []
    daily_turnover = []
    daily_costs = []

    weight_rows = []
    signal_rows = []

    previous_month = None

    for position in range(
        LOOKBACK,
        len(log_returns),
    ):
        date = log_returns.index[
            position
        ]

        month = (
            date.year,
            date.month,
        )

        rebalance = (
            previous_month is None
            or month != previous_month
        )

        turnover = 0.0
        cost = 0.0

        if rebalance:
            historical_log_returns = (
                log_returns.iloc[
                    position - LOOKBACK:
                    position
                ]
            )

            historical_simple_returns = (
                simple_returns.iloc[
                    position - LOOKBACK:
                    position
                ]
            )

            signal = momentum_signal(
                historical_log_returns
            )

            annualized_volatility = (
                historical_simple_returns
                .tail(
                    VOLATILITY_LOOKBACK
                )
                .std()
                * np.sqrt(TRADING_DAYS)
            )

            base_weights = (
                build_signal_weights(
                    signal,
                    annualized_volatility,
                )
            )

            covariance = (
                LedoitWolf()
                .fit(
                    historical_simple_returns
                    .to_numpy(dtype=float)
                )
                .covariance_
            )

            target_weights, predicted_vol = (
                scale_to_target_volatility(
                    base_weights,
                    covariance,
                )
            )

            weight_change = (
                target_weights
                - current_weights
            )

            turnover = float(
                np.abs(
                    weight_change
                ).sum()
            )

            cost = float(
                np.sum(
                    np.abs(weight_change)
                    * cost_bps
                    / 10_000.0
                )
            )

            current_weights = (
                target_weights
            )

            for asset, weight in zip(
                assets,
                current_weights,
            ):
                weight_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "weight": weight,
                        "signal": signal[asset],
                        "predicted_volatility": (
                            predicted_vol
                        ),
                    }
                )

                signal_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "signal": signal[asset],
                    }
                )

        asset_returns = (
            simple_returns
            .iloc[position]
            .to_numpy(dtype=float)
        )

        gross_return = float(
            current_weights
            @ asset_returns
        )

        net_return = (
            gross_return - cost
        )

        daily_portfolio_returns.append(
            net_return
        )

        daily_turnover.append(
            turnover
        )

        daily_costs.append(
            cost
        )

        previous_month = month

    portfolio_returns = pd.Series(
        daily_portfolio_returns,
        index=dates,
        name="TSMOM",
    )

    turnover_series = pd.Series(
        daily_turnover,
        index=dates,
        name="turnover",
    )

    cost_series = pd.Series(
        daily_costs,
        index=dates,
        name="cost",
    )

    nav = (
        1.0 + portfolio_returns
    ).cumprod()

    weight_history = pd.DataFrame(
        weight_rows
    )

    signal_history = pd.DataFrame(
        signal_rows
    )

    performance = calculate_performance(
        portfolio_returns,
        turnover_series,
        cost_series,
    )

    outputs = ROOT / "outputs"

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    portfolio_returns.to_csv(
        outputs
        / "tsmom_daily_returns.csv"
    )

    nav.to_csv(
        outputs
        / "tsmom_nav.csv"
    )

    weight_history.to_csv(
        outputs
        / "tsmom_weights.csv",
        index=False,
    )

    signal_history.to_csv(
        outputs
        / "tsmom_signals.csv",
        index=False,
    )

    pd.DataFrame(
        [performance],
        index=["TSMOM"],
    ).to_csv(
        outputs
        / "tsmom_summary.csv"
    )

    latest_date = (
        weight_history["date"].max()
    )

    latest = (
        weight_history[
            weight_history["date"]
            == latest_date
        ]
        .sort_values(
            "weight",
            ascending=False,
        )
    )

    print(
        f"Backtest period: "
        f"{dates.min().date()} "
        f"to {dates.max().date()}"
    )

    print(
        "\nTSMOM performance:"
    )

    print(
        pd.Series(performance)
        .round(4)
        .to_string()
    )

    print(
        "\nLatest signals and weights:"
    )

    print(
        latest[
            [
                "asset",
                "signal",
                "weight",
            ]
        ]
        .assign(
            weight=lambda frame: (
                100.0 * frame["weight"]
            )
        )
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nLatest net exposure:"
    )

    print(
        f"{100.0 * latest['weight'].sum():.2f}%"
    )

    print(
        "Latest gross exposure:"
    )

    print(
        f"{100.0 * latest['weight'].abs().sum():.2f}%"
    )


if __name__ == "__main__":
    main()