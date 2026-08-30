from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from backtest_tsmom import (
    calculate_performance,
    momentum_signal,
    transaction_cost_vector,
)


ROOT = Path(__file__).resolve().parent

LOOKBACK = 252
VOLATILITY_LOOKBACK = 126

TARGET_VOLATILITY = 0.10
MAX_GROSS_EXPOSURE = 1.50

DEFAULT_ASSET_CAP = 0.15
FX_ASSET_CAP = 0.10

TRADING_DAYS = 252

FX_ASSETS = {
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
}


def build_tsmom_weights(
    signal: pd.Series,
    annualized_volatility: pd.Series,
) -> np.ndarray:
    volatility = np.maximum(
        annualized_volatility.to_numpy(
            dtype=float
        ),
        1e-6,
    )

    raw_weights = (
        signal.to_numpy(dtype=float)
        / volatility
    )

    gross = np.abs(
        raw_weights
    ).sum()

    if gross <= 0:
        return np.zeros_like(
            raw_weights
        )

    # Begin with 100% gross exposure.
    return raw_weights / gross


def scale_and_constrain_weights(
    weights: np.ndarray,
    covariance: np.ndarray,
    assets: list[str],
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

    current_gross = float(
        np.abs(weights).sum()
    )

    if (
        predicted_volatility <= 0
        or current_gross <= 0
    ):
        return weights, 0.0

    volatility_scale = (
        TARGET_VOLATILITY
        / predicted_volatility
    )

    gross_scale = (
        MAX_GROSS_EXPOSURE
        / current_gross
    )

    scale = min(
        volatility_scale,
        gross_scale,
    )

    scaled_weights = weights * scale

    caps = np.asarray(
        [
            (
                FX_ASSET_CAP
                if asset in FX_ASSETS
                else DEFAULT_ASSET_CAP
            )
            for asset in assets
        ],
        dtype=float,
    )

    constrained_weights = np.clip(
        scaled_weights,
        -caps,
        caps,
    )

    final_gross = float(
        np.abs(
            constrained_weights
        ).sum()
    )

    if final_gross > MAX_GROSS_EXPOSURE:
        constrained_weights *= (
            MAX_GROSS_EXPOSURE
            / final_gross
        )

    final_volatility = float(
        np.sqrt(
            max(
                constrained_weights
                @ annualized_covariance
                @ constrained_weights,
                0.0,
            )
        )
    )

    return (
        constrained_weights,
        final_volatility,
    )


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

    log_returns = (
        log_returns_pct / 100.0
    )

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

    daily_returns = []
    daily_turnover = []
    daily_costs = []

    weight_rows = []

    previous_quarter = None

    for position in range(
        LOOKBACK,
        len(log_returns),
    ):
        date = log_returns.index[
            position
        ]

        quarter = (
            date.year,
            (date.month - 1) // 3 + 1,
        )

        rebalance = (
            previous_quarter is None
            or quarter != previous_quarter
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
                build_tsmom_weights(
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

            (
                target_weights,
                predicted_volatility,
            ) = scale_and_constrain_weights(
                base_weights,
                covariance,
                assets,
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
                        "signal": signal[asset],
                        "weight": weight,
                        "predicted_volatility": (
                            predicted_volatility
                        ),
                        "net_exposure": float(
                            current_weights.sum()
                        ),
                        "gross_exposure": float(
                            np.abs(
                                current_weights
                            ).sum()
                        ),
                    }
                )

        asset_returns = (
            simple_returns.iloc[position]
            .to_numpy(dtype=float)
        )

        gross_return = float(
            current_weights
            @ asset_returns
        )

        daily_returns.append(
            gross_return - cost
        )

        daily_turnover.append(
            turnover
        )

        daily_costs.append(
            cost
        )

        previous_quarter = quarter

    portfolio_returns = pd.Series(
        daily_returns,
        index=dates,
        name="Constrained TSMOM",
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
        / "tsmom_constrained_daily_returns.csv"
    )

    nav.to_csv(
        outputs
        / "tsmom_constrained_nav.csv"
    )

    weight_history.to_csv(
        outputs
        / "tsmom_constrained_weights.csv",
        index=False,
    )

    pd.DataFrame(
        [performance],
        index=["Constrained TSMOM"],
    ).to_csv(
        outputs
        / "tsmom_constrained_summary.csv"
    )

    latest_date = (
        weight_history["date"].max()
    )

    latest = (
        weight_history[
            weight_history["date"]
            == latest_date
        ]
        .copy()
        .sort_values(
            "weight",
            ascending=False,
        )
    )

    latest["position"] = np.where(
        latest["weight"] > 0,
        "LONG",
        np.where(
            latest["weight"] < 0,
            "SHORT",
            "FLAT",
        ),
    )

    print(
        f"Backtest period: "
        f"{dates.min().date()} "
        f"to {dates.max().date()}"
    )

    print(
        "\nConstrained TSMOM performance:"
    )

    print(
        pd.Series(performance)
        .round(4)
        .to_string()
    )

    print(
        "\nLatest signals and positions:"
    )

    display = latest[
        [
            "asset",
            "position",
            "signal",
            "weight",
        ]
    ].copy()

    display["weight"] *= 100.0

    print(
        display
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

    print(
        "Latest predicted volatility:"
    )

    print(
        f"{100.0 * latest['predicted_volatility'].iloc[0]:.2f}%"
    )


if __name__ == "__main__":
    main()