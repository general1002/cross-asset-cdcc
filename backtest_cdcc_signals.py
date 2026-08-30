from pathlib import Path

import numpy as np
import pandas as pd

from backtest_cdcc_portfolios import (
    MAX_TRAINING,
    MIN_TRAINING,
    OUTPUT_DIR,
    TRADING_DAYS,
    estimate_cdcc_covariance,
    estimate_shrinkage_covariance,
    performance_summary,
    regularize_covariance,
)


ROOT = Path(__file__).resolve().parent
RETURN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "long_short_returns_pct.csv"
)

LOOKBACKS = (63, 126, 252)

TARGET_VOLATILITY = 0.10
MAX_GROSS_EXPOSURE = 1.50
MAX_NET_EXPOSURE = 1.00
MAX_ASSET_WEIGHT = 0.15
MAX_FX_WEIGHT = 0.10

TRANSACTION_COST_BPS = 5.0
RIDGE_STRENGTH = 0.10

FX_ASSETS = {
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
}


def calculate_tsmom_signal(
    historical_returns_pct: pd.DataFrame,
) -> pd.Series:
    signals = []

    for lookback in LOOKBACKS:
        cumulative_log_return = (
            historical_returns_pct
            .tail(lookback)
            .sum()
        )

        signals.append(
            np.sign(cumulative_log_return)
        )

    combined = pd.concat(
        signals,
        axis=1,
    ).mean(axis=1)

    return combined.astype(float)


def inverse_volatility_signal_weights(
    signal: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    volatility = np.sqrt(
        np.maximum(
            np.diag(covariance),
            1e-12,
        )
    )

    return signal / volatility


def precision_signal_weights(
    signal: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = regularize_covariance(
        covariance
    )

    diagonal = np.diag(
        np.diag(covariance)
    )

    regularized_covariance = (
        covariance
        + RIDGE_STRENGTH * diagonal
    )

    volatility = np.sqrt(
        np.maximum(
            np.diag(covariance),
            1e-12,
        )
    )

    # signal × volatility とすることで、
    # 対角共分散なら inverse-volatility 配分になる。
    expected_return_score = (
        signal * volatility
    )

    raw_weights = np.linalg.solve(
        regularized_covariance,
        expected_return_score,
    )

    # 共分散ヘッジによってシグナルと逆方向の
    # ポジションが生成されるのを防ぐ。
    aligned_exposure = (
        np.sign(signal) * raw_weights
    )

    aligned_exposure = np.maximum(
        aligned_exposure,
        0.0,
    )

    aligned_weights = (
        np.sign(signal)
        * aligned_exposure
    )

    if np.abs(aligned_weights).sum() < 1e-12:
        return inverse_volatility_signal_weights(
            signal=signal,
            covariance=covariance,
        )

    return aligned_weights


def apply_constraints(
    raw_weights: np.ndarray,
    covariance: np.ndarray,
    assets: list[str],
) -> np.ndarray:
    weights = np.asarray(
        raw_weights,
        dtype=float,
    ).copy()

    gross = np.abs(weights).sum()

    if (
        not np.isfinite(gross)
        or gross <= 1e-12
    ):
        return np.zeros_like(weights)

    # 最初にグロス100%へ正規化。
    weights /= gross

    predicted_volatility = np.sqrt(
        max(
            float(
                weights
                @ covariance
                @ weights
            ),
            0.0,
        )
        * TRADING_DAYS
    )

    if predicted_volatility > 1e-12:
        weights *= (
            TARGET_VOLATILITY
            / predicted_volatility
        )

    # 資産別上限
    for index, asset in enumerate(assets):
        limit = (
            MAX_FX_WEIGHT
            if asset in FX_ASSETS
            else MAX_ASSET_WEIGHT
        )

        weights[index] = np.clip(
            weights[index],
            -limit,
            limit,
        )

    gross = np.abs(weights).sum()

    if gross > MAX_GROSS_EXPOSURE:
        weights *= (
            MAX_GROSS_EXPOSURE / gross
        )

    net = weights.sum()

    if abs(net) > MAX_NET_EXPOSURE:
        weights *= (
            MAX_NET_EXPOSURE / abs(net)
        )

    return weights


def equal_weight_benchmark(
    assets: list[str],
) -> np.ndarray:
    weights = np.zeros(
        len(assets),
        dtype=float,
    )

    funded_indices = [
        index
        for index, asset in enumerate(assets)
        if asset not in FX_ASSETS
    ]

    weights[funded_indices] = (
        1.0 / len(funded_indices)
    )

    return weights


def calculate_target_weights(
    signal: np.ndarray,
    shrinkage_covariance: np.ndarray,
    cdcc_covariance: np.ndarray,
    assets: list[str],
) -> dict[str, np.ndarray]:
    inverse_volatility_raw = (
        inverse_volatility_signal_weights(
            signal=signal,
            covariance=shrinkage_covariance,
        )
    )

    shrinkage_raw = precision_signal_weights(
        signal=signal,
        covariance=shrinkage_covariance,
    )

    cdcc_raw = precision_signal_weights(
        signal=signal,
        covariance=cdcc_covariance,
    )

    return {
        "Equal Weight": (
            equal_weight_benchmark(assets)
        ),
        "TSMOM Inverse Volatility": (
            apply_constraints(
                raw_weights=inverse_volatility_raw,
                covariance=shrinkage_covariance,
                assets=assets,
            )
        ),
        "TSMOM Shrinkage Precision": (
            apply_constraints(
                raw_weights=shrinkage_raw,
                covariance=shrinkage_covariance,
                assets=assets,
            )
        ),
        "TSMOM cDCC Precision": (
            apply_constraints(
                raw_weights=cdcc_raw,
                covariance=cdcc_covariance,
                assets=assets,
            )
        ),
    }


def calculate_turnover(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    return 0.5 * float(
        np.abs(
            target_weights
            - current_weights
        ).sum()
    )


def drift_long_short_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    portfolio_return = float(
        weights @ asset_returns
    )

    denominator = 1.0 + portfolio_return

    if (
        not np.isfinite(denominator)
        or denominator <= 0
    ):
        raise RuntimeError(
            "Portfolio value became non-positive."
        )

    return (
        weights
        * (1.0 + asset_returns)
        / denominator
    )


def is_new_quarter(
    dates: pd.DatetimeIndex,
    position: int,
    previous_rebalance: pd.Timestamp | None,
) -> bool:
    if previous_rebalance is None:
        return True

    return (
        dates[position].to_period("Q")
        != previous_rebalance.to_period("Q")
    )


def predicted_annualized_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    variance = float(
        weights @ covariance @ weights
    )

    return np.sqrt(
        max(variance, 0.0)
        * TRADING_DAYS
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    returns_pct = pd.read_csv(
        RETURN_FILE,
        index_col=0,
        parse_dates=True,
    )

    returns_pct = (
        returns_pct
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    simple_returns = np.expm1(
        returns_pct / 100.0
    )

    dates = returns_pct.index
    assets = list(returns_pct.columns)

    strategies = [
        "Equal Weight",
        "TSMOM Inverse Volatility",
        "TSMOM Shrinkage Precision",
        "TSMOM cDCC Precision",
    ]

    backtest_dates = dates[MIN_TRAINING:]

    strategy_returns = pd.DataFrame(
        index=backtest_dates,
        columns=strategies,
        dtype=float,
    )

    turnover_history = pd.DataFrame(
        0.0,
        index=backtest_dates,
        columns=strategies,
    )

    cost_history = pd.DataFrame(
        0.0,
        index=backtest_dates,
        columns=strategies,
    )

    current_weights = {
        strategy: np.zeros(
            len(assets),
            dtype=float,
        )
        for strategy in strategies
    }

    weight_records = []
    diagnostic_records = []

    previous_rebalance = None

    cost_rate = (
        TRANSACTION_COST_BPS / 10000.0
    )

    for position in range(
        MIN_TRAINING,
        len(dates),
    ):
        date = dates[position]

        if is_new_quarter(
            dates=dates,
            position=position,
            previous_rebalance=previous_rebalance,
        ):
            training_start = max(
                0,
                position - MAX_TRAINING,
            )

            training_returns_pct = (
                returns_pct.iloc[
                    training_start:position
                ]
            )

            training_simple_returns = (
                simple_returns.iloc[
                    training_start:position
                ]
            )

            signal = calculate_tsmom_signal(
                returns_pct.iloc[:position]
            )

            (
                shrinkage_covariance,
                shrinkage,
            ) = estimate_shrinkage_covariance(
                training_simple_returns
            )

            (
                cdcc_covariance,
                cdcc_result,
            ) = estimate_cdcc_covariance(
                training_returns_pct
            )

            target_weights = calculate_target_weights(
                signal=signal.to_numpy(),
                shrinkage_covariance=(
                    shrinkage_covariance
                ),
                cdcc_covariance=cdcc_covariance,
                assets=assets,
            )

            print()
            print("Rebalancing:", date.date())
            print(
                "Training:",
                training_returns_pct.index[0].date(),
                "to",
                training_returns_pct.index[-1].date(),
            )
            print(
                "cDCC:",
                f"a={cdcc_result['a']:.6f},",
                f"b={cdcc_result['b']:.6f},",
                "persistence="
                f"{cdcc_result['a'] + cdcc_result['b']:.6f}",
            )

            for strategy in strategies:
                target = target_weights[strategy]

                turnover = calculate_turnover(
                    current_weights[strategy],
                    target,
                )

                cost = turnover * cost_rate

                turnover_history.loc[
                    date,
                    strategy,
                ] = turnover

                cost_history.loc[
                    date,
                    strategy,
                ] = cost

                current_weights[strategy] = (
                    target.copy()
                )

                covariance_for_diagnostic = (
                    cdcc_covariance
                    if strategy
                    == "TSMOM cDCC Precision"
                    else shrinkage_covariance
                )

                diagnostic_records.append(
                    {
                        "date": date,
                        "strategy": strategy,
                        "gross_exposure": (
                            np.abs(target).sum()
                        ),
                        "net_exposure": (
                            target.sum()
                        ),
                        "predicted_volatility": (
                            predicted_annualized_volatility(
                                target,
                                covariance_for_diagnostic,
                            )
                        ),
                        "turnover": turnover,
                        "cdcc_a": cdcc_result["a"],
                        "cdcc_b": cdcc_result["b"],
                        "cdcc_persistence": (
                            cdcc_result["a"]
                            + cdcc_result["b"]
                        ),
                        "shrinkage": shrinkage,
                    }
                )

                for asset_number, asset in enumerate(
                    assets
                ):
                    weight_records.append(
                        {
                            "date": date,
                            "strategy": strategy,
                            "asset": asset,
                            "signal": signal.loc[asset],
                            "weight": target[
                                asset_number
                            ],
                        }
                    )

            previous_rebalance = date

        asset_returns = (
            simple_returns.iloc[position]
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

            strategy_returns.loc[
                date,
                strategy,
            ] = net_return

            current_weights[strategy] = (
                drift_long_short_weights(
                    weights=weights,
                    asset_returns=asset_returns,
                )
            )

    nav = (
        1.0 + strategy_returns
    ).cumprod()

    performance = performance_summary(
        portfolio_returns=strategy_returns,
        turnover=turnover_history,
        transaction_costs=cost_history,
    )

    weights_frame = pd.DataFrame(
        weight_records
    )

    diagnostics = pd.DataFrame(
        diagnostic_records
    )

    strategy_returns.to_csv(
        OUTPUT_DIR
        / "cdcc_signal_returns.csv"
    )

    nav.to_csv(
        OUTPUT_DIR
        / "cdcc_signal_nav.csv"
    )

    performance.to_csv(
        OUTPUT_DIR
        / "cdcc_signal_performance.csv"
    )

    weights_frame.to_csv(
        OUTPUT_DIR
        / "cdcc_signal_weights.csv",
        index=False,
    )

    diagnostics.to_csv(
        OUTPUT_DIR
        / "cdcc_signal_diagnostics.csv",
        index=False,
    )

    print()
    print("Backtest period:")
    print(
        strategy_returns.index[0].date(),
        "to",
        strategy_returns.index[-1].date(),
    )

    print()
    print("Performance after transaction costs:")
    print(
        performance.round(4).to_string()
    )

    latest_date = weights_frame[
        "date"
    ].max()

    latest = (
        weights_frame[
            weights_frame["date"]
            == latest_date
        ]
        .copy()
    )

    latest["position"] = np.where(
        latest["weight"] > 1e-6,
        "LONG",
        np.where(
            latest["weight"] < -1e-6,
            "SHORT",
            "FLAT",
        ),
    )

    latest["weight_pct"] = (
        latest["weight"] * 100.0
    )

    print()
    print("Latest cDCC signals and positions:")

    latest_cdcc = (
        latest[
            latest["strategy"]
            == "TSMOM cDCC Precision"
        ][
            [
                "asset",
                "position",
                "signal",
                "weight_pct",
            ]
        ]
        .sort_values(
            "weight_pct",
            ascending=False,
        )
    )

    print(
        latest_cdcc.round(3).to_string(
            index=False
        )
    )

    latest_diagnostic = (
        diagnostics[
            (
                diagnostics["date"]
                == latest_date
            )
            & (
                diagnostics["strategy"]
                == "TSMOM cDCC Precision"
            )
        ]
        .iloc[-1]
    )

    print()
    print(
        "Latest cDCC gross exposure:",
        f"{latest_diagnostic['gross_exposure'] * 100:.2f}%",
    )
    print(
        "Latest cDCC net exposure:",
        f"{latest_diagnostic['net_exposure'] * 100:.2f}%",
    )
    print(
        "Latest predicted volatility:",
        f"{latest_diagnostic['predicted_volatility'] * 100:.2f}%",
    )

    print()
    print("Outputs saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()