from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from src.dcc import (
    covariance_to_correlation,
    estimate_dcc,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

RETURN_FILE = (
    DATA_DIR / "long_short_returns_pct.csv"
)

WEIGHT_FILE = (
    OUTPUT_DIR / "cdcc_signal_weights.csv"
)

TRADING_DAYS = 252
FORECAST_HORIZON = 5
MAX_TRAINING = 756

warnings.filterwarnings("ignore")


def regularize_covariance(
    covariance: np.ndarray,
    floor: float = 1e-12,
) -> np.ndarray:
    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    covariance = (
        covariance + covariance.T
    ) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    eigenvalues = np.maximum(
        eigenvalues,
        floor,
    )

    covariance = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    return (
        covariance + covariance.T
    ) / 2.0


def fit_garch_forecasts(
    returns_pct: pd.DataFrame,
    horizon: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
]:
    assets = list(returns_pct.columns)

    standardized_residuals = pd.DataFrame(
        index=returns_pct.index,
        columns=assets,
        dtype=float,
    )

    # 行：予測ホライズン
    # 列：資産
    # 単位：パーセントリターンの二乗
    variance_forecasts_pct2 = np.zeros(
        (
            horizon,
            len(assets),
        ),
        dtype=float,
    )

    parameter_rows = []

    for asset_number, asset in enumerate(
        assets
    ):
        print(
            f"Fitting GARCH: {asset}"
        )

        model = arch_model(
            returns_pct[asset],
            mean="Constant",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="StudentsT",
            rescale=False,
        )

        result = model.fit(
            disp="off",
            show_warning=False,
            options={
                "maxiter": 2000,
            },
        )

        if result.convergence_flag != 0:
            raise RuntimeError(
                "GARCH estimation failed for "
                f"{asset}. Convergence flag: "
                f"{result.convergence_flag}"
            )

        standardized_residuals[asset] = (
            result.std_resid
        )

        forecast = result.forecast(
            horizon=horizon,
            reindex=False,
        )

        variance_path = (
            forecast.variance
            .iloc[-1]
            .to_numpy(dtype=float)
        )

        if (
            len(variance_path) != horizon
            or not np.all(
                np.isfinite(variance_path)
            )
            or np.any(
                variance_path <= 0
            )
        ):
            raise RuntimeError(
                "Invalid GARCH variance forecast "
                f"for {asset}: {variance_path}"
            )

        variance_forecasts_pct2[
            :,
            asset_number,
        ] = variance_path

        parameters = result.params

        alpha = float(
            parameters.get(
                "alpha[1]",
                np.nan,
            )
        )

        beta = float(
            parameters.get(
                "beta[1]",
                np.nan,
            )
        )

        parameter_rows.append(
            {
                "asset": asset,
                "mu": float(
                    parameters.get(
                        "mu",
                        np.nan,
                    )
                ),
                "omega": float(
                    parameters.get(
                        "omega",
                        np.nan,
                    )
                ),
                "alpha": alpha,
                "beta": beta,
                "persistence": (
                    alpha + beta
                ),
                "nu": float(
                    parameters.get(
                        "nu",
                        np.nan,
                    )
                ),
                "loglikelihood": float(
                    result.loglikelihood
                ),
                "aic": float(
                    result.aic
                ),
                "bic": float(
                    result.bic
                ),
                "converged": True,
            }
        )

    standardized_residuals = (
        standardized_residuals
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    # GARCHはパーセント単位のリターンで推定。
    # 共分散計算用にdecimal return単位へ変換。
    variance_forecasts_decimal = (
        variance_forecasts_pct2
        / 10000.0
    )

    parameter_summary = (
        pd.DataFrame(
            parameter_rows
        )
        .set_index("asset")
    )

    return (
        standardized_residuals,
        variance_forecasts_decimal,
        parameter_summary,
    )


def forecast_cdcc_path(
    cdcc_result: dict,
    latest_standardized_residual: np.ndarray,
    horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    a = float(
        cdcc_result["a"]
    )

    b = float(
        cdcc_result["b"]
    )

    target = np.asarray(
        cdcc_result["target"],
        dtype=float,
    )

    latest_q = np.asarray(
        cdcc_result["q_history"][-1],
        dtype=float,
    )

    latest_standardized_residual = (
        np.asarray(
            latest_standardized_residual,
            dtype=float,
        )
    )

    scaling = np.sqrt(
        np.maximum(
            np.diag(latest_q),
            1e-12,
        )
    )

    corrected_innovation = (
        scaling
        * latest_standardized_residual
    )

    # 翌営業日のQ予測
    q_forecast = (
        (1.0 - a - b) * target
        + a
        * np.outer(
            corrected_innovation,
            corrected_innovation,
        )
        + b * latest_q
    )

    q_forecast = (
        q_forecast
        + q_forecast.T
    ) / 2.0

    q_path = []
    correlation_path = []

    for step in range(horizon):
        if step > 0:
            # 将来の標準化ショックの外積について、
            # 条件付き期待値を長期ターゲットと置く。
            q_forecast = (
                (1.0 - b) * target
                + b * q_forecast
            )

            q_forecast = (
                q_forecast
                + q_forecast.T
            ) / 2.0

        correlation = (
            covariance_to_correlation(
                q_forecast
            )
        )

        q_path.append(
            q_forecast.copy()
        )

        correlation_path.append(
            correlation.copy()
        )

    return (
        np.stack(q_path),
        np.stack(correlation_path),
    )


def build_covariance_path(
    variance_path: np.ndarray,
    correlation_path: np.ndarray,
) -> np.ndarray:
    horizon, number_of_assets = (
        variance_path.shape
    )

    covariance_path = np.zeros(
        (
            horizon,
            number_of_assets,
            number_of_assets,
        ),
        dtype=float,
    )

    for step in range(horizon):
        volatility = np.sqrt(
            np.maximum(
                variance_path[step],
                0.0,
            )
        )

        volatility_matrix = np.diag(
            volatility
        )

        covariance = (
            volatility_matrix
            @ correlation_path[step]
            @ volatility_matrix
        )

        covariance_path[step] = (
            regularize_covariance(
                covariance
            )
        )

    return covariance_path


def calculate_realized_volatility(
    returns_pct: pd.DataFrame,
    window: int,
) -> pd.Series:
    return (
        returns_pct
        .tail(window)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )


def drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    strategy: str,
) -> np.ndarray:
    portfolio_return = float(
        weights @ asset_returns
    )

    denominator = (
        1.0 + portfolio_return
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 0
    ):
        raise RuntimeError(
            "Invalid portfolio value while "
            f"drifting {strategy}."
        )

    return (
        weights
        * (1.0 + asset_returns)
        / denominator
    )


def load_latest_weights(
    assets: list[str],
    returns_pct: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray],
    pd.Timestamp | None,
]:
    if not WEIGHT_FILE.exists():
        return {}, None

    weights = pd.read_csv(
        WEIGHT_FILE,
        parse_dates=["date"],
    )

    latest_weight_date = (
        weights["date"].max()
    )

    weights = weights[
        weights["date"]
        == latest_weight_date
    ]

    subsequent_returns_pct = (
        returns_pct.loc[
            returns_pct.index
            > latest_weight_date
        ]
    )

    subsequent_simple_returns = (
        np.expm1(
            subsequent_returns_pct
            / 100.0
        )
    )

    strategy_weights = {}

    for strategy, group in weights.groupby(
        "strategy"
    ):
        vector = (
            group
            .set_index("asset")["weight"]
            .reindex(assets)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        for _, daily_returns in (
            subsequent_simple_returns.iterrows()
        ):
            asset_returns = (
                daily_returns
                .reindex(assets)
                .to_numpy(dtype=float)
            )

            vector = drift_weights(
                weights=vector,
                asset_returns=asset_returns,
                strategy=strategy,
            )

        strategy_weights[
            strategy
        ] = vector

    return (
        strategy_weights,
        latest_weight_date,
    )


def portfolio_period_volatility(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    variance = float(
        weights
        @ covariance
        @ weights
    )

    return np.sqrt(
        max(
            variance,
            0.0,
        )
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
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    training_returns = (
        returns_pct.tail(
            MAX_TRAINING
        )
    )

    assets = list(
        training_returns.columns
    )

    forecast_origin = (
        training_returns.index[-1]
    )

    print("Forecast origin:")
    print(
        forecast_origin.date()
    )

    print()
    print("Training sample:")
    print(
        training_returns.index[0].date(),
        "to",
        training_returns.index[-1].date(),
        f"({len(training_returns)} observations)",
    )

    print()
    print(
        "Estimating five-step "
        "DCC-GARCH forecast..."
    )
    print()

    (
        standardized_residuals,
        variance_path,
        garch_parameters,
    ) = fit_garch_forecasts(
        returns_pct=training_returns,
        horizon=FORECAST_HORIZON,
    )

    cdcc_result = estimate_dcc(
        standardized_residuals.to_numpy(),
        corrected=True,
    )

    if not cdcc_result[
        "converged"
    ]:
        raise RuntimeError(
            "cDCC estimation did not converge."
        )

    (
        q_path,
        correlation_path,
    ) = forecast_cdcc_path(
        cdcc_result=cdcc_result,
        latest_standardized_residual=(
            standardized_residuals
            .iloc[-1]
            .to_numpy()
        ),
        horizon=FORECAST_HORIZON,
    )

    covariance_path = (
        build_covariance_path(
            variance_path=variance_path,
            correlation_path=(
                correlation_path
            ),
        )
    )

    # 日次リターンに自己相関がないという
    # 通常の仮定のもとで5日分を合計する。
    weekly_covariance = (
        covariance_path.sum(axis=0)
    )

    weekly_covariance = (
        regularize_covariance(
            weekly_covariance
        )
    )

    next_day_covariance = (
        covariance_path[0]
    )

    next_day_correlation = (
        correlation_path[0]
    )

    weekly_effective_correlation = (
        covariance_to_correlation(
            weekly_covariance
        )
    )

    next_day_volatility = np.sqrt(
        np.maximum(
            np.diag(
                next_day_covariance
            ),
            0.0,
        )
    )

    weekly_volatility = np.sqrt(
        np.maximum(
            np.diag(
                weekly_covariance
            ),
            0.0,
        )
    )

    next_day_annualized_volatility = (
        next_day_volatility
        * np.sqrt(TRADING_DAYS)
    )

    weekly_annualized_volatility = (
        weekly_volatility
        * np.sqrt(
            TRADING_DAYS
            / FORECAST_HORIZON
        )
    )

    realized_63 = (
        calculate_realized_volatility(
            training_returns,
            window=63,
        )
    )

    realized_252 = (
        calculate_realized_volatility(
            training_returns,
            window=252,
        )
    )

    asset_forecast = pd.DataFrame(
        {
            "asset": assets,
            "forecast_origin": (
                forecast_origin
            ),
            "next_day_volatility_pct": (
                next_day_volatility
                * 100.0
            ),
            "next_week_volatility_pct": (
                weekly_volatility
                * 100.0
            ),
            "next_day_annualized_volatility_pct": (
                next_day_annualized_volatility
                * 100.0
            ),
            "next_week_annualized_volatility_pct": (
                weekly_annualized_volatility
                * 100.0
            ),
            "realized_63d_annualized_volatility_pct": (
                realized_63
                .reindex(assets)
                .to_numpy()
            ),
            "realized_252d_annualized_volatility_pct": (
                realized_252
                .reindex(assets)
                .to_numpy()
            ),
        }
    )

    asset_forecast[
        "weekly_forecast_vs_63d_ratio"
    ] = (
        asset_forecast[
            "next_week_annualized_volatility_pct"
        ]
        / asset_forecast[
            "realized_63d_annualized_volatility_pct"
        ]
    )

    asset_forecast[
        "weekly_forecast_vs_252d_ratio"
    ] = (
        asset_forecast[
            "next_week_annualized_volatility_pct"
        ]
        / asset_forecast[
            "realized_252d_annualized_volatility_pct"
        ]
    )

    horizon_rows = []

    for step in range(
        FORECAST_HORIZON
    ):
        daily_volatility = np.sqrt(
            np.maximum(
                np.diag(
                    covariance_path[step]
                ),
                0.0,
            )
        )

        for asset_number, asset in enumerate(
            assets
        ):
            horizon_rows.append(
                {
                    "forecast_origin": (
                        forecast_origin
                    ),
                    "horizon_day": (
                        step + 1
                    ),
                    "asset": asset,
                    "daily_volatility_pct": (
                        daily_volatility[
                            asset_number
                        ]
                        * 100.0
                    ),
                    "annualized_volatility_pct": (
                        daily_volatility[
                            asset_number
                        ]
                        * np.sqrt(
                            TRADING_DAYS
                        )
                        * 100.0
                    ),
                }
            )

    horizon_forecast = (
        pd.DataFrame(
            horizon_rows
        )
    )

    next_day_covariance_frame = (
        pd.DataFrame(
            next_day_covariance,
            index=assets,
            columns=assets,
        )
    )

    weekly_covariance_frame = (
        pd.DataFrame(
            weekly_covariance,
            index=assets,
            columns=assets,
        )
    )

    next_day_correlation_frame = (
        pd.DataFrame(
            next_day_correlation,
            index=assets,
            columns=assets,
        )
    )

    weekly_correlation_frame = (
        pd.DataFrame(
            weekly_effective_correlation,
            index=assets,
            columns=assets,
        )
    )

    asset_forecast.to_csv(
        OUTPUT_DIR
        / "dcc_garch_volatility_forecast.csv",
        index=False,
    )

    horizon_forecast.to_csv(
        OUTPUT_DIR
        / "dcc_garch_horizon_volatility_forecast.csv",
        index=False,
    )

    next_day_covariance_frame.to_csv(
        OUTPUT_DIR
        / "dcc_garch_next_day_covariance_forecast.csv"
    )

    weekly_covariance_frame.to_csv(
        OUTPUT_DIR
        / "dcc_garch_weekly_covariance_forecast.csv"
    )

    next_day_correlation_frame.to_csv(
        OUTPUT_DIR
        / "dcc_garch_next_day_correlation_forecast.csv"
    )

    weekly_correlation_frame.to_csv(
        OUTPUT_DIR
        / "dcc_garch_weekly_correlation_forecast.csv"
    )

    # 従来ファイル名との互換性を維持。
    weekly_covariance_frame.to_csv(
        OUTPUT_DIR
        / "dcc_garch_covariance_forecast.csv"
    )

    weekly_correlation_frame.to_csv(
        OUTPUT_DIR
        / "dcc_garch_correlation_forecast.csv"
    )

    garch_parameters.to_csv(
        OUTPUT_DIR
        / "dcc_garch_univariate_parameters.csv"
    )

    (
        strategy_weights,
        latest_weight_date,
    ) = load_latest_weights(
        assets=assets,
        returns_pct=training_returns,
    )

    portfolio_rows = []

    for strategy, weights in (
        strategy_weights.items()
    ):
        next_day_portfolio_volatility = (
            portfolio_period_volatility(
                weights=weights,
                covariance=(
                    next_day_covariance
                ),
            )
        )

        weekly_portfolio_volatility = (
            portfolio_period_volatility(
                weights=weights,
                covariance=(
                    weekly_covariance
                ),
            )
        )

        portfolio_rows.append(
            {
                "strategy": strategy,
                "forecast_origin": (
                    forecast_origin
                ),
                "target_weight_date": (
                    latest_weight_date
                ),
                "weights_drifted_to": (
                    forecast_origin
                ),
                "next_day_volatility_pct": (
                    next_day_portfolio_volatility
                    * 100.0
                ),
                "next_week_volatility_pct": (
                    weekly_portfolio_volatility
                    * 100.0
                ),
                "next_day_annualized_volatility_pct": (
                    next_day_portfolio_volatility
                    * np.sqrt(
                        TRADING_DAYS
                    )
                    * 100.0
                ),
                "next_week_annualized_volatility_pct": (
                    weekly_portfolio_volatility
                    * np.sqrt(
                        TRADING_DAYS
                        / FORECAST_HORIZON
                    )
                    * 100.0
                ),
                "gross_exposure_pct": (
                    np.abs(weights).sum()
                    * 100.0
                ),
                "net_exposure_pct": (
                    weights.sum()
                    * 100.0
                ),
            }
        )

    if portfolio_rows:
        portfolio_forecast = (
            pd.DataFrame(
                portfolio_rows
            )
            .set_index("strategy")
        )

        portfolio_forecast.to_csv(
            OUTPUT_DIR
            / "dcc_garch_portfolio_volatility_forecast.csv"
        )

        print()
        print(
            "Portfolio weekly volatility forecast:"
        )

        print(
            portfolio_forecast[
                [
                    "next_week_volatility_pct",
                    "next_week_annualized_volatility_pct",
                    "gross_exposure_pct",
                    "net_exposure_pct",
                ]
            ]
            .round(3)
            .to_string()
        )

        print()
        print(
            "Target weights dated:",
            latest_weight_date.date(),
        )

        print(
            "Weights drifted to:",
            forecast_origin.date(),
        )

    print()
    print("cDCC parameters:")
    print(
        f"a={cdcc_result['a']:.6f}, "
        f"b={cdcc_result['b']:.6f}, "
        "persistence="
        f"{cdcc_result['a'] + cdcc_result['b']:.6f}"
    )

    print()
    print(
        "Asset weekly volatility forecast:"
    )

    print(
        asset_forecast[
            [
                "asset",
                "next_week_volatility_pct",
                "next_week_annualized_volatility_pct",
                "realized_63d_annualized_volatility_pct",
                "realized_252d_annualized_volatility_pct",
                "weekly_forecast_vs_63d_ratio",
            ]
        ]
        .sort_values(
            "next_week_annualized_volatility_pct",
            ascending=False,
        )
        .round(3)
        .to_string(index=False)
    )

    print()
    print("Outputs saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()