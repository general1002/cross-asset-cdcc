from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.dcc import (
    estimate_dcc,
    forecast_next_correlation,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

RETURN_FILE = DATA_DIR / "long_only_returns_pct.csv"

TRADING_DAYS = 252
MIN_TRAINING = 504
MAX_TRAINING = 756
MAX_WEIGHT = 0.20
TRANSACTION_COST_BPS = 5.0

warnings.filterwarnings("ignore")


def regularize_covariance(
    covariance: np.ndarray,
    floor: float = 1e-10,
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


def equal_weight(
    number_of_assets: int,
) -> np.ndarray:
    return np.full(
        number_of_assets,
        1.0 / number_of_assets,
    )


def minimum_variance_weights(
    covariance: np.ndarray,
    maximum_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    covariance = regularize_covariance(
        covariance
    )

    number_of_assets = covariance.shape[0]
    initial = equal_weight(number_of_assets)

    def objective(
        weights: np.ndarray,
    ) -> float:
        return float(
            weights @ covariance @ weights
        )

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[
            (0.0, maximum_weight)
            for _ in range(number_of_assets)
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: (
                    np.sum(weights) - 1.0
                ),
            }
        ],
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-variance optimization failed: "
            f"{result.message}"
        )

    weights = np.maximum(
        result.x,
        0.0,
    )

    return weights / weights.sum()


def risk_parity_weights(
    covariance: np.ndarray,
    maximum_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    covariance = regularize_covariance(
        covariance
    )

    number_of_assets = covariance.shape[0]
    target_contribution = np.full(
        number_of_assets,
        1.0 / number_of_assets,
    )

    inverse_volatility = 1.0 / np.sqrt(
        np.diag(covariance)
    )

    initial = (
        inverse_volatility
        / inverse_volatility.sum()
    )

    initial = np.minimum(
        initial,
        maximum_weight,
    )
    initial = initial / initial.sum()

    def objective(
        weights: np.ndarray,
    ) -> float:
        portfolio_variance = float(
            weights @ covariance @ weights
        )

        if portfolio_variance <= 0:
            return 1e10

        marginal_contribution = (
            covariance @ weights
        )

        risk_contribution = (
            weights * marginal_contribution
        )

        normalized_contribution = (
            risk_contribution
            / portfolio_variance
        )

        return float(
            np.sum(
                (
                    normalized_contribution
                    - target_contribution
                )
                ** 2
            )
        )

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[
            (1e-8, maximum_weight)
            for _ in range(number_of_assets)
        ],
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: (
                    np.sum(weights) - 1.0
                ),
            }
        ],
        options={
            "maxiter": 3000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Risk-parity optimization failed: "
            f"{result.message}"
        )

    weights = np.maximum(
        result.x,
        0.0,
    )

    return weights / weights.sum()


def fit_univariate_garch(
    returns_pct: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    standardized_residuals = pd.DataFrame(
        index=returns_pct.index,
        columns=returns_pct.columns,
        dtype=float,
    )

    forecast_volatility_pct = np.zeros(
        returns_pct.shape[1],
        dtype=float,
    )

    for column_number, asset in enumerate(
        returns_pct.columns
    ):
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
                "maxiter": 1000,
            },
        )

        standardized_residuals[asset] = (
            result.std_resid
        )

        forecast = result.forecast(
            horizon=1,
            reindex=False,
        )

        forecast_variance = float(
            forecast.variance.iloc[-1, 0]
        )

        if (
            not np.isfinite(forecast_variance)
            or forecast_variance <= 0
        ):
            raise RuntimeError(
                "Invalid volatility forecast for "
                f"{asset}: {forecast_variance}"
            )

        forecast_volatility_pct[
            column_number
        ] = np.sqrt(forecast_variance)

    standardized_residuals = (
        standardized_residuals
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return (
        standardized_residuals,
        forecast_volatility_pct,
    )


def estimate_cdcc_covariance(
    training_returns_pct: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    (
        standardized_residuals,
        forecast_volatility_pct,
    ) = fit_univariate_garch(
        training_returns_pct
    )

    cdcc_result = estimate_dcc(
        standardized_residuals.to_numpy(),
        corrected=True,
    )

    if not cdcc_result["converged"]:
        raise RuntimeError(
            "cDCC estimation did not converge."
        )

    forecast_correlation = (
        forecast_next_correlation(
            result=cdcc_result,
            latest_standardized_residual=(
                standardized_residuals
                .iloc[-1]
                .to_numpy()
            ),
            corrected=True,
        )
    )

    volatility_decimal = (
        forecast_volatility_pct / 100.0
    )

    volatility_matrix = np.diag(
        volatility_decimal
    )

    forecast_covariance = (
        volatility_matrix
        @ forecast_correlation
        @ volatility_matrix
    )

    forecast_covariance = (
        regularize_covariance(
            forecast_covariance
        )
    )

    return forecast_covariance, cdcc_result


def estimate_shrinkage_covariance(
    training_simple_returns: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    estimator = LedoitWolf().fit(
        training_simple_returns.to_numpy()
    )

    covariance = regularize_covariance(
        estimator.covariance_
    )

    return (
        covariance,
        float(estimator.shrinkage_),
    )


def calculate_target_weights(
    shrinkage_covariance: np.ndarray,
    cdcc_covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    number_of_assets = (
        shrinkage_covariance.shape[0]
    )

    return {
        "Equal Weight": equal_weight(
            number_of_assets
        ),
        "Shrinkage Risk Parity": (
            risk_parity_weights(
                shrinkage_covariance
            )
        ),
        "cDCC Risk Parity": (
            risk_parity_weights(
                cdcc_covariance
            )
        ),
        "Shrinkage Minimum Variance": (
            minimum_variance_weights(
                shrinkage_covariance
            )
        ),
        "cDCC Minimum Variance": (
            minimum_variance_weights(
                cdcc_covariance
            )
        ),
    }


def should_rebalance(
    dates: pd.DatetimeIndex,
    position: int,
    previous_rebalance: pd.Timestamp | None,
) -> bool:
    if previous_rebalance is None:
        return True

    current_quarter = dates[position].to_period("Q")
    previous_quarter = (
        previous_rebalance.to_period("Q")
    )

    return current_quarter != previous_quarter


def calculate_turnover(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    return 0.5 * float(
        np.abs(
            target_weights - current_weights
        ).sum()
    )


def drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
) -> np.ndarray:
    updated_values = weights * (
        1.0 + asset_returns
    )

    total_value = updated_values.sum()

    if (
        not np.isfinite(total_value)
        or total_value <= 0
    ):
        raise RuntimeError(
            "Invalid portfolio value while "
            "drifting weights."
        )

    return updated_values / total_value


def performance_summary(
    portfolio_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    transaction_costs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for strategy in portfolio_returns.columns:
        returns = (
            portfolio_returns[strategy]
            .dropna()
        )

        number_of_observations = len(returns)

        nav = (1.0 + returns).cumprod()

        annualized_return = (
            nav.iloc[-1]
            ** (
                TRADING_DAYS
                / number_of_observations
            )
            - 1.0
        )

        annualized_volatility = (
            returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )
        annualized_mean_return = (
            returns.mean() * TRADING_DAYS
        )

        sharpe_ratio = (
            annualized_mean_return
            / annualized_volatility
            if annualized_volatility > 0
            else np.nan
        )

        drawdown = (
            nav / nav.cummax() - 1.0
        )

        maximum_drawdown = drawdown.min()

        calmar_ratio = (
            annualized_return
            / abs(maximum_drawdown)
            if maximum_drawdown < 0
            else np.nan
        )

        annualized_turnover = (
            turnover[strategy].sum()
            * TRADING_DAYS
            / number_of_observations
        )

        total_cost = (
            transaction_costs[strategy]
            .sum()
        )

        rows.append(
            {
                "strategy": strategy,
                "annualized_return_pct": (
                    annualized_return * 100.0
                ),
                "annualized_volatility_pct": (
                    annualized_volatility
                    * 100.0
                ),
                "sharpe_ratio": sharpe_ratio,
                "maximum_drawdown_pct": (
                    maximum_drawdown * 100.0
                ),
                "calmar_ratio": calmar_ratio,
                "annualized_turnover_pct": (
                    annualized_turnover
                    * 100.0
                ),
                "total_transaction_cost_pct": (
                    total_cost * 100.0
                ),
                "final_nav": nav.iloc[-1],
                "observations": (
                    number_of_observations
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("strategy")
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

    # 入力はパーセント単位の対数リターン。
    # ポートフォリオ損益計算では単純リターンに変換する。
    simple_returns = np.expm1(
        returns_pct / 100.0
    )

    if len(returns_pct) <= MIN_TRAINING:
        raise ValueError(
            "Insufficient observations. "
            f"Need more than {MIN_TRAINING}."
        )

    dates = returns_pct.index
    assets = list(returns_pct.columns)

    strategy_names = [
        "Equal Weight",
        "Shrinkage Risk Parity",
        "cDCC Risk Parity",
        "Shrinkage Minimum Variance",
        "cDCC Minimum Variance",
    ]

    backtest_dates = dates[MIN_TRAINING:]

    portfolio_returns = pd.DataFrame(
        index=backtest_dates,
        columns=strategy_names,
        dtype=float,
    )

    turnover_history = pd.DataFrame(
        0.0,
        index=backtest_dates,
        columns=strategy_names,
    )

    transaction_cost_history = pd.DataFrame(
        0.0,
        index=backtest_dates,
        columns=strategy_names,
    )

    current_weights = {
        strategy: np.zeros(
            len(assets),
            dtype=float,
        )
        for strategy in strategy_names
    }

    weight_records = []
    parameter_records = []

    previous_rebalance = None
    cost_rate = (
        TRANSACTION_COST_BPS / 10000.0
    )

    for position in range(
        MIN_TRAINING,
        len(dates),
    ):
        date = dates[position]

        rebalance = should_rebalance(
            dates=dates,
            position=position,
            previous_rebalance=previous_rebalance,
        )

        if rebalance:
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

            print()
            print(
                "Rebalancing:",
                date.date(),
            )
            print(
                "Training sample:",
                training_returns_pct.index[0].date(),
                "to",
                training_returns_pct.index[-1].date(),
                f"({len(training_returns_pct)} observations)",
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

            target_weights = (
                calculate_target_weights(
                    shrinkage_covariance=(
                        shrinkage_covariance
                    ),
                    cdcc_covariance=(
                        cdcc_covariance
                    ),
                )
            )

            print(
                "cDCC:",
                f"a={cdcc_result['a']:.6f},",
                f"b={cdcc_result['b']:.6f},",
                "persistence="
                f"{cdcc_result['a'] + cdcc_result['b']:.6f},",
                f"shrinkage={shrinkage:.4f}",
            )

            parameter_records.append(
                {
                    "date": date,
                    "training_start": (
                        training_returns_pct
                        .index[0]
                    ),
                    "training_end": (
                        training_returns_pct
                        .index[-1]
                    ),
                    "observations": len(
                        training_returns_pct
                    ),
                    "cdcc_a": cdcc_result["a"],
                    "cdcc_b": cdcc_result["b"],
                    "cdcc_persistence": (
                        cdcc_result["a"]
                        + cdcc_result["b"]
                    ),
                    "cdcc_converged": (
                        cdcc_result["converged"]
                    ),
                    "ledoit_wolf_shrinkage": (
                        shrinkage
                    ),
                }
            )

            for strategy in strategy_names:
                target = target_weights[
                    strategy
                ]

                turnover = calculate_turnover(
                    current_weights[strategy],
                    target,
                )

                transaction_cost = (
                    turnover * cost_rate
                )

                turnover_history.loc[
                    date,
                    strategy,
                ] = turnover

                transaction_cost_history.loc[
                    date,
                    strategy,
                ] = transaction_cost

                current_weights[strategy] = (
                    target.copy()
                )

                for asset_number, asset in enumerate(
                    assets
                ):
                    weight_records.append(
                        {
                            "date": date,
                            "strategy": strategy,
                            "asset": asset,
                            "weight": target[
                                asset_number
                            ],
                        }
                    )

            previous_rebalance = date

        daily_asset_returns = (
            simple_returns.iloc[position]
            .to_numpy(dtype=float)
        )

        for strategy in strategy_names:
            weights = current_weights[
                strategy
            ]

            gross_return = float(
                weights @ daily_asset_returns
            )

            transaction_cost = (
                transaction_cost_history.loc[
                    date,
                    strategy,
                ]
            )

            net_return = (
                gross_return
                - transaction_cost
            )

            portfolio_returns.loc[
                date,
                strategy,
            ] = net_return

            current_weights[strategy] = (
                drift_weights(
                    weights=weights,
                    asset_returns=(
                        daily_asset_returns
                    ),
                )
            )

    nav = (
        1.0 + portfolio_returns
    ).cumprod()

    summary = performance_summary(
        portfolio_returns=portfolio_returns,
        turnover=turnover_history,
        transaction_costs=(
            transaction_cost_history
        ),
    )

    weights_long = pd.DataFrame(
        weight_records
    )

    parameters = (
        pd.DataFrame(parameter_records)
        .set_index("date")
    )

    portfolio_returns.to_csv(
        OUTPUT_DIR
        / "cdcc_portfolio_returns.csv"
    )

    nav.to_csv(
        OUTPUT_DIR
        / "cdcc_portfolio_nav.csv"
    )

    weights_long.to_csv(
        OUTPUT_DIR
        / "cdcc_portfolio_weights.csv",
        index=False,
    )

    parameters.to_csv(
        OUTPUT_DIR
        / "cdcc_walkforward_parameters.csv"
    )

    summary.to_csv(
        OUTPUT_DIR
        / "cdcc_portfolio_performance.csv"
    )

    print()
    print("Backtest period:")
    print(
        portfolio_returns.index[0].date(),
        "to",
        portfolio_returns.index[-1].date(),
    )

    print()
    print("Performance after transaction costs:")
    print(
        summary.round(4).to_string()
    )

    print()
    print("Latest target weights (%):")

    latest_date = weights_long["date"].max()

    latest_weights = (
        weights_long[
            weights_long["date"]
            == latest_date
        ]
        .pivot(
            index="asset",
            columns="strategy",
            values="weight",
        )
        * 100.0
    )

    print(
        latest_weights.round(2).to_string()
    )

    print()
    print("Outputs saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()