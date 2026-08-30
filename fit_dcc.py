from pathlib import Path

import numpy as np
import pandas as pd

from src.dcc import estimate_dcc


ROOT = Path(__file__).resolve().parent


def create_summary(
    result: dict,
    observations: int,
) -> dict:
    parameter_count = 2

    negative_loglikelihood = result[
        "negative_loglikelihood"
    ]

    aic = (
        2.0 * negative_loglikelihood
        + 2.0 * parameter_count
    )

    bic = (
        2.0 * negative_loglikelihood
        + parameter_count
        * np.log(observations)
    )

    return {
        "model": result["model"],
        "a": result["a"],
        "b": result["b"],
        "persistence": result["persistence"],
        "negative_loglikelihood": (
            negative_loglikelihood
        ),
        "aic": aic,
        "bic": bic,
        "shrinkage": result["shrinkage"],
        "converged": result["converged"],
        "iterations": result["iterations"],
        "optimization_message": result[
            "optimization_message"
        ],
    }


def correlation_frame(
    matrix: np.ndarray,
    assets: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=assets,
        columns=assets,
    )


def create_pair_summary(
    correlations: np.ndarray,
    assets: list[str],
) -> pd.DataFrame:
    latest = correlations[-1]
    average = correlations.mean(axis=0)

    lookback = min(
        60,
        len(correlations) - 1,
    )

    previous = correlations[
        -1 - lookback
    ]

    rows = []

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            rows.append(
                {
                    "asset_1": assets[i],
                    "asset_2": assets[j],
                    "latest_correlation": (
                        latest[i, j]
                    ),
                    "average_correlation": (
                        average[i, j]
                    ),
                    "change_60_observations": (
                        latest[i, j]
                        - previous[i, j]
                    ),
                    "deviation_from_average": (
                        latest[i, j]
                        - average[i, j]
                    ),
                }
            )

    pairs = pd.DataFrame(rows)

    pairs["absolute_change_60"] = (
        pairs[
            "change_60_observations"
        ].abs()
    )

    pairs["absolute_deviation"] = (
        pairs[
            "deviation_from_average"
        ].abs()
    )

    return pairs.sort_values(
        "absolute_change_60",
        ascending=False,
    )


def save_model_outputs(
    result: dict,
    dates: pd.Index,
    assets: list[str],
    processed: Path,
    outputs: Path,
) -> None:
    model_key = result["model"].lower()

    correlations = result["r_history"]

    correlation_frame(
        correlations[-1],
        assets,
    ).to_csv(
        outputs
        / f"{model_key}_latest_correlation.csv"
    )

    correlation_frame(
        correlations.mean(axis=0),
        assets,
    ).to_csv(
        outputs
        / f"{model_key}_average_correlation.csv"
    )

    correlation_frame(
        result["target"],
        assets,
    ).to_csv(
        outputs
        / f"{model_key}_shrinkage_target.csv"
    )

    pair_summary = create_pair_summary(
        correlations,
        assets,
    )

    pair_summary.to_csv(
        outputs
        / f"{model_key}_pair_summary.csv",
        index=False,
    )

    np.savez_compressed(
        processed
        / f"{model_key}_correlation_history.npz",
        dates=np.asarray(
            dates.astype(str)
        ),
        assets=np.asarray(assets),
        correlations=correlations,
    )


def main() -> None:
    residual_path = (
        ROOT
        / "data"
        / "processed"
        / "selected_standardized_residuals.csv"
    )

    residuals = pd.read_csv(
        residual_path,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    if residuals.isna().any().any():
        missing = int(
            residuals.isna().sum().sum()
        )

        raise ValueError(
            f"Standardized residual matrix "
            f"contains {missing} missing values."
        )

    values = residuals.to_numpy(
        dtype=float
    )

    assets = residuals.columns.tolist()

    print(
        f"Observations: {values.shape[0]}"
    )

    print(
        f"Assets: {values.shape[1]}"
    )

    print(
        "\nEstimating standard DCC..."
    )

    dcc_result = estimate_dcc(
        values,
        corrected=False,
    )

    print(
        "Standard DCC completed: "
        f"a={dcc_result['a']:.6f}, "
        f"b={dcc_result['b']:.6f}, "
        f"a+b={dcc_result['persistence']:.6f}, "
        f"converged={dcc_result['converged']}"
    )

    print(
        "\nEstimating corrected cDCC..."
    )

    cdcc_result = estimate_dcc(
        values,
        corrected=True,
    )

    print(
        "cDCC completed: "
        f"a={cdcc_result['a']:.6f}, "
        f"b={cdcc_result['b']:.6f}, "
        f"a+b={cdcc_result['persistence']:.6f}, "
        f"converged={cdcc_result['converged']}"
    )

    processed = (
        ROOT
        / "data"
        / "processed"
    )

    outputs = ROOT / "outputs"

    processed.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model_outputs(
        dcc_result,
        residuals.index,
        assets,
        processed,
        outputs,
    )

    save_model_outputs(
        cdcc_result,
        residuals.index,
        assets,
        processed,
        outputs,
    )

    comparison = pd.DataFrame(
        [
            create_summary(
                dcc_result,
                len(residuals),
            ),
            create_summary(
                cdcc_result,
                len(residuals),
            ),
        ]
    ).set_index("model")

    comparison.to_csv(
        outputs
        / "dcc_model_comparison.csv"
    )

    print("\nModel comparison:")
    print(
        comparison[
            [
                "a",
                "b",
                "persistence",
                "negative_loglikelihood",
                "aic",
                "bic",
                "shrinkage",
                "converged",
                "iterations",
            ]
        ].to_string()
    )

    cdcc_pairs = create_pair_summary(
        cdcc_result["r_history"],
        assets,
    )

    print(
        "\nLargest cDCC correlation changes "
        "over 60 common observations:"
    )

    print(
        cdcc_pairs[
            [
                "asset_1",
                "asset_2",
                "latest_correlation",
                "change_60_observations",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    if (
        dcc_result["negative_loglikelihood"]
        < cdcc_result[
            "negative_loglikelihood"
        ]
    ):
        preferred = "DCC"
    else:
        preferred = "cDCC"

    print(
        f"\nLower in-sample correlation "
        f"negative log-likelihood: {preferred}"
    )

    print(
        "\nOutputs saved under:"
    )

    print(outputs)


if __name__ == "__main__":
    main()