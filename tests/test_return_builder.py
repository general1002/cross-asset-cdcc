import numpy as np
import pandas as pd

from src.return_builder import data_diagnostics, log_returns


def test_log_returns_uses_consecutive_valid_prices() -> None:
    index = pd.date_range("2026-01-01", periods=4)

    prices = pd.DataFrame(
        {"asset": [100.0, 101.0, np.nan, 102.0]},
        index=index,
    )

    result = log_returns(prices)

    assert np.isnan(result.iloc[0, 0])

    assert np.isclose(
        result.iloc[1, 0],
        100 * np.log(101.0 / 100.0),
    )

    assert np.isnan(result.iloc[2, 0])

    assert np.isclose(
        result.iloc[3, 0],
        100 * np.log(102.0 / 101.0),
    )


def test_diagnostics_detects_zero_returns() -> None:
    index = pd.date_range("2026-01-01", periods=4)

    prices = pd.DataFrame(
        {"asset": [100.0, 100.0, 101.0, 101.0]},
        index=index,
    )

    result = data_diagnostics(prices)

    assert np.isclose(
        result.loc["asset", "zero_return_pct"],
        200.0 / 3.0,
    )