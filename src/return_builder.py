from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(
    prices: pd.DataFrame,
    scale: float = 100.0,
) -> pd.DataFrame:
    """Calculate returns between consecutive valid observations."""
    if (prices.dropna(how="all") <= 0).any().any():
        raise ValueError(
            "Prices must be positive to calculate log returns."
        )

    returns = pd.DataFrame(
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )

    for column in prices.columns:
        valid_prices = prices[column].dropna()

        asset_returns = scale * np.log(
            valid_prices / valid_prices.shift(1)
        )

        returns.loc[asset_returns.index, column] = asset_returns

    return returns


def data_diagnostics(prices: pd.DataFrame) -> pd.DataFrame:
    returns = log_returns(prices)
    rows = []

    for column in prices.columns:
        price = prices[column]
        ret = returns[column]
        available = price.dropna()
        non_missing_returns = ret.dropna()

        rows.append(
            {
                "asset": column,
                "first_date": available.index.min() if not available.empty else pd.NaT,
                "last_date": available.index.max() if not available.empty else pd.NaT,
                "price_observations": int(available.size),
                "missing_price_pct": float(100 * price.isna().mean()),
                "zero_return_pct": (
                    float(100 * non_missing_returns.eq(0).mean())
                    if not non_missing_returns.empty
                    else np.nan
                ),
                "annualized_vol_pct": (
                    float(non_missing_returns.std() * np.sqrt(252))
                    if len(non_missing_returns) > 1
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).set_index("asset")
