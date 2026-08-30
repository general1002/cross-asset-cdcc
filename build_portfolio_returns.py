from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from src.return_builder import (
    data_diagnostics,
    log_returns,
)


ROOT = Path(__file__).resolve().parent


def load_portfolio_config() -> dict:
    path = (
        ROOT
        / "config"
        / "portfolio_assets.yaml"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    interval: str,
) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no observations."
        )

    if isinstance(
        raw.columns,
        pd.MultiIndex,
    ):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    close.index = pd.to_datetime(
        close.index
    )

    if close.index.tz is not None:
        close.index = (
            close.index.tz_localize(None)
        )

    return (
        close
        .sort_index()
        .astype(float)
    )


def apply_known_data_fixes(
    prices: pd.DataFrame,
    asset_tickers: dict[str, str],
) -> pd.DataFrame:
    prices = prices.copy()

    germany_ticker = asset_tickers[
        "Germany Government Bond"
    ]

    # Confirmed isolated Yahoo Finance error.
    prices.loc[
        "2025-10-24",
        germany_ticker,
    ] = float("nan")

    return prices


def build_jpy_prices(
    local_prices: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    asset_config = config["assets"]
    fx_config = config["fx_overlays"]

    currency_to_fx_ticker = {
        values["foreign_currency"]: (
            values["ticker"]
        )
        for values in fx_config.values()
    }

    jpy_prices = pd.DataFrame(
        index=local_prices.index
    )

    for asset, values in asset_config.items():
        ticker = values["ticker"]
        currency = values["trading_currency"]

        price = local_prices[ticker]

        if currency == "JPY":
            jpy_price = price
        else:
            fx_ticker = currency_to_fx_ticker[
                currency
            ]

            # FX trades on more calendar days than ETFs.
            # Carry the latest FX price for at most five days.
            fx_price = (
                local_prices[fx_ticker]
                .ffill(limit=5)
            )

            jpy_price = price * fx_price

        jpy_prices[asset] = jpy_price

    return jpy_prices


def build_fx_overlay_prices(
    prices: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    fx_prices = pd.DataFrame(
        index=prices.index
    )

    for name, values in (
        config["fx_overlays"].items()
    ):
        fx_prices[name] = prices[
            values["ticker"]
        ]

    return fx_prices


def build_universe_metadata(
    config: dict,
) -> pd.DataFrame:
    rows = []

    for asset, values in (
        config["assets"].items()
    ):
        rows.append(
            {
                "asset": asset,
                "ticker": values["ticker"],
                "asset_class": (
                    values["asset_class"]
                ),
                "trading_currency": (
                    values["trading_currency"]
                ),
                "portfolio_role": (
                    "funded_asset"
                ),
            }
        )

    for asset, values in (
        config["fx_overlays"].items()
    ):
        rows.append(
            {
                "asset": asset,
                "ticker": values["ticker"],
                "asset_class": "FX",
                "trading_currency": (
                    values["foreign_currency"]
                ),
                "portfolio_role": (
                    "currency_overlay"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("asset")
    )


def main() -> None:
    config = load_portfolio_config()

    asset_tickers = {
        asset: values["ticker"]
        for asset, values
        in config["assets"].items()
    }

    fx_tickers = {
        name: values["ticker"]
        for name, values
        in config["fx_overlays"].items()
    }

    tickers = list(
        dict.fromkeys(
            list(asset_tickers.values())
            + list(fx_tickers.values())
        )
    )

    print(
        f"Downloading {len(tickers)} "
        "portfolio price series..."
    )

    prices = download_prices(
        tickers=tickers,
        start=config["start"],
        end=config.get("end"),
        interval=config.get(
            "interval",
            "1d",
        ),
    )

    prices = apply_known_data_fixes(
        prices,
        asset_tickers,
    )

    jpy_prices = build_jpy_prices(
        prices,
        config,
    )

    fx_prices = build_fx_overlay_prices(
        prices,
        config,
    )

    funded_returns = log_returns(
        jpy_prices
    )

    fx_returns = log_returns(
        fx_prices
    )

    long_only_returns = (
        funded_returns
        .dropna(how="any")
    )

    long_short_returns = (
        pd.concat(
            [
                funded_returns,
                fx_returns,
            ],
            axis=1,
        )
        .dropna(how="any")
    )

    diagnostics = data_diagnostics(
        jpy_prices
    )

    metadata = build_universe_metadata(
        config
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

    prices.to_csv(
        processed
        / "portfolio_local_prices.csv"
    )

    jpy_prices.to_csv(
        processed
        / "portfolio_jpy_prices.csv"
    )

    funded_returns.to_csv(
        processed
        / "portfolio_jpy_returns_pct.csv"
    )

    fx_returns.to_csv(
        processed
        / "fx_overlay_returns_pct.csv"
    )

    long_only_returns.to_csv(
        processed
        / "long_only_returns_pct.csv"
    )

    long_short_returns.to_csv(
        processed
        / "long_short_returns_pct.csv"
    )

    diagnostics.to_csv(
        outputs
        / "portfolio_data_diagnostics.csv"
    )

    metadata.to_csv(
        outputs
        / "portfolio_universe.csv"
    )

    print(
        "\nLong-only return matrix:"
    )

    print(
        long_only_returns.shape
    )

    print(
        long_only_returns.index.min(),
        "to",
        long_only_returns.index.max(),
    )

    print(
        "\nLong/short return matrix:"
    )

    print(
        long_short_returns.shape
    )

    print(
        long_short_returns.index.min(),
        "to",
        long_short_returns.index.max(),
    )

    print(
        "\nAnnualized JPY volatility:"
    )

    annualized_volatility = (
        long_only_returns.std()
        * np.sqrt(252)
    )

    print(
        annualized_volatility
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print(
        "\nMaximum absolute daily "
        "JPY returns:"
    )

    print(
        long_only_returns
        .abs()
        .max()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


if __name__ == "__main__":
    main()