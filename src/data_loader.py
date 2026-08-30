from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf


@dataclass(frozen=True)
class AssetSpec:
    name: str
    ticker: str
    asset_class: str
    currency: str
    candidate: bool = False


def load_config(path: str | Path) -> tuple[dict[str, Any], list[AssetSpec]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assets = [
        AssetSpec(
            name=name,
            ticker=values["ticker"],
            asset_class=values["asset_class"],
            currency=values["currency"],
            candidate=bool(values.get("candidate", False)),
        )
        for name, values in config["assets"].items()
    ]
    return config, assets


def download_adjusted_close(
    assets: list[AssetSpec],
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    ticker_to_name = {asset.ticker: asset.name for asset in assets}
    tickers = list(ticker_to_name)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no observations.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Downloaded data has no Close field.")
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise RuntimeError("Downloaded data has no Close field.")
        close = raw[["Close"]].copy()
        close.columns = tickers

    close = close.rename(columns=ticker_to_name)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index().astype(float)
