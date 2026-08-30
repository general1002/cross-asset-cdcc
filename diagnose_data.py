from pathlib import Path

from src.data_loader import (
    download_adjusted_close,
    load_config,
)
from src.return_builder import (
    data_diagnostics,
    log_returns,
)


ROOT = Path(__file__).resolve().parent


def apply_known_data_fixes(prices):
    """Exclude confirmed vendor data errors."""

    # IDEU.L temporarily jumped on 2025-10-24 without
    # a distribution or stock split and returned to its
    # previous price level on the next trading day.
    prices.loc[
        "2025-10-24",
        "Germany Government Bond",
    ] = float("nan")

    return prices


def main() -> None:
    config, assets = load_config(
        ROOT / "config" / "assets.yaml"
    )

    prices = download_adjusted_close(
        assets,
        start=config["start"],
        end=config.get("end"),
        interval=config.get("interval", "1d"),
    )

    prices = apply_known_data_fixes(prices)

    returns = log_returns(prices)
    diagnostics = data_diagnostics(prices)

    processed = ROOT / "data" / "processed"
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
        processed / "adjusted_close.csv"
    )

    returns.to_csv(
        processed / "log_returns_pct.csv"
    )

    diagnostics.to_csv(
        outputs / "data_diagnostics.csv"
    )

    print(diagnostics.to_string())

    print(
        f"\nSaved prices and returns under: "
        f"{processed}"
    )

    print(
        f"Saved diagnostics under: "
        f"{outputs}"
    )


if __name__ == "__main__":
    main()