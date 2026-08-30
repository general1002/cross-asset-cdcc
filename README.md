# Cross-Asset cDCC

Research project for cross-asset volatility and dynamic-correlation modelling.

## First milestone

Download candidate adjusted-close series and diagnose:

- available history;
- missing observations;
- zero-return frequency;
- annualized raw-return volatility.

European-listed tickers marked `candidate: true` in `config/assets.yaml` must be
confirmed from the first diagnostic run. Yahoo symbols can vary by exchange and
share class.

## Run

```bat
conda activate cdcc-app
python -m pip install -r requirements.txt
pytest -q
python diagnose_data.py
```

Outputs are written to `data/processed/` and `outputs/` and are excluded from Git.

## Important modelling convention

The initial universe mixes local-market and US-listed ETF prices. Before cDCC
estimation, each foreign asset must be converted to a clearly defined local-currency
return or JPY-investor return. FX returns must not be counted twice.
