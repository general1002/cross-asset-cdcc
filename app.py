from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

SIGNAL_WEIGHTS_FILE = (
    OUTPUT_DIR / "cdcc_signal_weights.csv"
)
SIGNAL_NAV_FILE = (
    OUTPUT_DIR / "cdcc_signal_nav.csv"
)
SIGNAL_PERFORMANCE_FILE = (
    OUTPUT_DIR / "cdcc_signal_performance.csv"
)
SIGNAL_DIAGNOSTICS_FILE = (
    OUTPUT_DIR / "cdcc_signal_diagnostics.csv"
)

PORTFOLIO_WEIGHTS_FILE = (
    OUTPUT_DIR / "cdcc_portfolio_weights.csv"
)
PORTFOLIO_NAV_FILE = (
    OUTPUT_DIR / "cdcc_portfolio_nav.csv"
)
PORTFOLIO_PERFORMANCE_FILE = (
    OUTPUT_DIR / "cdcc_portfolio_performance.csv"
)

CDCC_PARAMETERS_FILE = (
    OUTPUT_DIR / "cdcc_walkforward_parameters.csv"
)

RETURN_FILE = (
    DATA_DIR / "long_short_returns_pct.csv"
)

VOLATILITY_FORECAST_FILE = (
    OUTPUT_DIR / "dcc_garch_volatility_forecast.csv"
)
PORTFOLIO_VOLATILITY_FORECAST_FILE = (
    OUTPUT_DIR / "dcc_garch_portfolio_volatility_forecast.csv"
)
WEEKLY_CORRELATION_FORECAST_FILE = (
    OUTPUT_DIR / "dcc_garch_weekly_correlation_forecast.csv"
)
HORIZON_VOLATILITY_FORECAST_FILE = (
    OUTPUT_DIR / "dcc_garch_horizon_volatility_forecast.csv"
)


st.set_page_config(
    page_title="Cross-Asset Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
)


ASSET_CLASS = {
    "Japan Equity": "Equity",
    "US Equity": "Equity",
    "Europe Equity": "Equity",
    "Emerging Equity": "Equity",
    "Japan Government Bond": "Government Bond",
    "US Treasury 7-10Y": "Government Bond",
    "Germany Government Bond": "Government Bond",
    "UK Gilt": "Government Bond",
    "US IG": "Credit",
    "US HY": "Credit",
    "Europe IG": "Credit",
    "Europe HY": "Credit",
    "US REIT": "REIT",
    "Japan REIT": "REIT",
    "Europe REIT": "REIT",
    "Crude Oil": "Commodity",
    "Gold": "Commodity",
    "Copper": "Commodity",
    "USDJPY": "FX",
    "EURJPY": "FX",
    "GBPJPY": "FX",
    "AUDJPY": "FX",
}


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        st.error(
            f"Required file not found: {path}"
        )
        st.stop()


def read_timeseries(
    path: Path,
) -> pd.DataFrame:
    require_file(path)

    frame = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    return frame.sort_index()


def read_csv(
    path: Path,
) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def format_performance(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    display = performance.copy()

    percentage_columns = [
        "annualized_return_pct",
        "annualized_volatility_pct",
        "maximum_drawdown_pct",
        "annualized_turnover_pct",
        "total_transaction_cost_pct",
    ]

    for column in percentage_columns:
        if column in display.columns:
            display[column] = display[
                column
            ].map(
                lambda value: (
                    f"{value:.2f}%"
                )
            )

    for column in [
        "sharpe_ratio",
        "calmar_ratio",
        "final_nav",
    ]:
        if column in display.columns:
            display[column] = display[
                column
            ].map(
                lambda value: (
                    f"{value:.3f}"
                )
            )

    return display


def latest_signal_positions() -> pd.DataFrame:
    weights = read_csv(
        SIGNAL_WEIGHTS_FILE
    )

    weights["date"] = pd.to_datetime(
        weights["date"]
    )

    selected = weights[
        weights["strategy"]
        == "TSMOM Inverse Volatility"
    ].copy()

    latest_date = selected["date"].max()

    selected = selected[
        selected["date"] == latest_date
    ].copy()

    selected["asset_class"] = (
        selected["asset"]
        .map(ASSET_CLASS)
        .fillna("Other")
    )

    selected["position"] = np.where(
        selected["weight"] > 1e-6,
        "LONG",
        np.where(
            selected["weight"] < -1e-6,
            "SHORT",
            "FLAT",
        ),
    )

    selected["weight_pct"] = (
        selected["weight"] * 100.0
    )

    selected = selected.sort_values(
        "weight_pct",
        ascending=True,
    )

    return selected


def show_overview() -> None:
    st.title(
        "Cross-Asset Portfolio Dashboard"
    )

    st.caption(
        "JPY-based cross-asset allocation, "
        "TSMOM signals and dynamic correlation monitoring"
    )

    signal_performance = read_timeseries(
        SIGNAL_PERFORMANCE_FILE
    )

    portfolio_performance = read_timeseries(
        PORTFOLIO_PERFORMANCE_FILE
    )

    positions = latest_signal_positions()

    signal_row = signal_performance.loc[
        "TSMOM Inverse Volatility"
    ]

    rp_row = portfolio_performance.loc[
        "Shrinkage Risk Parity"
    ]

    equal_row = portfolio_performance.loc[
        "Equal Weight"
    ]

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Latest Long Signals",
        int(
            (
                positions["position"]
                == "LONG"
            ).sum()
        ),
    )

    column2.metric(
        "Latest Short Signals",
        int(
            (
                positions["position"]
                == "SHORT"
            ).sum()
        ),
    )

    column3.metric(
        "TSMOM Sharpe",
        f"{signal_row['sharpe_ratio']:.2f}",
    )

    column4.metric(
        "TSMOM Maximum Drawdown",
        f"{signal_row['maximum_drawdown_pct']:.2f}%",
    )

    st.subheader("Model summary")

    summary = pd.DataFrame(
        {
            "Primary role": [
                "Benchmark",
                "Long-only diversified allocation",
                "Long/short directional signals",
                "Dynamic risk monitoring",
            ],
            "Selected model": [
                "Equal Weight",
                "Shrinkage Risk Parity",
                "TSMOM Inverse Volatility",
                "cDCC",
            ],
            "Out-of-sample Sharpe": [
                equal_row["sharpe_ratio"],
                rp_row["sharpe_ratio"],
                signal_row["sharpe_ratio"],
                np.nan,
            ],
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Latest TSMOM positions"
    )

    figure = px.bar(
        positions,
        x="weight_pct",
        y="asset",
        orientation="h",
        color="position",
        color_discrete_map={
            "LONG": "#2ca02c",
            "SHORT": "#d62728",
            "FLAT": "#7f7f7f",
        },
        labels={
            "weight_pct": "Weight (%)",
            "asset": "",
            "position": "Position",
        },
    )

    figure.add_vline(
        x=0,
        line_color="black",
        line_width=1,
    )

    figure.update_layout(
        height=650,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def show_signals() -> None:
    st.title(
        "Latest Long / Short Signals"
    )

    st.info(
        "方向は3・6・12カ月TSMOM、"
        "ウェイトはInverse Volatility、"
        "目標ボラティリティ10%、"
        "四半期更新です。"
    )

    weights = read_csv(
        SIGNAL_WEIGHTS_FILE
    )

    weights["date"] = pd.to_datetime(
        weights["date"]
    )

    strategies = sorted(
        weights["strategy"].unique()
    )

    default_strategy = (
        "TSMOM Inverse Volatility"
    )

    default_index = (
        strategies.index(default_strategy)
        if default_strategy in strategies
        else 0
    )

    strategy = st.selectbox(
        "Signal allocation model",
        strategies,
        index=default_index,
    )

    selected = weights[
        weights["strategy"] == strategy
    ].copy()

    latest_date = selected["date"].max()

    selected = selected[
        selected["date"] == latest_date
    ].copy()

    selected["asset_class"] = (
        selected["asset"]
        .map(ASSET_CLASS)
        .fillna("Other")
    )

    selected["position"] = np.where(
        selected["weight"] > 1e-6,
        "LONG",
        np.where(
            selected["weight"] < -1e-6,
            "SHORT",
            "FLAT",
        ),
    )

    selected["weight_pct"] = (
        selected["weight"] * 100.0
    )

    gross = selected[
        "weight"
    ].abs().sum()

    net = selected["weight"].sum()

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "Signal Date",
        latest_date.strftime("%Y-%m-%d"),
    )

    column2.metric(
        "Gross Exposure",
        f"{gross * 100:.2f}%",
    )

    column3.metric(
        "Net Exposure",
        f"{net * 100:.2f}%",
    )

    chart_data = selected.sort_values(
        "weight_pct",
        ascending=True,
    )

    figure = px.bar(
        chart_data,
        x="weight_pct",
        y="asset",
        orientation="h",
        color="position",
        hover_data=[
            "asset_class",
            "signal",
        ],
        color_discrete_map={
            "LONG": "#2ca02c",
            "SHORT": "#d62728",
            "FLAT": "#7f7f7f",
        },
        labels={
            "weight_pct": "Weight (%)",
            "asset": "",
        },
    )

    figure.add_vline(
        x=0,
        line_color="black",
        line_width=1,
    )

    figure.update_layout(
        height=700,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    table = selected[
        [
            "asset",
            "asset_class",
            "position",
            "signal",
            "weight_pct",
        ]
    ].sort_values(
        "weight_pct",
        ascending=False,
    )

    table = table.rename(
        columns={
            "asset": "Asset",
            "asset_class": "Asset Class",
            "position": "Position",
            "signal": "Signal",
            "weight_pct": "Weight (%)",
        }
    )

    st.dataframe(
        table.round(3),
        use_container_width=True,
        hide_index=True,
    )


def show_portfolio_weights() -> None:
    st.title(
        "Portfolio Allocation"
    )

    weights = read_csv(
        PORTFOLIO_WEIGHTS_FILE
    )

    weights["date"] = pd.to_datetime(
        weights["date"]
    )

    strategies = sorted(
        weights["strategy"].unique()
    )

    default_strategy = (
        "Shrinkage Risk Parity"
    )

    default_index = (
        strategies.index(default_strategy)
        if default_strategy in strategies
        else 0
    )

    strategy = st.selectbox(
        "Portfolio model",
        strategies,
        index=default_index,
    )

    selected = weights[
        weights["strategy"] == strategy
    ].copy()

    latest_date = selected["date"].max()

    selected = selected[
        selected["date"] == latest_date
    ].copy()

    selected["asset_class"] = (
        selected["asset"]
        .map(ASSET_CLASS)
        .fillna("Other")
    )

    selected["weight_pct"] = (
        selected["weight"] * 100.0
    )

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "Weight Date",
        latest_date.strftime("%Y-%m-%d"),
    )

    column2.metric(
        "Maximum Weight",
        f"{selected['weight_pct'].max():.2f}%",
    )

    effective_assets = (
        1.0
        / np.square(
            selected["weight"]
        ).sum()
    )

    column3.metric(
        "Effective Asset Count",
        f"{effective_assets:.1f}",
    )

    chart_data = selected.sort_values(
        "weight_pct",
        ascending=True,
    )

    figure = px.bar(
        chart_data,
        x="weight_pct",
        y="asset",
        orientation="h",
        color="asset_class",
        labels={
            "weight_pct": "Weight (%)",
            "asset": "",
            "asset_class": "Asset Class",
        },
    )

    figure.update_layout(
        height=650,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    class_weights = (
        selected.groupby(
            "asset_class"
        )["weight"]
        .sum()
        .mul(100.0)
        .sort_values(
            ascending=False
        )
    )

    st.subheader(
        "Asset-class allocation"
    )

    st.dataframe(
        class_weights
        .rename("Weight (%)")
        .round(2)
        .to_frame(),
        use_container_width=True,
    )


def show_backtests() -> None:
    st.title(
        "Backtest Comparison"
    )

    signal_nav = read_timeseries(
        SIGNAL_NAV_FILE
    )

    portfolio_nav = read_timeseries(
        PORTFOLIO_NAV_FILE
    )

    signal_nav = signal_nav.rename(
        columns={
            "Equal Weight": (
                "Equal Weight – Signal sample"
            )
        }
    )

    combined_nav = pd.concat(
        [
            portfolio_nav,
            signal_nav.drop(
                columns=[
                    "Equal Weight – Signal sample"
                ],
                errors="ignore",
            ),
        ],
        axis=1,
    )

    available = list(
        combined_nav.columns
    )

    defaults = [
        name
        for name in [
            "Equal Weight",
            "Shrinkage Risk Parity",
            "cDCC Minimum Variance",
            "TSMOM Inverse Volatility",
        ]
        if name in available
    ]

    selected_strategies = st.multiselect(
        "Strategies",
        available,
        default=defaults,
    )

    if selected_strategies:
        chart = go.Figure()

        for strategy in selected_strategies:
            series = (
                combined_nav[strategy]
                .dropna()
            )

            chart.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    name=strategy,
                    mode="lines",
                )
            )

        chart.update_layout(
            height=550,
            xaxis_title="",
            yaxis_title="NAV",
            hovermode="x unified",
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    st.subheader(
        "Long-only portfolio performance"
    )

    portfolio_performance = read_timeseries(
        PORTFOLIO_PERFORMANCE_FILE
    )

    st.dataframe(
        format_performance(
            portfolio_performance
        ),
        use_container_width=True,
    )

    st.subheader(
        "Signal strategy performance"
    )

    signal_performance = read_timeseries(
        SIGNAL_PERFORMANCE_FILE
    )

    st.dataframe(
        format_performance(
            signal_performance
        ),
        use_container_width=True,
    )


def show_correlations() -> None:
    st.title(
        "Realized Correlation Monitor"
    )

    returns = read_timeseries(
        RETURN_FILE
    )

    maximum_window = min(
        504,
        len(returns),
    )

    window = st.slider(
        "Realized correlation window",
        min_value=63,
        max_value=maximum_window,
        value=min(126, maximum_window),
        step=21,
    )

    correlation = (
        returns.tail(window).corr()
    )

    figure = go.Figure(
        data=go.Heatmap(
            z=correlation.to_numpy(),
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={
                "title": "Correlation"
            },
        )
    )

    figure.update_layout(
        height=800,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.caption(
        f"Latest {window} common observations. "
        "This page shows historical realized correlation, "
        "not the cDCC forecast correlation."
    )


def show_risk_forecast() -> None:
    st.title(
        "Weekly DCC-GARCH Risk Forecast"
    )

    require_file(
        VOLATILITY_FORECAST_FILE
    )
    require_file(
        WEEKLY_CORRELATION_FORECAST_FILE
    )

    forecast = pd.read_csv(
        VOLATILITY_FORECAST_FILE,
        parse_dates=["forecast_origin"],
    )

    latest_origin = forecast[
        "forecast_origin"
    ].max()

    forecast = forecast[
        forecast["forecast_origin"]
        == latest_origin
    ].copy()

    risk_ratio_column = (
        "weekly_forecast_vs_63d_ratio"
    )

    rising_risk_count = int(
        (
            forecast[risk_ratio_column]
            > 1.0
        ).sum()
    )

    highest_risk_row = forecast.loc[
        forecast[
            risk_ratio_column
        ].idxmax()
    ]

    highest_volatility_row = forecast.loc[
        forecast[
            "next_week_annualized_volatility_pct"
        ].idxmax()
    ]

    largest_increase = (
        highest_risk_row[
            risk_ratio_column
        ]
        - 1.0
    ) * 100.0

    highest_forecast_volatility = (
        highest_volatility_row[
            "next_week_annualized_volatility_pct"
        ]
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Forecast Origin",
        latest_origin.strftime(
            "%Y-%m-%d"
        ),
    )

    column2.metric(
        "Assets with Rising Risk",
        f"{rising_risk_count} / {len(forecast)}",
    )

    column3.metric(
        "Largest Risk Increase",
        highest_risk_row["asset"],
        delta=(
            f"{largest_increase:.1f}% vs 63-day"
        ),
    )

    column4.metric(
        "Highest Forecast Volatility",
        highest_volatility_row["asset"],
        delta=(
            f"{highest_forecast_volatility:.1f}% annualized"
        ),
        delta_color="off",
    )

    st.caption(
        "The model forecasts five daily GARCH variance "
        "paths and combines them with the expected cDCC "
        "correlation path. The forecast concerns the "
        "magnitude of risk, not return direction."
    )

    st.subheader(
        "Forecast versus realized volatility"
    )

    chart_data = forecast[
        [
            "asset",
            "next_week_annualized_volatility_pct",
            "realized_63d_annualized_volatility_pct",
        ]
    ].rename(
        columns={
            "next_week_annualized_volatility_pct": (
                "DCC-GARCH Weekly Forecast"
            ),
            "realized_63d_annualized_volatility_pct": (
                "63-Day Realized"
            ),
        }
    )

    chart_data = chart_data.melt(
        id_vars="asset",
        var_name="Measure",
        value_name="Annualized Volatility (%)",
    )

    asset_order = forecast.sort_values(
        "next_week_annualized_volatility_pct",
        ascending=False,
    )["asset"].tolist()

    volatility_chart = px.bar(
        chart_data,
        x="asset",
        y="Annualized Volatility (%)",
        color="Measure",
        barmode="group",
        category_orders={
            "asset": asset_order
        },
        color_discrete_map={
            "DCC-GARCH Weekly Forecast": "#1f77b4",
            "63-Day Realized": "#b8b8b8",
        },
    )

    volatility_chart.update_layout(
        height=600,
        xaxis_tickangle=-45,
        legend_title="",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=150,
        ),
    )

    st.plotly_chart(
        volatility_chart,
        use_container_width=True,
    )

    st.subheader(
        "Forecast risk change"
    )

    ratio_data = forecast.copy()

    ratio_data["risk_change_pct"] = (
        ratio_data[risk_ratio_column]
        - 1.0
    ) * 100.0

    ratio_data["risk_direction"] = np.where(
        ratio_data["risk_change_pct"] >= 0,
        "Risk Increasing",
        "Risk Decreasing",
    )

    ratio_data = ratio_data.sort_values(
        "risk_change_pct",
        ascending=True,
    )

    risk_chart = px.bar(
        ratio_data,
        x="risk_change_pct",
        y="asset",
        orientation="h",
        color="risk_direction",
        color_discrete_map={
            "Risk Increasing": "#d62728",
            "Risk Decreasing": "#2ca02c",
        },
        labels={
            "risk_change_pct": (
                "Forecast change vs 63-day realized (%)"
            ),
            "asset": "",
            "risk_direction": "",
        },
    )

    risk_chart.add_vline(
        x=0,
        line_color="black",
        line_width=1,
    )

    risk_chart.update_layout(
        height=700,
        legend_title="",
    )

    st.plotly_chart(
        risk_chart,
        use_container_width=True,
    )

    st.subheader(
        "Portfolio weekly risk forecast"
    )

    if PORTFOLIO_VOLATILITY_FORECAST_FILE.exists():
        portfolio_forecast = pd.read_csv(
            PORTFOLIO_VOLATILITY_FORECAST_FILE,
            index_col=0,
        )

        portfolio_display = portfolio_forecast[
            [
                "next_week_volatility_pct",
                "next_week_annualized_volatility_pct",
                "gross_exposure_pct",
                "net_exposure_pct",
            ]
        ].copy()

        portfolio_display = portfolio_display.rename(
            columns={
                "next_week_volatility_pct": (
                    "Next Week Volatility (%)"
                ),
                "next_week_annualized_volatility_pct": (
                    "Annualized Volatility (%)"
                ),
                "gross_exposure_pct": (
                    "Gross Exposure (%)"
                ),
                "net_exposure_pct": (
                    "Net Exposure (%)"
                ),
            }
        )

        st.dataframe(
            portfolio_display.round(2),
            use_container_width=True,
        )

        excess_gross = (
            portfolio_forecast[
                "gross_exposure_pct"
            ]
            > 150.0
        )

        if excess_gross.any():
            affected = ", ".join(
                portfolio_forecast.index[
                    excess_gross
                ].tolist()
            )

            st.warning(
                "Gross exposure exceeds 150% after "
                "weight drift: "
                f"{affected}"
            )
    else:
        st.info(
            "Portfolio forecast file is not available. "
            "Run python forecast_dcc_garch.py first."
        )

    st.subheader(
        "Weekly cDCC correlation forecast"
    )

    correlation = pd.read_csv(
        WEEKLY_CORRELATION_FORECAST_FILE,
        index_col=0,
    )

    correlation_chart = go.Figure(
        data=go.Heatmap(
            z=correlation.to_numpy(),
            x=correlation.columns,
            y=correlation.index,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            colorbar={
                "title": "Correlation"
            },
        )
    )

    correlation_chart.update_layout(
        height=800,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        correlation_chart,
        use_container_width=True,
    )

    st.caption(
        "This is the effective five-day correlation "
        "implied by the summed DCC-GARCH covariance "
        "forecasts, not a historical realized correlation."
    )

    st.subheader(
        "Detailed asset forecast"
    )

    detail = forecast[
        [
            "asset",
            "next_day_volatility_pct",
            "next_week_volatility_pct",
            "next_week_annualized_volatility_pct",
            "realized_63d_annualized_volatility_pct",
            "realized_252d_annualized_volatility_pct",
            "weekly_forecast_vs_63d_ratio",
        ]
    ].copy()

    detail = detail.sort_values(
        "weekly_forecast_vs_63d_ratio",
        ascending=False,
    )

    detail = detail.rename(
        columns={
            "asset": "Asset",
            "next_day_volatility_pct": (
                "Next Day Vol (%)"
            ),
            "next_week_volatility_pct": (
                "Next Week Vol (%)"
            ),
            "next_week_annualized_volatility_pct": (
                "Forecast Annualized Vol (%)"
            ),
            "realized_63d_annualized_volatility_pct": (
                "63-Day Realized Vol (%)"
            ),
            "realized_252d_annualized_volatility_pct": (
                "252-Day Realized Vol (%)"
            ),
            "weekly_forecast_vs_63d_ratio": (
                "Forecast / 63-Day"
            ),
        }
    )

    st.dataframe(
        detail.round(3),
        use_container_width=True,
        hide_index=True,
    )


def show_model_diagnostics() -> None:
    st.title(
        "Model Diagnostics"
    )

    parameters = read_timeseries(
        CDCC_PARAMETERS_FILE
    )

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=parameters.index,
            y=parameters[
                "cdcc_persistence"
            ],
            name="cDCC persistence",
            mode="lines+markers",
        )
    )

    chart.add_hline(
        y=0.99,
        line_dash="dash",
        line_color="red",
    )

    chart.update_layout(
        height=450,
        yaxis_title="a + b",
        xaxis_title="",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "Maximum Persistence",
        f"{parameters['cdcc_persistence'].max():.4f}",
    )

    column2.metric(
        "Latest Persistence",
        f"{parameters['cdcc_persistence'].iloc[-1]:.4f}",
    )

    converged = parameters[
        "cdcc_converged"
    ].astype(str).str.lower().eq(
        "true"
    ).all()

    column3.metric(
        "All Fits Converged",
        "Yes" if converged else "No",
    )

    st.dataframe(
        parameters.round(6),
        use_container_width=True,
    )

    if SIGNAL_DIAGNOSTICS_FILE.exists():
        st.subheader(
            "Signal portfolio diagnostics"
        )

        diagnostics = read_csv(
            SIGNAL_DIAGNOSTICS_FILE
        )

        diagnostics["date"] = pd.to_datetime(
            diagnostics["date"]
        )

        strategy = st.selectbox(
            "Diagnostic strategy",
            sorted(
                diagnostics[
                    "strategy"
                ].unique()
            ),
        )

        selected = diagnostics[
            diagnostics["strategy"]
            == strategy
        ].copy()

        diagnostic_chart = go.Figure()

        diagnostic_chart.add_trace(
            go.Scatter(
                x=selected["date"],
                y=(
                    selected[
                        "gross_exposure"
                    ]
                    * 100.0
                ),
                name="Gross Exposure",
                mode="lines+markers",
            )
        )

        diagnostic_chart.add_trace(
            go.Scatter(
                x=selected["date"],
                y=(
                    selected[
                        "net_exposure"
                    ]
                    * 100.0
                ),
                name="Net Exposure",
                mode="lines+markers",
            )
        )

        diagnostic_chart.add_trace(
            go.Scatter(
                x=selected["date"],
                y=(
                    selected[
                        "predicted_volatility"
                    ]
                    * 100.0
                ),
                name="Predicted Volatility",
                mode="lines+markers",
            )
        )

        diagnostic_chart.update_layout(
            height=450,
            yaxis_title="Percent",
            hovermode="x unified",
        )

        st.plotly_chart(
            diagnostic_chart,
            use_container_width=True,
        )


page = st.sidebar.selectbox(
    "Page",
    [
        "Overview",
        "Signals",
        "Portfolio Weights",
        "Risk Forecast",
        "Backtests",
        "Correlations",
        "Model Diagnostics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Research dashboard — not investment advice"
)




if page == "Overview":
    show_overview()
elif page == "Signals":
    show_signals()
elif page == "Portfolio Weights":
    show_portfolio_weights()
elif page == "Risk Forecast":
    show_risk_forecast()
elif page == "Backtests":
    show_backtests()
elif page == "Correlations":
    show_correlations()
elif page == "Model Diagnostics":
    show_model_diagnostics()
