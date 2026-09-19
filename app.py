from __future__ import annotations

from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import load_market_data
from src.pipeline import run_pipeline

DISCLAIMER = (
    "**Educational / research use only.** This project is a portfolio demonstration of financial "
    "machine learning. Forecasts are uncertain and are not financial advice or trading recommendations."
)


def price_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="ABNB OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], name="Volume", opacity=0.45),
        row=2,
        col=1,
    )
    fig.update_layout(
        title="ABNB price and trading volume",
        height=620,
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend_orientation="h",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def forecast_figure(pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["date"], y=pred["actual_vol"], mode="lines", name="Realized volatility"))
    fig.add_trace(go.Scatter(x=pred["date"], y=pred["lstm_vol"], mode="lines", name="LSTM forecast"))
    fig.add_trace(go.Scatter(x=pred["date"], y=pred["ewma_vol"], mode="lines", name="EWMA baseline"))
    fig.update_layout(
        title="Out-of-sample 5-day volatility forecasting",
        xaxis_title="Date",
        yaxis_title="Annualized volatility",
        yaxis_tickformat=".0%",
        height=500,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
        legend_orientation="h",
    )
    return fig


def loss_figure(train_loss: list[float], val_loss: list[float]) -> go.Figure:
    epochs = list(range(1, len(train_loss) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Training loss"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Validation loss"))
    fig.update_layout(
        title="Training diagnostics",
        xaxis_title="Epoch",
        yaxis_title="Mean squared error",
        height=360,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


@lru_cache(maxsize=6)
def _cached_run(lookback: int, horizon: int, epochs: int):
    market, source = load_market_data()
    result = run_pipeline(market, lookback=lookback, horizon=horizon, epochs=epochs)
    return market, source, result


def analyze(lookback: int, horizon: int, epochs: int, progress=gr.Progress()):
    try:
        progress(0.05, desc="Loading ABNB market data")
        market, source, result = _cached_run(int(lookback), int(horizon), int(epochs))
        progress(0.95, desc="Rendering out-of-sample results")

        metrics = result.metrics.copy()
        for c in ["RMSE", "MAE", "R²", "Correlation"]:
            metrics[c] = metrics[c].map(lambda x: None if pd.isna(x) else round(float(x), 4))

        lstm_rmse = float(result.metrics.loc[result.metrics["Model"] == "LSTM", "RMSE"].iloc[0])
        ewma_rmse = float(result.metrics.loc[result.metrics["Model"] == "EWMA baseline", "RMSE"].iloc[0])
        improvement = (ewma_rmse - lstm_rmse) / ewma_rmse if ewma_rmse else np.nan
        verdict = (
            f"LSTM reduces test RMSE by **{improvement:.1%}** versus EWMA."
            if np.isfinite(improvement) and improvement > 0
            else f"EWMA remains stronger on this test split; LSTM RMSE is **{abs(improvement):.1%}** worse."
            if np.isfinite(improvement)
            else "The baseline comparison could not be summarized."
        )

        first_date = market["date"].min().date().isoformat()
        last_date = market["date"].max().date().isoformat()
        summary = f"""
### Model result

- **Data source:** {source}
- **Observed period:** {first_date} to {last_date} ({len(market):,} trading days)
- **Task:** use the previous **{lookback} trading days** to forecast annualized realized volatility over the **next {horizon} trading days**
- **Validation-selected epoch:** {result.training.best_epoch}
- **Latest LSTM forecast:** **{result.next_forecast:.1%} annualized volatility** for the upcoming {horizon}-day horizon
- **Out-of-sample comparison:** {verdict}

The test period is held out chronologically; future observations are not shuffled into training. A weaker LSTM result is still meaningful evidence that a traditional risk model can be difficult to beat on a single-stock daily dataset.
"""
        recent = result.predictions.tail(25).copy()
        for c in ["actual_vol", "lstm_vol", "ewma_vol"]:
            recent[c] = recent[c].map(lambda x: f"{x:.2%}")
        recent = recent.rename(
            columns={
                "date": "Date",
                "actual_vol": "Realized volatility",
                "lstm_vol": "LSTM forecast",
                "ewma_vol": "EWMA forecast",
            }
        )
        recent["Date"] = pd.to_datetime(recent["Date"]).dt.date.astype(str)
        progress(1.0, desc="Done")
        return (
            summary,
            price_figure(market),
            forecast_figure(result.predictions),
            metrics,
            loss_figure(result.training.train_loss, result.training.val_loss),
            recent,
            gr.Markdown(visible=False),
        )
    except Exception as exc:  # noqa: BLE001
        message = (
            "### Analysis could not run\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Check the Space internet connection and try again. The app uses Yahoo Finance first and "
            "the original Hugging Face ABNB dataset as a fallback; it never substitutes simulated market observations."
        )
        return (
            "",
            None,
            None,
            pd.DataFrame(),
            None,
            pd.DataFrame(),
            gr.Markdown(value=message, visible=True),
        )


with gr.Blocks(title="ABNB Deep Learning Volatility Lab") as demo:
    gr.Markdown(
        "# ABNB Deep Learning Volatility Lab\n"
        "Forecast Airbnb market risk with an LSTM and compare it against a transparent EWMA baseline."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        lookback = gr.Slider(10, 60, value=20, step=5, label="Lookback window (trading days)")
        horizon = gr.Slider(2, 10, value=5, step=1, label="Forecast horizon (trading days)")
        epochs = gr.Slider(10, 50, value=25, step=5, label="Maximum LSTM epochs")
    run_btn = gr.Button("Run deep-learning analysis", variant="primary")
    error_box = gr.Markdown(visible=False)

    summary = gr.Markdown("Select parameters and run the analysis. First execution may take about a minute on CPU.")
    price_plot = gr.Plot(label="Market overview")
    forecast_plot = gr.Plot(label="Volatility forecast")
    metrics_table = gr.Dataframe(label="Out-of-sample metrics", interactive=False)
    loss_plot = gr.Plot(label="Training diagnostics")
    recent_table = gr.Dataframe(label="Most recent held-out predictions", interactive=False)

    gr.Markdown(
        "**Data attribution.** Full-history data are requested from Yahoo Finance through `yfinance`. "
        "If that source is unavailable, the app attempts the original "
        "[`nateraw/airbnb-stock-price-2`](https://huggingface.co/datasets/nateraw/airbnb-stock-price-2) "
        "Hugging Face dataset. All returns, volatility targets, EWMA values, and model forecasts are derived variables; "
        "they are not presented as source observations."
    )

    run_btn.click(
        analyze,
        inputs=[lookback, horizon, epochs],
        outputs=[summary, price_plot, forecast_plot, metrics_table, loss_plot, recent_table, error_box],
    )

if __name__ == "__main__":
    demo.launch()
