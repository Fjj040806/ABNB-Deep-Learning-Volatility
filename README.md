---
title: ABNB Deep Learning Volatility Lab
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# ABNB Deep Learning Volatility Lab

An end-to-end financial machine-learning project that forecasts **Airbnb (ABNB) short-horizon realized volatility** with an LSTM and evaluates it against a transparent **EWMA risk-model baseline**.

This repository is designed to work both as a normal **GitHub project** and as a **Hugging Face Gradio Space**.

## Why this project

Predicting the exact next stock price is noisy and easy to overstate. This project instead asks a more defensible FinTech question:

> Can a sequence model improve short-horizon volatility forecasting relative to a traditional exponentially weighted risk model?

The answer is determined empirically on a chronological held-out test set. The code does **not** assume that deep learning must win.

## What it does

- Downloads ABNB daily OHLCV history from Yahoo Finance through `yfinance`.
- Falls back to the original `nateraw/airbnb-stock-price-2` Hugging Face dataset if needed.
- Engineers leakage-safe market features using only information available at time `t`.
- Defines the target as annualized realized volatility over the next 2–10 trading days.
- Trains a two-layer PyTorch LSTM with chronological train / validation / test splits.
- Benchmarks the LSTM against RiskMetrics-style EWMA volatility.
- Reports RMSE, MAE, R², and forecast correlation on the held-out test period.
- Visualizes ABNB price/volume, realized vs. forecast volatility, training diagnostics, and recent predictions.
- Produces a latest short-horizon risk forecast while clearly labeling it as experimental and not financial advice.

## Model design

For day `t`, the model consumes a trailing sequence of market features including:

- log return and absolute return
- intraday high-low range
- close-to-open return
- log volume change
- 5-day momentum
- 5-day rolling volatility
- 20-day rolling volatility

The default target is:

`annualized standard deviation of log returns over t+1 ... t+5`

The default LSTM receives the prior 20 trading days of features.

## Leakage controls

Financial ML projects are especially vulnerable to look-ahead bias. This project uses:

1. A strictly chronological 70% / 15% / 15% train-validation-test split.
2. A `StandardScaler` fit only on the training period.
3. Rolling input features calculated only from current and past observations.
4. Forward realized volatility used only as the supervised target.
5. Final performance reported only on the held-out test period.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Train from the command line

```bash
python train.py --lookback 20 --horizon 5 --epochs 25
```

Outputs are written to `artifacts/`:

- `predictions.csv`
- `metrics.csv`
- `model.pt`
- `scaler.pkl`
- `run_summary.json`

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose **Gradio** as the SDK.
3. Upload all files in this repository while preserving the folder structure.
4. Commit the files.
5. Hugging Face installs `requirements.txt` and launches `app.py` automatically.

No API key or environment variable is required.

## Suggested GitHub repository structure

```text
abnb-volatility-lab/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── pipeline.py
├── tests/
│   └── test_features.py
└── artifacts/
    └── .gitkeep
```

## Interpretation

A deep-learning model should not be judged only by whether it looks sophisticated. If EWMA beats the LSTM out of sample, that is a valid result and illustrates the difficulty of extracting stable predictive structure from a single stock's daily history. A stronger future version could train across a panel of equities, incorporate macro/market variables such as VIX and NASDAQ returns, or add text-derived features from earnings/news.

## Data attribution

Primary market data are requested from Yahoo Finance through `yfinance`. The application also contains a fallback loader for the original Hugging Face dataset used in the earlier visualization project: `nateraw/airbnb-stock-price-2`.

Returns, rolling volatility, forward realized volatility, EWMA estimates, and LSTM predictions are **derived variables** and are never represented as source observations.

## Disclaimer

This project is for educational, research, and portfolio demonstration purposes only. It does not constitute financial advice, an investment recommendation, or a trading signal.
