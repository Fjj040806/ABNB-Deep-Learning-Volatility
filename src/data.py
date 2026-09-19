from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests
import yfinance as yf

HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=nateraw%2Fairbnb-stock-price-2&config=default&split=train"
)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output across single- and multi-index versions."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) or (ticker, field).
        level0 = set(map(str, df.columns.get_level_values(0)))
        wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if level0 & wanted:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def load_from_yahoo(
    ticker: str = "ABNB",
    start: str = "2020-12-10",
    end: str | None = None,
) -> pd.DataFrame:
    """Download full daily OHLCV history from Yahoo Finance through yfinance."""
    end = end or date.today().isoformat()
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no ABNB rows.")

    df = _flatten_yfinance_columns(df).copy()
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    rename = {date_col: "date", "Adj Close": "adj_close"}
    df = df.rename(columns=rename)

    # Prefer adjusted close for returns; fall back to unadjusted close.
    if "adj_close" not in df.columns and "Close" in df.columns:
        df["adj_close"] = df["Close"]

    keep = ["date", "Open", "High", "Low", "Close", "adj_close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"Yahoo Finance response is missing columns: {missing}")

    df = df[keep].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if len(df) < 300:
        raise RuntimeError(f"Only {len(df)} usable Yahoo Finance rows were returned.")
    df.attrs["source"] = "Yahoo Finance via yfinance"
    return df


def load_from_huggingface() -> pd.DataFrame:
    """Fallback to the original course dataset on Hugging Face."""
    rows: list[dict] = []
    offset = 0
    page_size = 100
    for _ in range(20):
        response = requests.get(
            f"{HF_ROWS_URL}&offset={offset}&length={page_size}", timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        batch = [item.get("row", item) for item in payload.get("rows", [])]
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += len(batch)

    if not rows:
        raise RuntimeError("Hugging Face fallback returned no rows.")

    df = pd.DataFrame(rows)
    normalized = {c.lower().replace(".", "").replace("_", ""): c for c in df.columns}

    def col(*names: str) -> str:
        for n in names:
            key = n.lower().replace(".", "").replace("_", "")
            if key in normalized:
                return normalized[key]
        raise KeyError(names)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[col("Date", "date")], errors="coerce"),
            "open": pd.to_numeric(df[col("Open", "open")], errors="coerce"),
            "high": pd.to_numeric(df[col("High", "high")], errors="coerce"),
            "low": pd.to_numeric(df[col("Low", "low")], errors="coerce"),
            "adj_close": pd.to_numeric(
                df[col("Adj.Close", "close_last", "close")], errors="coerce"
            ),
            "volume": pd.to_numeric(
                df[col("Volume", "volume")].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ),
        }
    ).dropna()
    out["close"] = out["adj_close"]
    out = out[["date", "open", "high", "low", "close", "adj_close", "volume"]]
    out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    out.attrs["source"] = "Hugging Face nateraw/airbnb-stock-price-2 (fallback)"
    return out


def load_market_data() -> tuple[pd.DataFrame, str]:
    """Use full Yahoo history when available, otherwise the original HF dataset."""
    errors: list[str] = []
    try:
        df = load_from_yahoo()
        return df, str(df.attrs.get("source", "Yahoo Finance via yfinance"))
    except Exception as exc:  # noqa: BLE001 - intentional source fallback
        errors.append(f"Yahoo: {exc}")

    try:
        df = load_from_huggingface()
        return df, str(df.attrs.get("source", "Hugging Face fallback"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Hugging Face: {exc}")

    raise RuntimeError("Could not load market data. " + " | ".join(errors))
