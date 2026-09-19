from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.data import load_market_data
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ABNB LSTM volatility model.")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    market, source = load_market_data()
    result = run_pipeline(market, args.lookback, args.horizon, args.epochs)

    result.predictions.to_csv(args.output / "predictions.csv", index=False)
    result.metrics.to_csv(args.output / "metrics.csv", index=False)
    joblib.dump(result.scaler, args.output / "scaler.pkl")
    torch.save(result.training.model.state_dict(), args.output / "model.pt")
    (args.output / "run_summary.json").write_text(
        json.dumps(
            {
                "source": source,
                "rows": len(market),
                "lookback": args.lookback,
                "horizon": args.horizon,
                "epochs_requested": args.epochs,
                "best_epoch": result.training.best_epoch,
                "next_annualized_volatility_forecast": result.next_forecast,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.metrics.to_string(index=False))
    print(f"Next {args.horizon}-day annualized volatility forecast: {result.next_forecast:.2%}")


if __name__ == "__main__":
    main()
