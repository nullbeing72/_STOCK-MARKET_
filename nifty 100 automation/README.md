# NIFTY 100 Automation

This folder contains a NIFTY 100 stock-tracking pipeline and a Streamlit dashboard.

## Files

- `nifty100tracker.py` runs the data collection, feature engineering, model training/fine-tuning, prediction logging, Excel export, and model metadata updates.
- `dashboard_streamlit.py` reads the generated Excel files and model metadata, then renders the research dashboard.
- `data.py` contains Yahoo Finance fetchers, technical/quant feature engineering, macro/commodity helpers, sequence building, and risk calculations.
- `model.py` contains the PyTorch hybrid CNN/BiLSTM/attention model, training helpers, Monte Carlo forecasting, calibration, and metrics.
- `platform_calendar.py` contains the dynamic NSE exchange calendar loader with fallback holidays and special-session support.
- `research_platform.py` contains the provider chain, SQLite research store, drift metrics, market regime detection, ranking, sector rotation, portfolio optimization, backtesting, optional MLflow logging, and notifications.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
cd "nifty 100 automation"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Data

Run the tracker once for a single ticker:

```bash
python nifty100tracker.py --once --force --ticker RELIANCE.NS
```

Run the tracker once for all configured tickers:

```bash
python nifty100tracker.py --once --force
```

The tracker writes generated files into:

- `nifty100_data/` for per-ticker Excel workbooks.
- `nifty100_models/` for PyTorch model files and metadata JSON.
- `logs/` for daily tracker logs.
- `tracker_state.json` for duplicate-run tracking.
- `nifty100_research.db` for SQLite market data, features, predictions, model metrics, rankings, backtests, and allocations.

## Open Dashboard

From inside this folder:

```bash
streamlit run dashboard_streamlit.py -- --data-dir ./nifty100_data --models-dir ./nifty100_models
```

The dashboard expects the tracker to have created at least one Excel workbook in `nifty100_data/`.

## Useful Commands

Show model metadata for one ticker:

```bash
python nifty100tracker.py --show-meta RELIANCE.NS
```

Show active exchange calendar status:

```bash
python nifty100tracker.py --calendar-status
```

Force a full retrain:

```bash
python nifty100tracker.py --once --force --retrain --ticker RELIANCE.NS
```

Backfill recent trading days:

```bash
python nifty100tracker.py --backfill --days 30 --ticker RELIANCE.NS
```

## Notes

- Yahoo Finance availability and rate limits can affect tracker runs.
- The dashboard is read-only; it visualizes generated Excel and metadata outputs.
- Paid/limited fallback providers activate via `ALPHAVANTAGE_API_KEY`, `TWELVEDATA_API_KEY`, or `NSE_HISTORY_ENDPOINT`.
- Telegram notifications activate via `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
- MLflow logging activates automatically when `mlflow` is installed; set `MLFLOW_TRACKING_URI` to direct runs to a tracking server.
- The code compiles with `python3 -m py_compile`, but this workspace currently does not have the runtime packages installed globally.
