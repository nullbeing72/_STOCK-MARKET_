from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


class RateLimiter:
    def __init__(self, min_interval_seconds: float = 1.0):
        self.min_interval_seconds = min_interval_seconds
        self._last_call = 0.0

    def wait(self) -> None:
        delta = time.time() - self._last_call
        if delta < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - delta)
        self._last_call = time.time()


@dataclass
class FetchResult:
    provider: str
    data: Optional[pd.DataFrame]
    warnings: list[str]


class MarketDataProviderChain:
    """
    Provider chain: Yahoo -> NSE API -> AlphaVantage -> TwelveData.

    AlphaVantage/TwelveData are activated only when their API keys are present
    in environment variables, keeping local research runs offline-friendly.
    """

    def __init__(self, yahoo_fetcher: Callable[[str, int], Optional[pd.DataFrame]]):
        self.yahoo_fetcher = yahoo_fetcher
        self.limiter = RateLimiter(float(os.getenv("MARKET_DATA_MIN_INTERVAL", "1.0")))

    def fetch_ohlcv(self, ticker: str, period_days: int) -> FetchResult:
        warnings: list[str] = []
        for name, fn in [
            ("yahoo", self._fetch_yahoo),
            ("nse", self._fetch_nse),
            ("alphavantage", self._fetch_alphavantage),
            ("twelvedata", self._fetch_twelvedata),
        ]:
            self.limiter.wait()
            try:
                df = fn(ticker, period_days)
                ok, issue = validate_ohlcv_frame(df)
                if ok:
                    return FetchResult(name, repair_ohlcv_frame(df), warnings)
                warnings.append(f"{name}: {issue}")
            except Exception as exc:
                warnings.append(f"{name}: {type(exc).__name__}: {exc}")
        return FetchResult("none", None, warnings)

    def _fetch_yahoo(self, ticker: str, period_days: int) -> Optional[pd.DataFrame]:
        return self.yahoo_fetcher(ticker, period_days)

    def _fetch_nse(self, ticker: str, period_days: int) -> Optional[pd.DataFrame]:
        # NSE has no stable unauthenticated historical equity API. This hook is
        # intentionally conservative and returns None unless a local bridge URL
        # is configured by deployment.
        base = os.getenv("NSE_HISTORY_ENDPOINT")
        if not base:
            return None
        symbol = ticker.replace(".NS", "")
        url = f"{base.rstrip('/')}?symbol={urllib.parse.quote(symbol)}&days={period_days}"
        return _json_ohlcv(url)

    def _fetch_alphavantage(self, ticker: str, period_days: int) -> Optional[pd.DataFrame]:
        key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not key:
            return None
        symbol = ticker.replace(".NS", ".BSE")
        params = urllib.parse.urlencode({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": key,
            "outputsize": "full",
        })
        with urllib.request.urlopen(f"https://www.alphavantage.co/query?{params}", timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("Time Series (Daily)", {})
        if not rows:
            return None
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "6. volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
        return df.sort_index().tail(period_days + 120)

    def _fetch_twelvedata(self, ticker: str, period_days: int) -> Optional[pd.DataFrame]:
        key = os.getenv("TWELVEDATA_API_KEY")
        if not key:
            return None
        symbol = ticker.replace(".NS", "")
        params = urllib.parse.urlencode({
            "symbol": symbol,
            "exchange": "NSE",
            "interval": "1day",
            "outputsize": min(period_days + 120, 5000),
            "apikey": key,
        })
        with urllib.request.urlopen(f"https://api.twelvedata.com/time_series?{params}", timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        values = payload.get("values", [])
        if not values:
            return None
        df = pd.DataFrame(values)
        df["Date"] = pd.to_datetime(df["datetime"])
        df = df.set_index("Date").rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
        return df.sort_index()


def _json_ohlcv(url: str) -> Optional[pd.DataFrame]:
    with urllib.request.urlopen(url, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data", payload if isinstance(payload, list) else [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    date_col = "Date" if "Date" in df else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    rename = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in needed if c in df.columns]].apply(pd.to_numeric, errors="coerce")


def validate_ohlcv_frame(df: Optional[pd.DataFrame]) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "empty data"
    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"
    if df["Close"].dropna().empty:
        return False, "no valid close prices"
    if len(df) < 20:
        return False, "insufficient rows"
    return True, ""


def repair_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    out = out[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    out = out.asfreq("B").ffill()
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)
    return out.dropna(subset=["Close"])


class ResearchStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def init_schema(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                create table if not exists market_data (
                    ticker text not null, date text not null,
                    open real, high real, low real, close real, volume real,
                    source text, version integer default 1,
                    created_at text default current_timestamp,
                    primary key (ticker, date)
                );
                create table if not exists features (
                    ticker text not null, date text not null,
                    payload_json text not null, version integer default 1,
                    created_at text default current_timestamp,
                    primary key (ticker, date)
                );
                create table if not exists predictions (
                    ticker text not null, date text not null,
                    yest_pred_mean real, today_pred_mean real, tomorrow_pred_mean real,
                    actual_price real, error1_pct real, error2_pct real,
                    payload_json text, version integer default 1,
                    created_at text default current_timestamp,
                    primary key (ticker, date)
                );
                create table if not exists model_metrics (
                    ticker text not null, run_ts text not null,
                    training_mode text, best_val_loss real, mape real,
                    directional_accuracy real, drift_json text,
                    primary key (ticker, run_ts)
                );
                create table if not exists backtest_results (
                    name text not null, run_ts text not null,
                    payload_json text not null,
                    primary key (name, run_ts)
                );
                create table if not exists portfolio_allocations (
                    run_date text not null, method text not null,
                    ticker text not null, weight real not null,
                    expected_return real, risk real,
                    primary key (run_date, method, ticker)
                );
                create table if not exists ranking_history (
                    run_date text not null, ticker text not null,
                    rank integer, side text, score real, confidence real,
                    payload_json text,
                    primary key (run_date, ticker, side)
                );
                """
            )

    def upsert_market_data(self, ticker: str, df: pd.DataFrame, source: str = "unknown") -> None:
        if df is None or df.empty:
            return
        rows = [
            (
                ticker, idx.date().isoformat() if hasattr(idx, "date") else str(idx),
                _safe_float(row.get("Open")), _safe_float(row.get("High")),
                _safe_float(row.get("Low")), _safe_float(row.get("Close")),
                _safe_float(row.get("Volume")), source,
            )
            for idx, row in df.iterrows()
        ]
        with self.connect() as con:
            con.executemany(
                """
                insert into market_data(ticker,date,open,high,low,close,volume,source)
                values(?,?,?,?,?,?,?,?)
                on conflict(ticker,date) do update set
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume, source=excluded.source,
                    version=market_data.version+1
                """,
                rows,
            )

    def upsert_feature_row(self, ticker: str, date_key: str, features: dict) -> None:
        payload = json.dumps(features, default=str, sort_keys=True)
        with self.connect() as con:
            con.execute(
                """
                insert into features(ticker,date,payload_json) values(?,?,?)
                on conflict(ticker,date) do update set
                    payload_json=excluded.payload_json, version=features.version+1
                """,
                (ticker, date_key, payload),
            )

    def upsert_prediction(self, ticker: str, date_key: str, row: dict) -> None:
        payload = json.dumps(row, default=str, sort_keys=True)
        with self.connect() as con:
            con.execute(
                """
                insert into predictions(
                    ticker,date,yest_pred_mean,today_pred_mean,tomorrow_pred_mean,
                    actual_price,error1_pct,error2_pct,payload_json
                ) values(?,?,?,?,?,?,?,?,?)
                on conflict(ticker,date) do update set
                    yest_pred_mean=excluded.yest_pred_mean,
                    today_pred_mean=excluded.today_pred_mean,
                    tomorrow_pred_mean=excluded.tomorrow_pred_mean,
                    actual_price=excluded.actual_price,
                    error1_pct=excluded.error1_pct,
                    error2_pct=excluded.error2_pct,
                    payload_json=excluded.payload_json,
                    version=predictions.version+1
                """,
                (
                    ticker, date_key,
                    _safe_float(row.get("Yest_Pred_Mean")),
                    _safe_float(row.get("Today_Pred_Mean")),
                    _safe_float(row.get("Tomorrow_Pred_Mean")),
                    _safe_float(row.get("Actual_Price")),
                    _safe_float(row.get("Error1_Pct")),
                    _safe_float(row.get("Error2_Pct")),
                    payload,
                ),
            )

    def insert_model_metrics(self, ticker: str, metrics: dict) -> None:
        with self.connect() as con:
            con.execute(
                """
                insert or replace into model_metrics(
                    ticker,run_ts,training_mode,best_val_loss,mape,directional_accuracy,drift_json
                ) values(?,?,?,?,?,?,?)
                """,
                (
                    ticker, metrics.get("run_ts", datetime.now().isoformat()),
                    metrics.get("training_mode"),
                    _safe_float(metrics.get("best_val_loss")),
                    _safe_float(metrics.get("mape")),
                    _safe_float(metrics.get("directional_accuracy")),
                    json.dumps(metrics.get("drift", {}), default=str, sort_keys=True),
                ),
            )

    def backup(self, backup_path: Path) -> Path:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as src, sqlite3.connect(backup_path) as dst:
            src.backup(dst)
        return backup_path


def detect_market_regime(market_df: pd.DataFrame, vix: Optional[pd.Series] = None) -> dict:
    if market_df is None or market_df.empty or "Close" not in market_df:
        return {"regime": "Unknown", "confidence": 0.0, "reason": "missing market data"}
    close = market_df["Close"].dropna()
    if len(close) < 80:
        return {"regime": "Unknown", "confidence": 0.0, "reason": "insufficient market data"}
    ret = close.pct_change().dropna()
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(min(200, len(close))).mean().iloc[-1]
    vol20 = ret.rolling(20).std().iloc[-1] * math.sqrt(252)
    peak = close.cummax()
    drawdown = float((close.iloc[-1] / peak.iloc[-1] - 1.0) * 100.0)
    trend = float((close.iloc[-1] / close.iloc[-60] - 1.0) * 100.0)
    vix_latest = _safe_float(vix.dropna().iloc[-1]) if vix is not None and not vix.dropna().empty else None

    if (vix_latest and vix_latest >= 22) or vol20 >= 0.28:
        regime = "High Volatility Market"
    elif close.iloc[-1] > sma50 > sma200 and trend > 3:
        regime = "Bull Market"
    elif close.iloc[-1] < sma50 < sma200 or drawdown < -12:
        regime = "Bear Market"
    else:
        regime = "Sideways Market"
    confidence = min(1.0, max(abs(trend) / 12.0, abs(drawdown) / 25.0, vol20 / 0.35))
    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "trend_60d_pct": round(trend, 3),
        "drawdown_pct": round(drawdown, 3),
        "vol_20d_ann": round(float(vol20), 4),
        "vix": vix_latest,
    }


def _histogram(values: np.ndarray, bins: int = 10) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.full(bins, 1.0 / bins)
    counts, _ = np.histogram(values, bins=bins)
    probs = counts.astype(float) + 1e-6
    return probs / probs.sum()


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    exp = _histogram(np.asarray(expected, dtype=float), bins)
    act = _histogram(np.asarray(actual, dtype=float), bins)
    return float(np.sum((act - exp) * np.log(act / exp)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    x = np.sort(np.asarray(expected, dtype=float)[np.isfinite(expected)])
    y = np.sort(np.asarray(actual, dtype=float)[np.isfinite(actual)])
    if len(x) == 0 or len(y) == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def kl_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    exp = _histogram(np.asarray(expected, dtype=float), bins)
    act = _histogram(np.asarray(actual, dtype=float), bins)
    return float(np.sum(act * np.log(act / exp)))


def wasserstein_distance_1d(expected: np.ndarray, actual: np.ndarray) -> float:
    x = np.sort(np.asarray(expected, dtype=float)[np.isfinite(expected)])
    y = np.sort(np.asarray(actual, dtype=float)[np.isfinite(actual)])
    if len(x) == 0 or len(y) == 0:
        return 0.0
    n = max(len(x), len(y))
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def compute_drift_report(
    frame: pd.DataFrame,
    columns: list[str],
    baseline_window: int = 120,
    recent_window: int = 30,
) -> dict:
    report = {"columns": {}, "drift_detected": False}
    if frame is None or frame.empty:
        return report
    for col in columns:
        if col not in frame:
            continue
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(series) < baseline_window + min(10, recent_window):
            continue
        baseline = series.iloc[-baseline_window:-recent_window].to_numpy()
        recent = series.iloc[-recent_window:].to_numpy()
        metrics = {
            "psi": round(population_stability_index(baseline, recent), 6),
            "ks": round(ks_statistic(baseline, recent), 6),
            "kl": round(kl_divergence(baseline, recent), 6),
            "wasserstein": round(wasserstein_distance_1d(baseline, recent), 6),
        }
        metrics["drift"] = metrics["psi"] >= 0.25 or metrics["ks"] >= 0.35
        report["columns"][col] = metrics
        report["drift_detected"] = report["drift_detected"] or metrics["drift"]
    return report


def optimize_portfolio(price_map: dict[str, pd.DataFrame], method: str = "mean_variance") -> pd.DataFrame:
    returns = {}
    for ticker, df in price_map.items():
        if df is not None and "Close" in df:
            returns[ticker] = df["Close"].pct_change().dropna()
    if not returns:
        return pd.DataFrame(columns=["ticker", "weight", "expected_return", "risk"])
    mat = pd.DataFrame(returns).dropna(how="all").fillna(0.0)
    mu = mat.mean() * 252
    cov = mat.cov() * 252
    tickers = list(mat.columns)
    if method == "risk_parity":
        vol = np.sqrt(np.diag(cov.values))
        inv = 1.0 / np.where(vol <= 1e-9, 1.0, vol)
        weights = inv / inv.sum()
    else:
        inv_cov = np.linalg.pinv(cov.values + np.eye(len(tickers)) * 1e-6)
        raw = inv_cov @ mu.values
        raw = np.clip(raw, 0, None)
        weights = raw / raw.sum() if raw.sum() > 1e-12 else np.full(len(tickers), 1 / len(tickers))
    rows = []
    for ticker, weight in zip(tickers, weights):
        rows.append({
            "ticker": ticker,
            "weight": round(float(weight), 6),
            "expected_return": round(float(mu[ticker]), 6),
            "risk": round(float(math.sqrt(max(cov.loc[ticker, ticker], 0))), 6),
        })
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def rank_opportunities(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["latest", "pct_change", "e2", "dir_acc"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    expected = df.get("pct_change", pd.Series(0, index=df.index)).fillna(0)
    accuracy = df.get("dir_acc", pd.Series(50, index=df.index)).fillna(50)
    error = df.get("e2", pd.Series(5, index=df.index)).fillna(5)
    df["score"] = expected + (accuracy - 50) / 10 - error / 2
    df["confidence"] = np.clip((accuracy / 100) * (1 / (1 + error / 10)), 0, 1)
    df["side"] = np.where(df["score"] >= 0, "long", "short")
    return df.sort_values("score", ascending=False).reset_index(drop=True)


SECTOR_MAP = {
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "KOTAKBANK.NS": "Banking",
    "SBIN.NS": "Banking", "AXISBANK.NS": "Banking", "TCS.NS": "IT",
    "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto",
    "TATASTEEL.NS": "Metal", "JSWSTEEL.NS": "Metal", "HINDALCO.NS": "Metal",
    "DLF.NS": "Realty", "GODREJPROP.NS": "Realty", "LT.NS": "Infrastructure",
}


def sector_rotation(rows: list[dict]) -> pd.DataFrame:
    ranked = rank_opportunities(rows)
    if ranked.empty or "ticker" not in ranked:
        return pd.DataFrame()
    ranked["sector"] = ranked["ticker"].map(SECTOR_MAP).fillna("Other")
    grouped = ranked.groupby("sector").agg(
        avg_score=("score", "mean"),
        avg_confidence=("confidence", "mean"),
        members=("ticker", "count"),
    ).reset_index()
    grouped["signal"] = np.where(grouped["avg_score"] > 1, "Overweight",
                         np.where(grouped["avg_score"] < -1, "Underweight", "Neutral"))
    return grouped.sort_values("avg_score", ascending=False)


def simple_backtest(df: pd.DataFrame, signal_col: str = "Today_Pred_Mean", cost_bps: float = 10.0) -> dict:
    if df is None or df.empty or "Close" not in df or signal_col not in df:
        return {}
    close = pd.to_numeric(df["Close"], errors="coerce")
    signal = pd.to_numeric(df[signal_col], errors="coerce")
    pos = np.sign(signal.shift(1) - close.shift(1)).fillna(0)
    ret = close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(0)
    strat = pos * ret - turnover * (cost_bps / 10000.0)
    equity = (1 + strat).cumprod()
    if len(equity) < 2:
        return {}
    years = max(len(equity) / 252, 1 / 252)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    vol = float(strat.std() * math.sqrt(252))
    sharpe = float((strat.mean() * 252) / vol) if vol > 1e-12 else 0.0
    downside = strat[strat < 0].std() * math.sqrt(252)
    sortino = float((strat.mean() * 252) / downside) if downside and downside > 1e-12 else 0.0
    dd = equity / equity.cummax() - 1
    wins = float((strat > 0).mean() * 100)
    return {
        "cagr": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(float(dd.min()) * 100, 3),
        "win_rate": round(wins, 3),
        "calmar": round((cagr / abs(float(dd.min()))) if dd.min() < 0 else 0.0, 3),
    }


class ExperimentTracker:
    def __init__(self, tracking_uri: Optional[str] = None):
        self.enabled = False
        try:
            import mlflow  # type: ignore

            self.mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.enabled = True
        except Exception:
            self.mlflow = None

    def log_run(self, name: str, params: dict, metrics: dict, tags: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        with self.mlflow.start_run(run_name=name):
            if tags:
                self.mlflow.set_tags(tags)
            self.mlflow.log_params({k: v for k, v in params.items() if v is not None})
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _safe_float(v) is not None})


class NotificationManager:
    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def notify(self, subject: str, body: str) -> None:
        if self.telegram_token and self.telegram_chat_id:
            params = urllib.parse.urlencode({
                "chat_id": self.telegram_chat_id,
                "text": f"{subject}\n{body}",
            }).encode("utf-8")
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            try:
                urllib.request.urlopen(url, data=params, timeout=10).read()
            except Exception:
                pass
