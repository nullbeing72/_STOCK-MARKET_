"""
  - _normalize_columns: explicit level-1 extraction for single-ticker MultiIndex
  - _holt_forecast: proper trend initialisation (global slope, not one-step)
  - fetch_commodities: exported for use in main.py
  - add_quant_features: bond column rename documented
  - compute_fcm_risk: Fuzzy C-Means risk segregation, year-wise fuzzy values + mean
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ── Feature column registry ───────────────────────────────────────────────────
PRICE_FEATURE_COLS: list[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50", "MACD", "MACD_Signal",
    "RSI", "ROC_10", "Stoch_K",
    "BB_Upper", "BB_Lower", "ATR",
    "OBV_norm", "RVI",
]

FACTOR_FEATURE_COLS: list[str] = [
    "Beta_60", "Alpha_60", "Momentum_12_1", "Sharpe_60", "Vol_Ratio", "Mkt_Return",
    "Bond_5Y", "Bond_10Y",
    "Gold", "Silver", "Copper", "Aluminium", "Brent_Oil",
    "Buffett_Proxy",
    "USDINR", "DXY", "India_VIX", "FII_Flow", "DII_Flow",
    "Repo_Rate", "CPI_Inflation", "PMI", "GDP_Growth",
    "PE_Ratio", "PB_Ratio", "EPS_Growth", "ROE", "ROCE",
    "Debt_To_Equity", "Institutional_Ownership",
]

FEATURE_COLS      = PRICE_FEATURE_COLS + FACTOR_FEATURE_COLS
CLOSE_IDX         = PRICE_FEATURE_COLS.index("Close")   # 3
N_PRICE_FEATURES  = len(PRICE_FEATURE_COLS)
N_FACTOR_FEATURES = len(FACTOR_FEATURE_COLS)

_GDP_ETFS: dict[str, str] = {
    "US":     "SPY",
    "India":  "INDA",
    "China":  "FXI",
    "EU":     "EZU",
    "Japan":  "EWJ",
    "UK":     "EWU",
    "Brazil": "EWZ",
}

_COMMODITY_TICKERS: dict[str, str] = {
    "Gold":      "GC=F",
    "Silver":    "SI=F",
    "Copper":    "HG=F",
    "Aluminium": "ALI=F",
    "Brent_Oil": "BZ=F",
}

_MACRO_TICKERS: dict[str, str] = {
    "USDINR": "INR=X",
    "DXY": "DX-Y.NYB",
    "India_VIX": "^INDIAVIX",
}

_STATIC_MACRO_DEFAULTS: dict[str, float] = {
    "FII_Flow": 0.0,
    "DII_Flow": 0.0,
    "Repo_Rate": 6.50,
    "CPI_Inflation": 4.75,
    "PMI": 56.0,
    "GDP_Growth": 6.8,
}

# ── In-memory cache ───────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, object]] = {}
_TTL   = 3600.0

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < _TTL:
        return entry[1]
    return None

def _cache_set(key: str, value):
    _CACHE[key] = (time.time(), value)
    return value

# ── Robust yfinance column normalizer ─────────────────────────────────────────
def _normalize_columns(df: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Flatten MultiIndex columns produced by yf.download().
    When tickers is None  → returns OHLCV DataFrame for a single ticker.
    When tickers is given → returns DataFrame with those tickers as columns
                            (each column = Close price for that ticker).
    """
    if df is None or df.empty:
        return df

    if not isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        return df

    level_0 = list(df.columns.get_level_values(0).unique())
    level_1 = list(df.columns.get_level_values(1).unique())

    if tickers is not None:
        # Multi-ticker: caller wants per-ticker Close prices.
        close_field = (
            "Close"     if "Close"     in level_0 else
            "Adj Close" if "Adj Close" in level_0 else None
        )
        if close_field:
            try:
                result = df[close_field]
                # result is a DataFrame with tickers as columns
                return result
            except Exception:
                pass
        # Fallback: flatten and pick ticker-named columns
        df.columns = [
            str(col[1]) if col[1] else str(col[0])
            for col in df.columns
        ]
        return df

    # Single-ticker multi-index — FIX: use level-1 (ticker symbol) to drop
    # the ticker dimension, leaving only field names as the column index.
    try:
        # level_1 contains the single ticker name; drop it.
        if len(level_1) == 1:
            df = df.droplevel(1, axis=1)
        else:
            # Unexpected: multiple tickers but tickers=None. Use level-0.
            df.columns = df.columns.get_level_values(0)
    except Exception:
        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]

    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    return df

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_data(ticker: str, period_days: int) -> Optional[pd.DataFrame]:
    key = f"price:{ticker}:{period_days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        end   = datetime.today()
        start = end - timedelta(days=period_days + 100)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = _normalize_columns(df)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.asfreq("B").ffill()
        return _cache_set(key, df)
    except Exception as e:
        print(f"[fetch_data] {e}")
        return None

def fetch_market_data(period_days: int, ticker: str = "SPY") -> Optional[pd.DataFrame]:
    key = f"market:{ticker}:{period_days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        end   = datetime.today()
        start = end - timedelta(days=period_days + 400)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = _normalize_columns(df)
        df = df.dropna().asfreq("B").ffill()
        return _cache_set(key, df)
    except Exception as e:
        print(f"[fetch_market_data] {e}")
        return None


# ── Global bond yield sources ─────────────────────────────────────────────────
_YAHOO_BOND_TICKERS: dict[str, str] = {
    "US_5Y":  "^FVX",
    "US_10Y": "^TNX",
}

_ETF_YIELD_PROXIES: dict[str, dict] = {
    "India_10Y":   {"etf": "INDA", "anchor": 6.80, "sensitivity": 0.30},
    "Germany_10Y": {"etf": "EZU",  "anchor": 2.45, "sensitivity": 0.70},
    "UK_10Y":      {"etf": "EWU",  "anchor": 4.40, "sensitivity": 0.85},
    "Japan_10Y":   {"etf": "EWJ",  "anchor": 1.10, "sensitivity": 0.10},
}

_YIELD_MIN, _YIELD_MAX = 0.01, 20.0

def _validate_yield_series(s: pd.Series) -> bool:
    if s is None or len(s) < 5:
        return False
    return _YIELD_MIN <= float(s.median()) <= _YIELD_MAX

def _fetch_yahoo_yield(ticker: str, start, end) -> Optional[pd.Series]:
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        raw = _normalize_columns(raw)
        for col in ("Close", "Adj Close"):
            if col in raw.columns:
                s = pd.to_numeric(raw[col], errors="coerce").dropna()
                if _validate_yield_series(s):
                    return s
        return None
    except Exception as e:
        print(f"[_fetch_yahoo_yield] {ticker}: {e}")
        return None

def _build_synthetic_yield(
    etf: str,
    anchor: float,
    sensitivity: float,
    us10y_series: Optional[pd.Series],
    start, end,
) -> pd.Series:
    try:
        raw = yf.download(etf, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            raise ValueError("empty ETF download")
        raw = _normalize_columns(raw)
        if "Close" not in raw.columns:
            raise ValueError("no Close column")

        prices    = raw["Close"].dropna()
        log_ret   = np.log(prices / prices.shift(1)).dropna()
        cum_drift = log_ret.rolling(20).mean().cumsum().fillna(0)

        rng = cum_drift.max() - cum_drift.min()
        normalised = ((cum_drift - cum_drift.mean()) / rng) if rng > 1e-6 else cum_drift * 0

        if us10y_series is not None:
            us_aligned = us10y_series.reindex(normalised.index, method="ffill").bfill()
            us_delta   = (us_aligned - float(us_aligned.mean())).fillna(0)
            yield_s    = anchor + normalised * 0.5 + us_delta * sensitivity
        else:
            yield_s = anchor + normalised * 0.5

        yield_s = yield_s.clip(_YIELD_MIN, _YIELD_MAX)
        yield_s = yield_s[(yield_s.index >= pd.Timestamp(start)) &
                          (yield_s.index <= pd.Timestamp(end))]
        if len(yield_s) < 5:
            raise ValueError("too short after filtering")

        return yield_s

    except Exception as e:
        print(f"[bond_proxy] {etf}: {e} — using flat anchor {anchor}%")
        bday_range = pd.bdate_range(start=start, end=end)
        rng_obj    = np.random.default_rng(42)
        noise      = rng_obj.normal(0, 0.04, len(bday_range))
        return pd.Series(
            np.clip(anchor + noise, _YIELD_MIN, _YIELD_MAX), index=bday_range
        )

def fetch_bond_yields(period_days: int) -> Optional[pd.DataFrame]:
    key = f"bonds_v4:{period_days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        end   = datetime.today()
        start = end - timedelta(days=period_days + 50)
        frames: dict[str, pd.Series] = {}

        for col_name, ticker in _YAHOO_BOND_TICKERS.items():
            s = _fetch_yahoo_yield(ticker, start, end)
            if s is not None:
                frames[col_name] = s
            else:
                print(f"[fetch_bond_yields] {col_name}: Yahoo failed")

        us10y = frames.get("US_10Y")
        for col_name, cfg in _ETF_YIELD_PROXIES.items():
            s = _build_synthetic_yield(
                etf=cfg["etf"], anchor=cfg["anchor"],
                sensitivity=cfg["sensitivity"],
                us10y_series=us10y, start=start, end=end,
            )
            frames[col_name] = s

        if not frames:
            return None

        bonds = pd.DataFrame(frames)
        bonds = bonds.sort_index().asfreq("B").ffill().bfill()
        bonds = bonds.dropna(how="all")
        return _cache_set(key, bonds)
    except Exception as e:
        print(f"[fetch_bond_yields] {e}")
        return None

def fetch_commodities(period_days: int) -> Optional[pd.DataFrame]:
    key = f"comm:{period_days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        end     = datetime.today()
        start   = end - timedelta(days=period_days + 50)
        tickers = list(_COMMODITY_TICKERS.values())
        names   = list(_COMMODITY_TICKERS.keys())

        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            return None

        close_df = _normalize_columns(raw, tickers=tickers)
        data = pd.DataFrame(index=close_df.index)

        for tkr, name in zip(tickers, names):
            if tkr in close_df.columns:
                data[name] = close_df[tkr]

        data = data.asfreq("B").ffill().dropna(how="all")
        return _cache_set(key, data)
    except Exception as e:
        print(f"[fetch_commodities] {e}")
        return None


def fetch_macro_indicators(period_days: int) -> Optional[pd.DataFrame]:
    key = f"macro_v1:{period_days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        end = datetime.today()
        start = end - timedelta(days=period_days + 100)
        frames: dict[str, pd.Series] = {}
        for name, ticker in _MACRO_TICKERS.items():
            try:
                raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                raw = _normalize_columns(raw)
                if raw is not None and not raw.empty and "Close" in raw.columns:
                    frames[name] = pd.to_numeric(raw["Close"], errors="coerce")
            except Exception as e:
                print(f"[fetch_macro_indicators] {name}: {e}")

        idx = None
        if frames:
            idx = pd.concat(frames.values(), axis=1).index
        else:
            idx = pd.bdate_range(start=start, end=end)

        macro = pd.DataFrame(index=idx)
        for name in _MACRO_TICKERS:
            macro[name] = frames.get(name, pd.Series(index=idx, dtype=float))
        for name, value in _STATIC_MACRO_DEFAULTS.items():
            macro[name] = value

        macro = macro.sort_index().asfreq("B").ffill().bfill()
        return _cache_set(key, macro)
    except Exception as e:
        print(f"[fetch_macro_indicators] {e}")
        return None


def fetch_fundamentals(ticker: str) -> dict[str, float]:
    key = f"fundamentals_v1:{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    metrics = {
        "PE_Ratio": np.nan,
        "PB_Ratio": np.nan,
        "EPS_Growth": np.nan,
        "ROE": np.nan,
        "ROCE": np.nan,
        "Debt_To_Equity": np.nan,
        "Institutional_Ownership": np.nan,
    }
    try:
        info = yf.Ticker(ticker).info or {}
        metrics["PE_Ratio"] = info.get("trailingPE", np.nan)
        metrics["PB_Ratio"] = info.get("priceToBook", np.nan)
        metrics["EPS_Growth"] = info.get("earningsQuarterlyGrowth", np.nan)
        metrics["ROE"] = info.get("returnOnEquity", np.nan)
        metrics["ROCE"] = info.get("returnOnCapital", np.nan)
        metrics["Debt_To_Equity"] = info.get("debtToEquity", np.nan)
        metrics["Institutional_Ownership"] = info.get("heldPercentInstitutions", np.nan)
    except Exception as e:
        print(f"[fetch_fundamentals] {ticker}: {e}")
    return _cache_set(key, metrics)

# ── Feature engineering ───────────────────────────────────────────────────────
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    ema12         = df["Close"].ewm(span=12, adjust=False).mean()
    ema26         = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]    = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta  = df["Close"].diff()
    gain   = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss   = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["RSI"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    df["ROC_10"]  = df["Close"].pct_change(10) * 100.0
    low14, high14 = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    df["Stoch_K"] = 100.0 * (df["Close"] - low14) / (high14 - low14).replace(0.0, np.nan)

    sma20, std20    = df["Close"].rolling(20).mean(), df["Close"].rolling(20).std()
    df["BB_Upper"]  = sma20 + 2.0 * std20
    df["BB_Lower"]  = sma20 - 2.0 * std20

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    direction     = np.sign(df["Close"].diff()).fillna(0.0)
    obv           = (direction * df["Volume"]).cumsum()
    obv_std       = obv.rolling(20).std().replace(0.0, np.nan)
    df["OBV_norm"] = (obv - obv.rolling(20).mean()) / obv_std

    d_vol         = (df["High"] - df["Low"]).diff()
    g_vol         = d_vol.where(d_vol > 0, 0.0).rolling(14).mean()
    l_vol         = (-d_vol.where(d_vol < 0, 0.0)).rolling(14).mean()
    df["RVI"]     = 100.0 - (100.0 / (1.0 + (g_vol / l_vol.replace(0, np.nan))))

    return df

def add_quant_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame] = None,
    comm_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    fundamentals: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    df        = df.copy()
    mkt_close = market_df["Close"].reindex(df.index, method="ffill")
    stock_ret = np.log(df["Close"] / df["Close"].shift(1))
    mkt_ret   = np.log(mkt_close  / mkt_close.shift(1))
    df["Mkt_Return"] = mkt_ret

    WINDOW = 60
    betas  = np.full(len(df), np.nan)
    alphas = np.full(len(df), np.nan)
    sr, mr = stock_ret.values, mkt_ret.values

    for i in range(WINDOW, len(df)):
        y, x = sr[i - WINDOW:i], mr[i - WINDOW:i]
        mask  = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 20:
            xf, yf = x[mask], y[mask]
            cov    = np.cov(xf, yf)[0, 1]
            var    = np.var(xf)
            if var > 1e-9:
                beta       = cov / var
                betas[i]   = beta
                alphas[i]  = (yf.mean() - beta * xf.mean()) * 252

    df["Beta_60"]       = betas
    df["Alpha_60"]      = alphas
    df["Momentum_12_1"] = (df["Close"].pct_change(252) - df["Close"].pct_change(21)) * 100.0
    df["Sharpe_60"]     = (
        stock_ret.rolling(WINDOW).mean()
        / stock_ret.rolling(WINDOW).std().replace(0, np.nan)
    ) * np.sqrt(252)
    df["Vol_Ratio"] = (
        stock_ret.rolling(10).std()
        / stock_ret.rolling(WINDOW).std().replace(0, np.nan)
    )

    # Bond yields — rename US_5Y/US_10Y → Bond_5Y/Bond_10Y for FEATURE_COLS
    _BOND_COL_MAP = {"US_5Y": "Bond_5Y", "US_10Y": "Bond_10Y"}
    if bond_df is not None and not bond_df.empty:
        aligned_bonds = bond_df.reindex(df.index, method="ffill")
        for src_col, dst_col in _BOND_COL_MAP.items():
            df[dst_col] = aligned_bonds[src_col] if src_col in aligned_bonds.columns else np.nan
    else:
        df["Bond_5Y"]  = np.nan
        df["Bond_10Y"] = np.nan

    # Commodities
    if comm_df is not None and not comm_df.empty:
        aligned_comm = comm_df.reindex(df.index, method="ffill")
        for col in list(_COMMODITY_TICKERS.keys()):
            df[col] = aligned_comm[col] if col in aligned_comm.columns else np.nan
    else:
        for col in list(_COMMODITY_TICKERS.keys()):
            df[col] = np.nan

    spy_sma = mkt_close.rolling(252).mean().replace(0, np.nan)
    df["Buffett_Proxy"] = (mkt_close / spy_sma) * 100.0

    # Macro/intermarket indicators. Daily market proxies are aligned to the
    # stock index; slower economic series use explicit neutral defaults until a
    # deployment provides authoritative history.
    if macro_df is not None and not macro_df.empty:
        aligned_macro = macro_df.reindex(df.index, method="ffill").bfill()
    else:
        aligned_macro = pd.DataFrame(index=df.index)

    for col in list(_MACRO_TICKERS.keys()) + list(_STATIC_MACRO_DEFAULTS.keys()):
        if col in aligned_macro.columns:
            df[col] = aligned_macro[col]
        else:
            df[col] = _STATIC_MACRO_DEFAULTS.get(col, np.nan)

    fundamentals = fundamentals or {}
    for col in [
        "PE_Ratio", "PB_Ratio", "EPS_Growth", "ROE", "ROCE",
        "Debt_To_Equity", "Institutional_Ownership",
    ]:
        df[col] = fundamentals.get(col, np.nan)

    return df

# ── Forecasting logic ─────────────────────────────────────────────────────────
def _holt_forecast(series: np.ndarray, alpha: float = 0.3, beta: float = 0.1, steps: int = 30):
    """
    Holt's Linear (Double) Exponential Smoothing.

    BUG FIX (v3.3): Trend was initialised as s[1]-s[0] (noisy single-step).
    Now uses global slope (s[-1]-s[0])/(n-1) per standard Holt initialisation.
    """
    s = series[~np.isnan(series)]
    if len(s) < 5:
        val = s[-1] if len(s) else 0.0
        return np.full(steps, val), np.zeros(steps)

    level = float(s[0])
    # FIX: global slope initialisation instead of s[1]-s[0]
    trend = float((s[-1] - s[0]) / (len(s) - 1)) if len(s) > 1 else 0.0

    for t in range(1, len(s)):
        last_l = level
        level  = alpha * s[t] + (1 - alpha) * (level + trend)
        trend  = beta  * (level - last_l) + (1 - beta) * trend

    fcast          = level + np.arange(1, steps + 1) * trend
    residue_sigma  = np.std(np.diff(s))
    bands          = residue_sigma * np.sqrt(np.arange(1, steps + 1))
    return fcast, bands

def forecast_commodity_prices(period_days: int = 730, forecast_days: int = 10) -> dict:
    key = f"comm_fcst:{period_days}:{forecast_days}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    results: dict = {}
    for name, ticker in _COMMODITY_TICKERS.items():
        try:
            raw = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            raw    = _normalize_columns(raw)
            prices = raw["Close"].dropna().values
            means, sigma = _holt_forecast(prices, steps=forecast_days)
            results[name] = {
                "last_price":     float(prices[-1]),
                "means":          means.tolist(),
                "upper":          (means + sigma).tolist(),
                "lower":          (means - sigma).tolist(),
                "pct_change_1d":  round(((means[0] - prices[-1]) / prices[-1]) * 100, 2),
                "dates": [
                    str((datetime.today() + timedelta(days=i)).date())
                    for i in range(1, forecast_days + 1)
                ],
                "alpha": 0.3,
                "beta":  0.1,
            }
        except Exception as e:
            print(f"[forecast_commodity_prices] {name}: {e}")

    return _cache_set(key, results)

def forecast_gdp_growth(period_days: int = 730, forecast_days: int = 30) -> dict:
    """
    Uses annualised 252-day ETF return as a GDP growth proxy, then applies
    Holt smoothing to produce a forecast with ±1σ bands.
    Falls back to consensus estimates when ETF data is unavailable.
    """
    consensus = {
        "US": 2.5, "India": 6.8, "China": 4.8,
        "EU": 1.2, "Japan": 1.0, "UK": 0.8, "Brazil": 2.2,
    }
    results: dict = {}

    for country, etf in _GDP_ETFS.items():
        base_rate = consensus[country]
        try:
            end   = datetime.today()
            start = end - timedelta(days=period_days + 50)
            raw   = yf.download(etf, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError("empty")
            raw    = _normalize_columns(raw)
            prices = raw["Close"].dropna()

            annual_ret = (prices / prices.shift(252) - 1) * 100.0
            annual_ret = annual_ret.dropna()

            if len(annual_ret) < 30:
                raise ValueError("insufficient return history")

            current_pct = float(annual_ret.iloc[-1])
            fcst, sigma = _holt_forecast(annual_ret.values, steps=forecast_days)

            results[country] = {
                "current_annual_pct": current_pct,
                "forecast_pct":       fcst.tolist(),
                "upper_pct":          (fcst + sigma).tolist(),
                "lower_pct":          (fcst - sigma).tolist(),
                "source":             f"{etf} ETF Proxy",
            }
        except Exception:
            _rng  = np.random.default_rng(abs(hash(country)) % (2**31))
            drift = np.linspace(base_rate, base_rate + _rng.uniform(-0.2, 0.2), forecast_days)
            results[country] = {
                "current_annual_pct": base_rate,
                "forecast_pct":       drift.tolist(),
                "upper_pct":          (drift + 0.3).tolist(),
                "lower_pct":          (drift - 0.3).tolist(),
                "source":             "Consensus Estimate",
            }

    return results

def compute_macro_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Close", "Bond_10Y", "Gold", "Brent_Oil",
        "Buffett_Proxy", "RSI", "Beta_60", "MACD",
        "Silver", "Copper",
    ]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    corr = df[available].dropna().corr()[["Close"]].drop("Close")
    corr.columns = ["correlation"]
    return corr.sort_values("correlation", ascending=False)

# ── ML helpers ────────────────────────────────────────────────────────────────
def build_sequences(scaled_data: np.ndarray, raw_close: np.ndarray, seq_length: int):
    """Predict log-returns — more stationary than raw prices."""
    log_returns = np.log(raw_close[1:] / np.where(raw_close[:-1] == 0, 1e-9, raw_close[:-1]))
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i])
        y.append(log_returns[i - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def split_data(X, y, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n         = len(X)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return (
        (X[:train_end],   y[:train_end]),
        (X[train_end:val_end], y[train_end:val_end]),
        (X[val_end:],     y[val_end:]),
        val_end,
        train_end,
    )

def returns_to_prices(log_returns: np.ndarray, start_price: float) -> np.ndarray:
    """Convert log-return sequence back to a price series."""
    prices = np.empty(len(log_returns))
    price  = start_price
    for i, r in enumerate(log_returns):
        price     = price * np.exp(r)
        prices[i] = price
    return prices


# ── Pure-NumPy Fuzzy C-Means ──────────────────────────────────────────────────
def _fuzzy_cmeans(
    X: np.ndarray,
    c: int   = 2,
    m: float = 2.0,
    max_iter: int   = 200,
    error:    float = 1e-5,
    seed:     int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuzzy C-Means clustering implemented in pure NumPy.
    No scikit-fuzzy dependency.

    Parameters
    ----------
    X        : shape (n_features, n_samples)  — matches skfuzzy convention
    c        : number of clusters
    m        : fuzzifier (>1); m=2 is standard
    max_iter : maximum iterations
    error    : convergence tolerance on membership matrix
    seed     : random seed for reproducible initialisation

    Returns
    -------
    cntr : shape (c, n_features)  — cluster centres
    u    : shape (c, n_samples)   — fuzzy membership matrix
    """
    rng = np.random.default_rng(seed)
    n_features, n_samples = X.shape

    # Initialise membership matrix randomly, then normalise columns to sum to 1
    u = rng.random((c, n_samples))
    u /= u.sum(axis=0, keepdims=True)

    for _ in range(max_iter):
        u_old = u.copy()

        # Compute cluster centres: cntr[k] = sum(u[k]^m * X) / sum(u[k]^m)
        um = u ** m                                      # (c, n_samples)
        cntr = (um @ X.T) / um.sum(axis=1, keepdims=True)  # (c, n_features)

        # Update membership matrix
        # dist[k, i] = ||X[:, i] - cntr[k]||^2
        dist = np.zeros((c, n_samples))
        for k in range(c):
            diff = X - cntr[k, :, np.newaxis]           # (n_features, n_samples)
            dist[k] = np.sum(diff ** 2, axis=0)

        # Avoid division by zero for points exactly on a centre
        dist = np.fmax(dist, 1e-12)

        # u[k, i] = 1 / sum_j( (dist[k,i] / dist[j,i])^(2/(m-1)) )
        exp = 2.0 / (m - 1.0)
        u = np.zeros((c, n_samples))
        for k in range(c):
            ratio = dist[k] / dist              # (c, n_samples)
            u[k]  = 1.0 / np.sum(ratio ** exp, axis=0)

        # Check convergence
        if np.linalg.norm(u - u_old) < error:
            break

    return cntr, u


# ── Fuzzy C-Means Risk Analysis ───────────────────────────────────────────────
def compute_fcm_risk(df: pd.DataFrame) -> dict:
    """
    Fuzzy C-Means (FCM) risk segregation — pure NumPy, no external fuzzy library.

    Methodology:
      1. Build a 3-feature risk matrix per trading day:
           - 21-day rolling annualised volatility
           - 60-day rolling beta (vs market proxy)
           - 60-day rolling max-drawdown magnitude
      2. Run FCM with c=2 clusters (high-risk, low-risk), fuzzifier m=2.
         FCM produces soft membership degrees [0,1] for every observation.
      3. Identify which cluster is "high risk" by comparing cluster centres
         on the volatility axis (higher vol centre = high risk).
      4. Group trading days by calendar year; compute the mean high-risk
         fuzzy membership for each year → "year-wise fuzzy risk value".
      5. Return yearly means + overall mean + current real-time risk score.
    """
    try:
        close   = df["Close"].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()

        # Feature 1: 21-day annualised volatility
        vol_21 = log_ret.rolling(21).std() * np.sqrt(252)

        # Feature 2: Beta_60 — guard against zero-variance after bfill
        if "Beta_60" in df.columns:
            beta_raw = df["Beta_60"].reindex(vol_21.index)
            beta = beta_raw if beta_raw.std() > 1e-6 else pd.Series(
                np.ones(len(vol_21)), index=vol_21.index
            )
        else:
            beta = pd.Series(np.ones(len(vol_21)), index=vol_21.index)

        # Feature 3: 60-day rolling max-drawdown magnitude
        rolling_max = close.reindex(vol_21.index).rolling(60, min_periods=5).max()
        drawdown    = (
            (close.reindex(vol_21.index) - rolling_max)
            / rolling_max.replace(0, np.nan)
        ).abs()

        # Assemble and clean
        feats = pd.DataFrame({
            "vol":      vol_21,
            "beta":     beta,
            "drawdown": drawdown,
        }).dropna()

        if len(feats) < 20:
            return _fcm_fallback(
                error=f"only {len(feats)} clean rows after dropna — need ≥20"
            )

        # Drop zero-variance columns (would make normalisation degenerate)
        feats = feats.loc[:, feats.std() > 1e-9]
        if feats.shape[1] < 2:
            return _fcm_fallback(error="insufficient feature variance")

        # Normalise to [0, 1] — shape (features, samples) for FCM
        from sklearn.preprocessing import MinMaxScaler
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(feats.values).T   # (n_features, n_samples)

        # Add tiny noise to any constant row so FCM distances stay non-zero
        row_std = X_scaled.std(axis=1, keepdims=True)
        if np.any(row_std < 1e-9):
            X_scaled += np.random.default_rng(0).uniform(0, 1e-6, X_scaled.shape)

        # Run pure-NumPy FCM
        cntr, u = _fuzzy_cmeans(X_scaled, c=2, m=2.0, max_iter=200, error=1e-5)

        # Sanity check
        if np.isnan(u).any() or np.isnan(cntr).any():
            return _fcm_fallback(error="FCM produced NaN — degenerate data")

        # High-risk cluster = higher normalised-volatility centre.
        # Look up "vol" by its position in the *surviving* feature columns
        # (zero-variance columns may have been dropped, shifting indices).
        feat_cols     = list(feats.columns)
        vol_feat_idx  = feat_cols.index("vol") if "vol" in feat_cols else 0
        vol_centres   = cntr[:, vol_feat_idx]
        high_risk_idx = int(np.argmax(vol_centres))
        high_risk_u   = u[high_risk_idx]

        membership = pd.Series(high_risk_u, index=feats.index)

        # Year-wise means (professor's requirement)
        yearly = (
            membership
            .groupby(membership.index.year)
            .mean()
            .round(4)
            .to_dict()
        )

        overall_mean  = float(np.mean(list(yearly.values())))
        current_score = float(membership.iloc[-1])

        risk_label = (
            "High Risk"     if current_score >= 0.65 else
            "Moderate Risk" if current_score >= 0.40 else
            "Low Risk"
        )

        # Denormalise cluster centre vols back to annualised %
        vol_min = float(feats["vol"].min()) * 100
        vol_max = float(feats["vol"].max()) * 100
        centre_vols = [
            round(vol_min + float(c) * (vol_max - vol_min), 2)
            for c in vol_centres
        ]

        return {
            "yearly":               {str(k): round(v, 4) for k, v in yearly.items()},
            "overall_mean":         round(overall_mean, 4),
            "current_score":        round(current_score, 4),
            "risk_label":           risk_label,
            "cluster_centers":      centre_vols,
            "high_risk_centre_idx": high_risk_idx,
            "error":                "",
        }

    except Exception as e:
        import traceback as _tb
        print(f"[compute_fcm_risk] FAILED — {type(e).__name__}: {e}")
        print(_tb.format_exc())
        return _fcm_fallback(error=str(e))


def _fcm_fallback(error: str = "") -> dict:
    return {
        "yearly":               {},
        "overall_mean":         None,
        "current_score":        None,
        "risk_label":           "Unavailable",
        "cluster_centers":      [],
        "high_risk_centre_idx": 1,
        "error":                error,
    }

def augment_with_prediction_errors(
    df: pd.DataFrame,
    hist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge per-day historical prediction error metrics from the Excel history
    back into the main feature DataFrame as two new columns:
      'Hist_Error2_Pct'  — rolling mean of |TodayPred - Actual| / Actual (%)
      'Hist_Bias_Pct'    — rolling mean of signed (TodayPred - Actual) / Actual (%)

    These columns give the model a signal about how well (or poorly) it has
    been predicting recently, helping it self-calibrate over time.

    Parameters
    ----------
    df   : feature DataFrame indexed by datetime (output of add_quant_features)
    hist : DataFrame from _read_excel_history(), must have Date, Today_Pred_Mean,
           Actual_Price columns.

    Returns
    -------
    df with two new columns added (or zeros if hist is empty / columns missing).
    """
    df = df.copy()
    df["Hist_Error2_Pct"] = 0.0
    df["Hist_Bias_Pct"]   = 0.0

    if hist.empty:
        return df

    required = {"Date", "Today_Pred_Mean", "Actual_Price"}
    if not required.issubset(hist.columns):
        return df

    valid = hist.dropna(subset=["Today_Pred_Mean", "Actual_Price"]).copy()
    if len(valid) < 5:
        return df

    valid = valid.set_index("Date").sort_index()
    actual  = valid["Actual_Price"]
    pred    = valid["Today_Pred_Mean"]
    denom   = actual.where(actual.abs() > 1e-9, 1e-9)

    error_pct = ((pred - actual).abs() / denom * 100.0).rolling(10, min_periods=3).mean()
    bias_pct  = ((pred - actual) / denom * 100.0).rolling(10, min_periods=3).mean()

    # Align with main df index; forward-fill gaps (weekend/holiday)
    aligned_error = error_pct.reindex(df.index, method="ffill").fillna(0.0)
    aligned_bias  = bias_pct.reindex(df.index,  method="ffill").fillna(0.0)

    df["Hist_Error2_Pct"] = aligned_error.values
    df["Hist_Bias_Pct"]   = aligned_bias.values
    return df


def validate_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    ticker: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate the feature matrix before scaling and sequence building.

    Checks:
    ① Each required column exists (fills missing with 0 and warns).
    ② No column is entirely NaN (drops it + warns).
    ③ Infinity values are replaced with NaN then forward-filled.
    ④ Returns the cleaned df and a list of warning strings.
    """
    warnings: list[str] = []
    df = df.copy()

    # ① Ensure all expected columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            warnings.append(f"[{ticker}] Missing feature '{col}' — filled with 0")

    # ② Drop all-NaN columns (they'd make MinMaxScaler produce NaN output)
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    if all_nan:
        for c in all_nan:
            df[c] = 0.0
        warnings.append(f"[{ticker}] All-NaN columns zeroed: {all_nan}")

    # ③ Replace infinities
    inf_mask = np.isinf(df[feature_cols].values)
    if inf_mask.any():
        n_inf = int(inf_mask.sum())
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        warnings.append(f"[{ticker}] {n_inf} ±Inf values replaced with NaN then ffilled")

    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

    return df, warnings
