"""
dashboard_streamlit.py — NIFTY 100 Quant Research Terminal  v1.0.0
==================================================================

• Per-ticker stats cached separately (TTL=300s) — unchanged filter
  interactions don't recompute heavy analytics.
• Multi-ticker data loaded lazily (only when the tab is active and
  tickers are selected), with a dedicated progress bar.
• All Plotly figures built inside @st.cache_data where safe; large
  correlation matrices only recomputed when the data window changes.
• Model metadata (JSON sidecars) loaded once per run, not once per chart.
• Model Health — reads <TICKER>_meta.json: MAPE trend, calibration
  status, drift flag, training history for every model.
• Prediction Calendar Heatmap — error2 % laid out on a calendar grid
  so you can spot day-of-week or seasonal patterns at a glance.
• Rolling Directional Accuracy chart — 20-day rolling window so you can
  see when the model gained/lost its edge.
• Signed Error table — data table colours Error1/Error2 cells:
    RED   = model over-predicted  (pred > actual)
    GREEN = model under-predicted (pred < actual)

Run:
    streamlit run dashboard_streamlit.py [-- --data-dir ./nifty100_data]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from research_platform import (
    optimize_portfolio,
    rank_opportunities,
    sector_rotation,
    simple_backtest,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY 100 · Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Column groups ─────────────────────────────────────────────────────────────
OHLCV_COLS  = ["Open", "High", "Low", "Close", "Volume"]
TECH_COLS   = ["SMA_10", "SMA_50", "MACD", "MACD_Signal", "RSI", "ROC_10",
               "Stoch_K", "BB_Upper", "BB_Lower", "ATR", "OBV_norm", "RVI"]
QUANT_COLS  = [
    "Beta_60", "Alpha_60", "Momentum_12_1", "Sharpe_60", "Vol_Ratio", "Mkt_Return",
    "Bond_5Y", "Bond_10Y", "Gold", "Silver", "Copper", "Aluminium", "Brent_Oil",
    "Buffett_Proxy", "USDINR", "DXY", "India_VIX", "FII_Flow", "DII_Flow",
    "Repo_Rate", "CPI_Inflation", "PMI", "GDP_Growth",
    "PE_Ratio", "PB_Ratio", "EPS_Growth", "ROE", "ROCE",
    "Debt_To_Equity", "Institutional_Ownership", "Regime_Confidence",
]
PRED_COLS   = ["Yest_Pred_Mean", "Yest_Pred_Std",
               "Today_Pred_Mean", "Today_Pred_Std", "Actual_Price",
               "Tomorrow_Pred_Mean", "Tomorrow_Pred_Std"]
ERROR1_COLS = ["Error1_Abs", "Error1_Pct"]
ERROR2_COLS = ["Error2_Abs", "Error2_Pct"]
ERROR_COLS  = ERROR1_COLS + ERROR2_COLS
ALL_FEATURE_COLS = OHLCV_COLS + TECH_COLS + QUANT_COLS

# ── Colour palette ────────────────────────────────────────────────────────────
P = {
    "bg":      "#08111f",
    "surface": "#0c1a2e",
    "card":    "#0f2040",
    "border":  "#1a3050",
    "accent":  "#00d4ff",
    "accent2": "#7b2fff",
    "green":   "#00e676",
    "red":     "#ff1744",
    "amber":   "#ffab00",
    "orange":  "#ff6d00",
    "teal":    "#00bfa5",
    "pink":    "#f50057",
    "text":    "#dce8f5",
    "muted":   "#4a6080",
    "err_red": "#D92B2B",
    "err_grn": "#1A8F4C",
}

CHART_COLORS = [
    P["accent"], P["green"], P["amber"], P["accent2"],
    P["teal"], P["orange"], P["pink"],
    "#a0c4ff", "#caffbf", "#ffd6a5", "#ffadad",
]


# ── CSS injection ─────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');

/* Base */
html, body, [data-testid="stApp"] {{
  background: {P["bg"]}; color: {P["text"]};
  font-family: 'Inter', sans-serif;
}}
/* Sidebar */
[data-testid="stSidebar"] {{
  background: {P["surface"]} !important;
  border-right: 1px solid {P["border"]};
}}
[data-testid="stSidebar"] * {{ color: {P["text"]} !important }}
[data-testid="stSidebar"] label {{
  color: {P["muted"]} !important; font-size: .68rem;
  letter-spacing: .1em; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
}}
/* Select boxes */
div[data-baseweb="select"] > div {{
  background: {P["card"]} !important;
  border: 1px solid {P["border"]} !important;
  border-radius: 6px !important;
}}
div[data-baseweb="select"] * {{ color: {P["text"]} !important }}

/* ── Metric card ── */
.mc {{
  background: {P["card"]}; border: 1px solid {P["border"]};
  border-radius: 10px; padding: 14px 16px;
  position: relative; overflow: hidden;
}}
.mc::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, {P["accent"]}, {P["accent2"]});
}}
.ml {{
  font-family: 'JetBrains Mono', monospace; font-size: .60rem;
  color: {P["muted"]}; letter-spacing: .14em; text-transform: uppercase;
  margin-bottom: 4px;
}}
.mv {{
  font-family: 'JetBrains Mono', monospace; font-size: 1.22rem;
  font-weight: 700; color: {P["text"]}; line-height: 1.1;
}}
.md {{ font-size: .70rem; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }}
.md-p {{ color: {P["green"]}; }}  .md-n {{ color: {P["red"]}; }}
.md-w {{ color: {P["amber"]}; }} .md-m {{ color: {P["muted"]}; }}

/* ── Hero ── */
.hero {{
  background: linear-gradient(135deg, {P["surface"]} 0%, #0a2040 60%, {P["surface"]} 100%);
  border: 1px solid {P["border"]}; border-radius: 14px;
  padding: 22px 28px; margin-bottom: 22px; position: relative; overflow: hidden;
}}
.hero::after {{
  content: ''; position: absolute; top: -1px; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, transparent, {P["accent"]}, {P["accent2"]}, transparent);
}}
.ht  {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: {P["accent"]}; }}
.hsb {{ font-size: .70rem; color: {P["muted"]}; font-family: 'JetBrains Mono', monospace; letter-spacing: .07em; }}
.hbadge {{
  display: inline-block;
  background: rgba(0,212,255,.10); border: 1px solid rgba(0,212,255,.28);
  color: {P["accent"]}; font-family: 'JetBrains Mono', monospace;
  font-size: .57rem; letter-spacing: .1em; padding: 2px 9px;
  border-radius: 20px; margin-right: 5px; text-transform: uppercase;
}}
.pg {{ display:inline-block; background:rgba(0,230,118,.1); border:1px solid rgba(0,230,118,.3); color:{P["green"]};  font-family:'JetBrains Mono',monospace; font-size:.57rem; padding:2px 8px; border-radius:20px; }}
.pr {{ display:inline-block; background:rgba(255,23,68,.1);  border:1px solid rgba(255,23,68,.3);  color:{P["red"]};   font-family:'JetBrains Mono',monospace; font-size:.57rem; padding:2px 8px; border-radius:20px; }}
.pa {{ display:inline-block; background:rgba(255,171,0,.1);  border:1px solid rgba(255,171,0,.3);  color:{P["amber"]}; font-family:'JetBrains Mono',monospace; font-size:.57rem; padding:2px 8px; border-radius:20px; }}

/* ── Section heading ── */
.sh {{
  font-family: 'JetBrains Mono', monospace; font-size: .63rem;
  letter-spacing: .18em; text-transform: uppercase; color: {P["accent"]};
  padding: 9px 0 5px; border-bottom: 1px solid {P["border"]}; margin-bottom: 13px;
}}

/* ── Error boxes ── */
.e1box {{
  background: rgba(255,109,0,.08); border: 1px solid rgba(255,109,0,.3);
  border-radius: 8px; padding: 11px 15px;
}}
.e2box {{
  background: rgba(255,23,68,.07); border: 1px solid rgba(255,23,68,.28);
  border-radius: 8px; padding: 11px 15px;
}}
.etitle {{ font-family:'JetBrains Mono',monospace; font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:5px; }}
.e1t {{ color: {P["orange"]}; }} .e2t {{ color: {P["red"]}; }}
.ebody {{ font-size:.73rem; color:{P["muted"]}; line-height:1.6; }}

/* ── Health panel ── */
.hcard {{
  background: {P["card"]}; border: 1px solid {P["border"]}; border-radius: 10px;
  padding: 14px; margin-bottom: 10px;
}}
.hc-ticker {{
  font-family: 'JetBrains Mono', monospace; font-size: .85rem;
  font-weight: 700; color: {P["accent"]}; margin-bottom: 6px;
}}
.hc-row {{
  font-family: 'JetBrains Mono', monospace; font-size: .68rem;
  color: {P["text"]}; margin-bottom: 2px; display: flex; justify-content: space-between;
}}
.hc-label {{ color: {P["muted"]}; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  background: {P["surface"]}; border-radius: 8px;
  border: 1px solid {P["border"]}; padding: 3px; gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: 'JetBrains Mono', monospace; font-size: .67rem;
  letter-spacing: .06em; color: {P["muted"]};
  background: transparent; border-radius: 6px; padding: 6px 12px;
}}
.stTabs [aria-selected="true"] {{
  background: {P["card"]} !important; color: {P["accent"]} !important;
}}

/* ── Buttons ── */
.stButton > button {{
  background: linear-gradient(135deg, {P["accent"]}, {P["accent2"]}) !important;
  color: #fff !important; border: none !important; border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: .67rem !important;
  padding: 5px 15px !important;
}}

/* ── Expander ── */
div[data-testid="stExpander"] {{
  border: 1px solid {P["border"]} !important;
  border-radius: 8px !important; background: {P["card"]} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
  background: {P["card"]} !important; border: 1px solid {P["border"]} !important;
  border-radius: 8px !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {P["surface"]}; }}
::-webkit-scrollbar-thumb {{ background: {P["border"]}; border-radius: 3px; }}
hr {{ border-color: {P["border"]}; margin: 16px 0; }}
</style>""", unsafe_allow_html=True)


# ── Plot layout defaults ──────────────────────────────────────────────────────
def _L(**kw) -> dict:
    base = dict(
        paper_bgcolor=P["bg"], plot_bgcolor=P["bg"],
        font=dict(family="JetBrains Mono, monospace", color=P["text"], size=11),
        margin=dict(l=56, r=30, t=46, b=38),
        legend=dict(bgcolor="rgba(12,26,46,.9)", bordercolor=P["border"],
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(gridcolor=P["border"], showgrid=True, zeroline=False,
                   linecolor=P["border"]),
        yaxis=dict(gridcolor=P["border"], showgrid=True, zeroline=False,
                   linecolor=P["border"]),
    )
    base.update(kw)
    return base


def _pc(fig: go.Figure, key: str, h: Optional[int] = None) -> None:
    if h:
        fig.update_layout(height=h)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER  (all cacheable)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def list_tickers(data_dir: str) -> list[str]:
    d = Path(data_dir)
    return sorted(f.stem for f in d.glob("*.xlsx")) if d.exists() else []


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker(stem: str, data_dir: str) -> Optional[pd.DataFrame]:
    path = Path(data_dir) / f"{stem}.xlsx"
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path, header=2, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(subset=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        nc = [c for c in df.columns if c != "Date"]
        df[nc] = df[nc].apply(pd.to_numeric, errors="coerce")
        return df
    except Exception as e:
        st.error(f"Load error [{stem}]: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_model_meta(ticker_stem: str, models_dir: str) -> Optional[dict]:
    path = Path(models_dir) / f"{ticker_stem}_meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def compute_stats(df_json: str) -> dict:
    """Accepts JSON-serialised df so Streamlit can hash it."""
    df = pd.read_json(df_json, orient="split")
    df["Date"] = pd.to_datetime(df["Date"])
    s: dict = {}

    cl = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
    if not cl.empty:
        s["price"] = float(cl.iloc[-1])
        s["prev"]  = float(cl.iloc[-2]) if len(cl) > 1 else s["price"]
        s["chg"]   = s["price"] - s["prev"]
        s["pct"]   = (s["chg"] / s["prev"] * 100) if s["prev"] else 0.0
        s["ret"]   = float((cl.iloc[-1] / cl.iloc[0] - 1) * 100) if cl.iloc[0] else 0.0

    for src, dst in [("Error1_Abs", "e1a"), ("Error1_Pct", "e1p"),
                     ("Error2_Abs", "e2a"), ("Error2_Pct", "e2p")]:
        col = df[src].dropna() if src in df.columns else pd.Series(dtype=float)
        if not col.empty:
            s[f"{dst}_mean"]   = float(col.mean())
            s[f"{dst}_median"] = float(col.median())
            s[f"{dst}_max"]    = float(col.max())
            s[f"{dst}_std"]    = float(col.std())

    if "Today_Pred_Mean" in df.columns and "Actual_Price" in df.columns:
        d2 = df.dropna(subset=["Today_Pred_Mean", "Actual_Price"])
        if len(d2) > 1:
            pd_ = np.sign(d2["Today_Pred_Mean"].diff()).fillna(0)
            ad_ = np.sign(d2["Actual_Price"].diff()).fillna(0)
            nz  = pd_ != 0
            s["dir_acc"] = float((pd_ == ad_)[nz].sum() / nz.sum() * 100) if nz.sum() else None
            s["over"]    = int((d2["Today_Pred_Mean"] > d2["Actual_Price"]).sum())
            s["under"]   = int((d2["Today_Pred_Mean"] < d2["Actual_Price"]).sum())

    if "Tomorrow_Pred_Mean" in df.columns:
        tm = df["Tomorrow_Pred_Mean"].dropna()
        if not tm.empty:
            s["tm_pred"] = float(tm.iloc[-1])
            ts = df["Tomorrow_Pred_Std"].dropna() if "Tomorrow_Pred_Std" in df.columns else pd.Series(dtype=float)
            s["tm_std"]  = float(ts.iloc[-1]) if not ts.empty else None
    return s


def date_filter(df: pd.DataFrame, s: datetime, e: datetime) -> pd.DataFrame:
    m = (df["Date"].dt.date >= s.date()) & (df["Date"].dt.date <= e.date())
    return df.loc[m].copy()


def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fp(v):  return f"₹{v:,.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"
def fpc(v): return f"{v:.2f}%"  if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"


def _axstyle(**kw) -> dict:
    return dict(gridcolor=P["border"], linecolor=P["border"], **kw)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Open", "High", "Low", "Close"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="OHLC",
        increasing=dict(line=dict(color=P["green"], width=1),
                        fillcolor="rgba(0,230,118,.45)"),
        decreasing=dict(line=dict(color=P["red"], width=1),
                        fillcolor="rgba(255,23,68,.45)"),
    ), row=1, col=1)
    for col, c, dsh in [("SMA_10", P["accent"], "dot"), ("SMA_50", P["amber"], "dash"),
                         ("BB_Upper", P["muted"], "dot"), ("BB_Lower", P["muted"], "dot")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=col,
                                     line=dict(color=c, width=1, dash=dsh)), row=1, col=1)
    if "Volume" in df.columns:
        vol = d["Volume"]
        colors = [P["green"] if c >= o else P["red"]
                  for c, o in zip(d["Close"], d["Open"])]
        fig.add_trace(go.Bar(x=d["Date"], y=vol, name="Volume",
                              marker_color=colors, opacity=0.55), row=2, col=1)
    for r in [1, 2]:
        fig.update_xaxes(**_axstyle(), row=r, col=1)
        fig.update_yaxes(**_axstyle(), row=r, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(**_L(title=dict(text=f"Candlestick + Volume · {ticker}", font=dict(size=13)),
                           height=480, showlegend=True, hovermode="x unified"))
    return fig


def fig_pred_overlay(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Actual_Price", "Today_Pred_Mean"])
    fig = go.Figure()
    # Confidence band
    if "Today_Pred_Std" in d.columns:
        ds = d.dropna(subset=["Today_Pred_Std"])
        u = ds["Today_Pred_Mean"] + ds["Today_Pred_Std"]
        l = ds["Today_Pred_Mean"] - ds["Today_Pred_Std"]
        fig.add_trace(go.Scatter(
            x=pd.concat([ds["Date"], ds["Date"].iloc[::-1]]),
            y=pd.concat([u, l.iloc[::-1]]),
            fill="toself", fillcolor="rgba(0,212,255,.06)",
            line=dict(color="rgba(0,0,0,0)"), name="±1σ", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Actual_Price"], name="Actual",
                              line=dict(color=P["green"], width=2)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Today_Pred_Mean"], name="Predicted",
                              line=dict(color=P["accent"], width=1.5, dash="dot")))
    # Tomorrow forecast extension
    if "Tomorrow_Pred_Mean" in df.columns:
        last = df.dropna(subset=["Tomorrow_Pred_Mean"]).iloc[-1:]
        if not last.empty:
            fd    = last["Date"].iloc[0] + pd.Timedelta(days=1)
            tm    = float(last["Tomorrow_Pred_Mean"].iloc[0])
            anc   = float(d["Actual_Price"].iloc[-1]) if not d.empty else tm
            ts    = last["Tomorrow_Pred_Std"].iloc[0] if "Tomorrow_Pred_Std" in last.columns else None
            fig.add_trace(go.Scatter(
                x=[last["Date"].iloc[0], fd], y=[anc, tm],
                mode="lines+markers", name="Tomorrow",
                line=dict(color=P["amber"], width=2, dash="dash"),
                marker=dict(size=[0, 9], color=P["amber"])))
            if ts and not np.isnan(ts):
                fig.add_trace(go.Scatter(
                    x=[fd, fd], y=[tm - ts, tm + ts], mode="lines",
                    showlegend=False, line=dict(color=P["amber"], width=6), opacity=0.3))
    fig.update_layout(**_L(title=dict(text=f"Actual vs Predicted · {ticker}", font=dict(size=13)),
                           height=360, hovermode="x unified"))
    return fig


def fig_rolling_dir_acc(df: pd.DataFrame, ticker: str, window: int = 20) -> go.Figure:
    d = df.dropna(subset=["Today_Pred_Mean", "Actual_Price"]).copy()
    if len(d) < window + 2:
        return go.Figure()
    # Rolling directional accuracy
    pred_dir   = np.sign(d["Today_Pred_Mean"].diff())
    actual_dir = np.sign(d["Actual_Price"].diff())
    correct    = (pred_dir == actual_dir).astype(float)
    # Exclude flat days
    flat       = (pred_dir == 0)
    correct[flat] = np.nan
    rolling = correct.rolling(window, min_periods=max(5, window // 2)).mean() * 100

    fig = go.Figure()
    # Shading
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(0,230,118,.05)", line_width=0)
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(255,23,68,.05)",  line_width=0)
    fig.add_hline(y=50, line_dash="dash", line_color=P["muted"], line_width=1)
    fig.add_hline(y=60, line_dash="dot",  line_color=P["green"],  line_width=0.8,
                  annotation_text="60% edge", annotation_font=dict(color=P["green"], size=9))

    colors = rolling.apply(
        lambda v: P["green"] if v >= 55 else (P["amber"] if v >= 45 else P["red"])
        if not np.isnan(v) else P["muted"]
    )
    fig.add_trace(go.Scatter(
        x=d["Date"], y=rolling, name=f"{window}d Roll Dir-Acc",
        mode="lines",
        line=dict(color=P["accent"], width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Dir Acc: %{y:.1f}%<extra></extra>",
    ))
    # Mark drift zones
    drift = rolling < 45
    dr = d[drift.values]
    if not dr.empty:
        fig.add_trace(go.Scatter(
            x=dr["Date"], y=rolling[drift.values], mode="markers",
            name="Sub-45% zone", marker=dict(color=P["red"], size=5, symbol="circle")))
    fig.update_layout(**_L(
        title=dict(text=f"{window}-Day Rolling Directional Accuracy · {ticker}", font=dict(size=13)),
        height=290, yaxis_title="Accuracy (%)",
        yaxis=dict(gridcolor=P["border"], range=[0, 100], zeroline=False),
        hovermode="x unified",
    ))
    return fig


def fig_cumulative_return(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Actual_Price"]).copy()
    if len(d) < 2:
        return go.Figure()
    d["_ca"] = (d["Actual_Price"] / d["Actual_Price"].iloc[0] - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["_ca"], name="Actual Return",
                              fill="tozeroy", fillcolor="rgba(0,230,118,.06)",
                              line=dict(color=P["green"], width=2)))
    if "Today_Pred_Mean" in d.columns:
        dp = d.dropna(subset=["Today_Pred_Mean"]).copy()
        if not dp.empty and dp["Today_Pred_Mean"].iloc[0] != 0:
            dp["_cp"] = (dp["Today_Pred_Mean"] / dp["Today_Pred_Mean"].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=dp["Date"], y=dp["_cp"], name="Pred Return",
                                      line=dict(color=P["accent"], width=1.5, dash="dot")))
    fig.add_hline(y=0, line_color=P["border"], line_width=1)
    fig.update_layout(**_L(
        title=dict(text=f"Cumulative Return · {ticker}", font=dict(size=13)),
        height=270, yaxis_title="Return (%)", hovermode="x unified",
    ))
    return fig


def fig_mc_bands(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Today_Pred_Mean", "Today_Pred_Std"]).copy()
    if d.empty:
        return go.Figure()
    u1 = d["Today_Pred_Mean"] + d["Today_Pred_Std"]
    l1 = d["Today_Pred_Mean"] - d["Today_Pred_Std"]
    u2 = d["Today_Pred_Mean"] + 2 * d["Today_Pred_Std"]
    l2 = d["Today_Pred_Mean"] - 2 * d["Today_Pred_Std"]
    fig = go.Figure()
    for u, l, fc, nm in [(u2, l2, "rgba(123,47,255,.07)", "±2σ"),
                          (u1, l1, "rgba(0,212,255,.10)",  "±1σ")]:
        fig.add_trace(go.Scatter(
            x=pd.concat([d["Date"], d["Date"].iloc[::-1]]),
            y=pd.concat([u, l.iloc[::-1]]),
            fill="toself", fillcolor=fc, line=dict(color="rgba(0,0,0,0)"),
            name=nm, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Today_Pred_Mean"],
                              name="Pred Mean", line=dict(color=P["accent"], width=1.8)))
    if "Actual_Price" in df.columns:
        da = df.dropna(subset=["Actual_Price"])
        fig.add_trace(go.Scatter(x=da["Date"], y=da["Actual_Price"],
                                  name="Actual", line=dict(color=P["green"], width=1.5)))
    fig.update_layout(**_L(
        title=dict(text=f"MC Uncertainty Bands · {ticker}", font=dict(size=13)),
        height=320, hovermode="x unified",
    ))
    return fig


def fig_scatter_pred(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Actual_Price", "Today_Pred_Mean"]).copy()
    if d.empty:
        return go.Figure()
    mn = min(d["Actual_Price"].min(), d["Today_Pred_Mean"].min()) * 0.98
    mx = max(d["Actual_Price"].max(), d["Today_Pred_Mean"].max()) * 1.02
    e2 = d["Error2_Pct"].fillna(0) if "Error2_Pct" in d.columns else pd.Series(0, index=d.index)
    # Direction: over vs under
    direction = np.where(d["Today_Pred_Mean"] > d["Actual_Price"], "Over", "Under")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect",
                              line=dict(color=P["muted"], dash="dash", width=1)))
    for lbl, clr, sym in [("Over", P["err_red"], "circle"), ("Under", P["err_grn"], "circle")]:
        mask = direction == lbl
        fig.add_trace(go.Scatter(
            x=d["Actual_Price"][mask], y=d["Today_Pred_Mean"][mask],
            mode="markers", name=lbl,
            marker=dict(size=6, color=clr, opacity=0.75, symbol=sym),
            hovertemplate="Actual: ₹%{x:.2f}<br>Pred: ₹%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**_L(
        title=dict(text=f"Scatter: Actual vs Predicted · {ticker} · RED=over / GREEN=under",
                   font=dict(size=12)),
        xaxis_title="Actual (₹)", yaxis_title="Predicted (₹)", height=330,
    ))
    return fig


def fig_error1_bar(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Error1_Pct"]).copy()
    if d.empty:
        return go.Figure()
    avg = float(d["Error1_Pct"].mean())
    # Color by magnitude relative to avg
    colors = [P["orange"] if v > avg * 1.5 else (P["amber"] if v > avg else "rgba(255,109,0,.4)")
              for v in d["Error1_Pct"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Date"], y=d["Error1_Pct"], marker_color=colors, name="E1 %",
                          hovertemplate="<b>%{x|%d %b %Y}</b><br>E1: %{y:.3f}%<extra></extra>"))
    fig.add_hline(y=avg, line_dash="dash", line_color=P["amber"],
                  annotation_text=f"avg {avg:.2f}%", annotation_font=dict(color=P["amber"], size=9))
    fig.update_layout(**_L(
        title=dict(text=f"Error 1 — Forecast Stability · {ticker}", font=dict(size=12, color=P["orange"])),
        height=255, yaxis_title="E1 %",
    ))
    return fig


def fig_error2_bar(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df.dropna(subset=["Error2_Pct"]).copy()
    if d.empty:
        return go.Figure()
    avg = float(d["Error2_Pct"].mean())
    # RED = over-prediction, GREEN = under-prediction
    if "Today_Pred_Mean" in d.columns and "Actual_Price" in d.columns:
        colors = [P["err_red"] if row["Today_Pred_Mean"] > row["Actual_Price"] else P["err_grn"]
                  for _, row in d.iterrows()]
    else:
        colors = P["red"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Date"], y=d["Error2_Pct"], marker_color=colors, name="E2 %",
                          hovertemplate="<b>%{x|%d %b %Y}</b><br>E2: %{y:.3f}%<br>"
                                        "<extra>🔴 over / 🟢 under</extra>"))
    fig.add_hline(y=avg, line_dash="dash", line_color=P["amber"],
                  annotation_text=f"avg {avg:.2f}%", annotation_font=dict(color=P["amber"], size=9))
    fig.update_layout(**_L(
        title=dict(text=f"Error 2 — Pred Accuracy · {ticker} · 🔴 over-pred / 🟢 under-pred",
                   font=dict(size=12, color=P["red"])),
        height=255, yaxis_title="E2 %",
    ))
    return fig


def fig_error_overlay(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    for col, nm, c, fc in [
        ("Error1_Pct", "E1 Stability", P["orange"], "rgba(255,109,0,.06)"),
        ("Error2_Pct", "E2 Accuracy",  P["red"],    "rgba(255,23,68,.06)"),
    ]:
        if col in df.columns:
            d = df.dropna(subset=[col])
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=nm, mode="lines",
                                      fill="tozeroy", fillcolor=fc,
                                      line=dict(color=c, width=1.6)))
    fig.update_layout(**_L(title=dict(text=f"Error 1 vs Error 2 Timeline · {ticker}", font=dict(size=13)),
                           height=270, yaxis_title="Error %", hovermode="x unified"))
    return fig


def fig_error_hist(df: pd.DataFrame, col: str, label: str, color: str,
                   ticker: str) -> go.Figure:
    e = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    if e.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=e, nbinsx=30,
        marker=dict(color=color, opacity=0.72, line=dict(color=P["border"], width=0.5)),
        name=label))
    fig.add_vline(x=float(e.mean()),   line_dash="dash", line_color=P["amber"],
                  annotation_text=f"μ={e.mean():.2f}%", annotation_font_color=P["amber"])
    fig.add_vline(x=float(e.median()), line_dash="dot",  line_color=P["green"],
                  annotation_text=f"med={e.median():.2f}%", annotation_font_color=P["green"])
    fig.update_layout(**_L(title=dict(text=f"{label} Distribution · {ticker}", font=dict(size=12)),
                           height=240, xaxis_title="Error %", yaxis_title="Count"))
    return fig


def fig_drift(df: pd.DataFrame, ticker: str, window: int) -> go.Figure:
    if "Error2_Pct" not in df.columns:
        return go.Figure()
    d = df.copy()
    d["_rm"]    = d["Error2_Pct"].rolling(window, min_periods=3).mean()
    hist_mean   = d["Error2_Pct"].expanding(min_periods=10).mean()
    d["_drift"] = d["_rm"] > (hist_mean * 1.5)
    d = d.dropna(subset=["_rm"])
    if d.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["_rm"],
                              name=f"{window}d Roll E2%",
                              line=dict(color=P["accent"], width=1.8)), row=1, col=1)
    drift_rows = d[d["_drift"]]
    if not drift_rows.empty:
        fig.add_trace(go.Scatter(x=drift_rows["Date"], y=drift_rows["_rm"],
                                  mode="markers", name="⚠ Drift",
                                  marker=dict(color=P["red"], size=7, symbol="x")), row=1, col=1)
    bar_colors = [P["red"] if v else P["accent"] for v in d["_drift"]]
    fig.add_trace(go.Bar(x=d["Date"], y=d["Error2_Pct"], name="Daily E2%",
                          marker_color=bar_colors, opacity=0.6), row=2, col=1)
    for r in [1, 2]:
        fig.update_xaxes(**_axstyle(), row=r, col=1)
        fig.update_yaxes(**_axstyle(), row=r, col=1)
    fig.update_layout(**_L(title=dict(text=f"Model Drift Detection · {ticker}", font=dict(size=13)),
                           height=340))
    return fig


def fig_prediction_calendar(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Calendar heatmap: Error2_Pct by day-of-week vs week-of-year."""
    d = df.dropna(subset=["Error2_Pct", "Date"]).copy()
    if len(d) < 10:
        return go.Figure()
    d["dow"]  = d["Date"].dt.dayofweek   # 0=Mon…4=Fri
    d["week"] = d["Date"].dt.isocalendar().week.astype(int)
    d["year"] = d["Date"].dt.year
    d["label"] = d["Date"].dt.strftime("%d %b %Y")

    # Pivot: week × DOW, mean error per cell
    pivot = d.pivot_table(values="Error2_Pct", index="week", columns="dow",
                          aggfunc="mean")
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    pivot.columns = [dow_names.get(c, str(c)) for c in pivot.columns]
    pivot = pivot.sort_index()

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[f"W{w}" for w in pivot.index],
        colorscale=[[0, P["err_grn"]], [0.5, P["amber"]], [1, P["err_red"]]],
        colorbar=dict(title="E2 %", thickness=11),
        hovertemplate="Day: %{x}<br>Week: %{y}<br>Mean E2: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_L(
        title=dict(text=f"Error 2 Calendar Heatmap · {ticker} · GREEN=small error / RED=large",
                   font=dict(size=12)),
        height=max(240, min(600, 30 * len(pivot) + 80)),
        xaxis=dict(gridcolor=P["border"]),
        yaxis=dict(gridcolor=P["border"], autorange="reversed"),
    ))
    return fig


def fig_bias_pie(df: pd.DataFrame, ticker: str) -> go.Figure:
    if "Today_Pred_Mean" not in df.columns or "Actual_Price" not in df.columns:
        return go.Figure()
    d = df.dropna(subset=["Today_Pred_Mean", "Actual_Price"])
    if d.empty:
        return go.Figure()
    over  = int((d["Today_Pred_Mean"] > d["Actual_Price"]).sum())
    under = int((d["Today_Pred_Mean"] < d["Actual_Price"]).sum())
    exact = int((d["Today_Pred_Mean"] == d["Actual_Price"]).sum())
    fig = go.Figure(go.Pie(
        labels=["Over-predicted", "Under-predicted", "Exact"],
        values=[over, under, exact],
        hole=0.55,
        marker=dict(colors=[P["err_red"], P["err_grn"], P["amber"]],
                    line=dict(color=P["border"], width=1)),
        textfont=dict(family="JetBrains Mono, monospace", size=10),
    ))
    fig.update_layout(**_L(
        title=dict(text=f"Prediction Bias Breakdown · {ticker}", font=dict(size=12)),
        height=290,
        showlegend=True,
        legend=dict(orientation="v", x=0.85, y=0.5),
    ))
    return fig


def fig_technical(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.50, 0.25, 0.25], vertical_spacing=0.04,
                        subplot_titles=["Price + Bollinger", "RSI", "MACD"])
    d = df.dropna(subset=["Close"])
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["BB_Upper"], name="BB↑",
                                  line=dict(color=P["muted"], width=0.8, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["Date"], y=d["BB_Lower"], name="BB↓",
                                  fill="tonexty", fillcolor="rgba(100,116,139,.05)",
                                  line=dict(color=P["muted"], width=0.8, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], name="Close",
                              line=dict(color=P["green"], width=2)), row=1, col=1)
    for col, c, dsh in [("SMA_10", P["accent"], "dot"), ("SMA_50", P["amber"], "dash")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=col,
                                      line=dict(color=c, width=1, dash=dsh)), row=1, col=1)
    if "RSI" in df.columns:
        dr = df.dropna(subset=["RSI"])
        rsi_color = [
            P["red"]   if v >= 70 else (P["green"] if v <= 30 else P["accent2"])
            for v in dr["RSI"]
        ]
        fig.add_trace(go.Scatter(x=dr["Date"], y=dr["RSI"], name="RSI",
                                  line=dict(color=P["accent2"], width=1.4)), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,23,68,.05)",
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,230,118,.05)",
                      line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_color=P["red"],   line_dash="dash", line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_color=P["green"], line_dash="dash", line_width=0.8, row=2, col=1)
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        dm   = df.dropna(subset=["MACD"])
        hist = dm["MACD"] - dm["MACD_Signal"]
        fig.add_trace(go.Bar(x=dm["Date"], y=hist, name="MACD Hist",
                              marker_color=[P["green"] if v >= 0 else P["red"] for v in hist],
                              opacity=0.65), row=3, col=1)
        fig.add_trace(go.Scatter(x=dm["Date"], y=dm["MACD"],
                                  name="MACD", line=dict(color=P["accent"], width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=dm["Date"], y=dm["MACD_Signal"],
                                  name="Signal", line=dict(color=P["red"], width=1.2, dash="dot")),
                      row=3, col=1)
    for r in [1, 2, 3]:
        fig.update_xaxes(**_axstyle(), row=r, col=1)
        fig.update_yaxes(**_axstyle(), row=r, col=1)
    fig.update_layout(**_L(title=dict(text=f"Technical Indicators · {ticker}", font=dict(size=13)),
                           height=520, showlegend=True))
    return fig


def fig_corr_heatmap(df: pd.DataFrame, ticker: str) -> go.Figure:
    cols = [c for c in TECH_COLS[:8] + QUANT_COLS + ["Actual_Price", "Error1_Pct", "Error2_Pct"]
            if c in df.columns]
    d = df[cols].dropna()
    if d.empty or len(cols) < 3:
        return go.Figure()
    corr = d.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, P["red"]], [0.5, P["bg"]], [1, P["green"]]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=11),
    ))
    fig.update_layout(**_L(
        title=dict(text=f"Feature Correlation · {ticker}", font=dict(size=13)),
        height=400,
        xaxis=dict(gridcolor=P["border"], tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=P["border"], tickfont=dict(size=9)),
    ))
    return fig


def fig_multi_ticker(dfs: dict, metric: str) -> go.Figure:
    fig = go.Figure()
    for i, (t, df) in enumerate(dfs.items()):
        if metric not in df.columns:
            continue
        d = df.dropna(subset=[metric])
        if d.empty or d[metric].iloc[0] == 0:
            continue
        fig.add_trace(go.Scatter(
            x=d["Date"], y=(d[metric] / d[metric].iloc[0]) * 100,
            name=t, mode="lines",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.6),
        ))
    fig.update_layout(**_L(
        title=dict(text=f"Multi-Ticker · {metric} (Indexed to 100)", font=dict(size=13)),
        height=390, yaxis_title="Index", hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0),
    ))
    return fig


def fig_feature_explorer(df: pd.DataFrame, ticker: str,
                          primary: list, secondary: list) -> go.Figure:
    if not primary and not secondary:
        return go.Figure()
    has_sec = bool(secondary)
    fig = make_subplots(specs=[[{"secondary_y": has_sec}]])
    ci  = 0
    for col in primary:
        if col not in df.columns:
            continue
        d = df.dropna(subset=[col])
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=col, mode="lines",
                                  line=dict(color=CHART_COLORS[ci % len(CHART_COLORS)], width=1.8)),
                      secondary_y=False)
        ci += 1
    for col in secondary:
        if col not in df.columns:
            continue
        d = df.dropna(subset=[col])
        fig.add_trace(go.Scatter(x=d["Date"], y=d[col], name=f"{col} (R)", mode="lines",
                                  line=dict(color=CHART_COLORS[ci % len(CHART_COLORS)],
                                            width=1.5, dash="dot")),
                      secondary_y=True)
        ci += 1
    fig.update_xaxes(**_axstyle())
    fig.update_yaxes(**_axstyle(), secondary_y=False)
    if has_sec:
        fig.update_yaxes(**_axstyle(), showgrid=False, secondary_y=True)
    title = ", ".join(primary) + (f"  |  R: {', '.join(secondary)}" if secondary else "")
    fig.update_layout(**_L(title=dict(text=f"{title} · {ticker}", font=dict(size=12)),
                           height=390, hovermode="x unified"))
    return fig


def fig_portfolio_table(rows: list[dict]) -> go.Figure:
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)

    def f(v, fmt):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        try:
            return fmt.format(v)
        except Exception:
            return str(v)

    # Color-code E2 column
    e2_colors = []
    for v in df.get("e2", [None] * len(df)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            e2_colors.append(P["muted"])
        elif v < 2:
            e2_colors.append(P["err_grn"])
        elif v < 5:
            e2_colors.append(P["amber"])
        else:
            e2_colors.append(P["err_red"])

    chg_colors = []
    for v in df.get("pct_change", [None] * len(df)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            chg_colors.append(P["muted"])
        else:
            chg_colors.append(P["err_grn"] if v >= 0 else P["err_red"])

    n = len(df)
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>TICKER</b>", "<b>PRICE</b>", "<b>CHG%</b>",
                    "<b>E1 AVG%</b>", "<b>E2 AVG%</b>", "<b>DIR ACC</b>",
                    "<b>OVER/UNDER</b>", "<b>ROWS</b>"],
            fill_color=P["card"],
            font=dict(color=P["accent"], family="JetBrains Mono, monospace", size=10),
            align="center", height=30, line_color=P["border"],
        ),
        cells=dict(
            values=[
                df["ticker"].tolist(),
                [f(v, "₹{:,.2f}")  for v in df.get("latest",      [None] * n)],
                [f(v, "{:+.2f}%")  for v in df.get("pct_change",   [None] * n)],
                [f(v, "{:.2f}%")   for v in df.get("e1",           [None] * n)],
                [f(v, "{:.2f}%")   for v in df.get("e2",           [None] * n)],
                [f(v, "{:.1f}%")   for v in df.get("dir_acc",      [None] * n)],
                [f"{v[0]}/{v[1]}"  for v in df.get("over_under",   [(0, 0)] * n)],
                [str(v)            for v in df.get("rows",          [0] * n)],
            ],
            fill_color=[[P["surface"] if i % 2 == 0 else P["card"] for i in range(n)]],
            font=dict(color=[
                [P["text"]] * n,
                [P["text"]] * n,
                chg_colors,
                [P["amber"]] * n,
                e2_colors,
                [P["green"] if (v and v >= 55) else P["red"] for v in df.get("dir_acc", [None] * n)],
                [P["muted"]] * n,
                [P["muted"]] * n,
            ], family="JetBrains Mono, monospace", size=10),
            align="center", height=26, line_color=P["border"],
        ),
    ))
    fig.update_layout(**_L(height=max(160, 56 + 26 * n), margin=dict(l=0, r=0, t=4, b=0)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def mc_card(label: str, value: str, delta: str = "", dt: str = "m") -> str:
    dc = {"p": "md-p", "n": "md-n", "w": "md-w", "m": "md-m"}.get(dt, "md-m")
    dh = f'<div class="{dc} md">{delta}</div>' if delta else ""
    return f'<div class="mc"><div class="ml">{label}</div><div class="mv">{value}</div>{dh}</div>'


def render_kpi(s: dict) -> None:
    cols = st.columns(8)
    chg = s.get("pct", 0) or 0
    cols[0].markdown(mc_card("Latest Close", fp(s.get("price")),
                              f"{'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%",
                              "p" if chg >= 0 else "n"), unsafe_allow_html=True)
    tm = s.get("tm_pred")
    cols[1].markdown(mc_card("Tomorrow Pred", fp(tm),
                              f"±{s['tm_std']:.2f}" if s.get("tm_std") else "No σ", "m"),
                     unsafe_allow_html=True)
    da = s.get("dir_acc")
    cols[2].markdown(mc_card("Dir Accuracy", fpc(da), "",
                              "p" if da and da >= 55 else ("w" if da and da >= 48 else "n")),
                     unsafe_allow_html=True)
    e1 = s.get("e1p_mean")
    cols[3].markdown(mc_card("E1 Avg% Stability", fpc(e1), "Forecast Consistency", "w"),
                     unsafe_allow_html=True)
    e2 = s.get("e2p_mean")
    e2t = "p" if e2 and e2 < 2 else ("w" if e2 and e2 < 5 else "n")
    cols[4].markdown(mc_card("E2 Avg% Accuracy", fpc(e2), "Pred vs Actual", e2t),
                     unsafe_allow_html=True)
    cols[5].markdown(mc_card("E2 Max (₹)", fp(s.get("e2a_max")), "Worst miss",  "n"),
                     unsafe_allow_html=True)
    ret = s.get("ret")
    cols[6].markdown(mc_card("Period Return", fpc(ret),
                              f"{'▲' if ret and ret >= 0 else '▼'} cumulative",
                              "p" if ret and ret >= 0 else "n"), unsafe_allow_html=True)
    ov = s.get("over", 0) or 0
    un = s.get("under", 0) or 0
    cols[7].markdown(mc_card("Over / Under", f"{ov} / {un}", "prediction bias", "m"),
                     unsafe_allow_html=True)


def render_hero(ticker: str, s: dict, df: pd.DataFrame,
                meta: Optional[dict] = None) -> None:
    price = s.get("price")
    chg   = s.get("pct", 0) or 0
    cc    = P["green"] if chg >= 0 else P["red"]
    arr   = "▲" if chg >= 0 else "▼"
    sd    = df["Date"].iloc[0].strftime("%d %b %Y") if not df.empty else "—"
    ed    = df["Date"].iloc[-1].strftime("%d %b %Y") if not df.empty else "—"

    # Drift badge from meta or computed
    drift = False
    if meta:
        drift = bool(meta.get("drift_detected", False))
    drift_html = (f'<span class="pr">⚠ DRIFT</span>' if drift
                  else f'<span class="pg">✓ STABLE</span>')

    # Calibration badge
    calib_html = ""
    if meta and meta.get("calibration_on"):
        bias = meta.get("bias_offset", 0)
        calib_html = f'<span class="pa">⚖ CALIB ₹{bias:+.2f}</span>'

    # Training mode badge
    mode_html = ""
    if meta:
        mode = meta.get("training_mode", "")
        mc_  = {"retrain": "pr", "finetune": "pa", "loaded": "pg"}.get(mode, "pa")
        mode_html = f'<span class="{mc_}">{mode.upper()}</span>'

    e2ok     = (s.get("e2p_mean") or 999) < 3
    e2_html  = (f'<span class="pg">E2 &lt; 3%</span>' if e2ok
                else f'<span class="pa">E2 ≥ 3%</span>')

    run_ct   = f' · runs: {meta["run_count"]}' if meta and meta.get("run_count") else ""

    st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div class="ht">{ticker}</div>
      <div class="hsb">NSE · NIFTY 100 · CNN-BiLSTM-Attention · Monte Carlo Dropout</div>
      <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:5px">
        <span class="hbadge">Error 1: Stability</span>
        <span class="hbadge">Error 2: Accuracy</span>
        {drift_html}&nbsp;{e2_html}&nbsp;{calib_html}&nbsp;{mode_html}
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;color:{P['text']}">
        {"₹{:,.2f}".format(price) if price else "—"}
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.9rem;color:{cc}">{arr} {abs(chg):.2f}%</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.60rem;color:{P['muted']};margin-top:3px">
        {len(df)} rows · {sd} → {ed}{run_ct}
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_error_legend() -> None:
    st.markdown(f"""
<div style="display:flex;gap:13px;margin-bottom:16px">
  <div class="e1box" style="flex:1">
    <div class="etitle e1t">⚡ Error 1 — Forecast Stability</div>
    <div class="ebody">
      <code>|Yest_Pred_Mean − Today_Pred_Mean| / Today_Pred_Mean × 100</code><br>
      How much the model <b>revised its estimate overnight</b>. Does not need Actual_Price.<br>
      Font colour in Excel: <b style="color:{P['err_red']}">RED</b> = model revised up,
      &nbsp;<b style="color:{P['err_grn']}">GREEN</b> = model revised down.
    </div>
  </div>
  <div class="e2box" style="flex:1">
    <div class="etitle e2t">🎯 Error 2 — Prediction Accuracy</div>
    <div class="ebody">
      <code>|Today_Pred_Mean − Actual_Price| / Actual_Price × 100</code><br>
      How far the final prediction was from the <b>true close price</b>. Primary drift signal.<br>
      Font colour in Excel: <b style="color:{P['err_red']}">RED</b> = over-predicted,
      &nbsp;<b style="color:{P['err_grn']}">GREEN</b> = under-predicted.
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def render_signed_error_table(df: pd.DataFrame, ticker: str,
                               show_verbose: bool = False) -> None:
    st.markdown(f'<div class="sh">🗃 Data Table · {ticker}</div>', unsafe_allow_html=True)
    base  = ["Date"] + OHLCV_COLS[:5] + PRED_COLS + ERROR_COLS
    extra = TECH_COLS + QUANT_COLS if show_verbose else []
    avail = [c for c in base + extra if c in df.columns]
    d = df[avail].copy()
    if "Date" in d.columns:
        d["Date"] = d["Date"].dt.strftime("%Y-%m-%d")

    # Build column config
    fmt: dict = {}
    for c in avail:
        if c == "Date":
            fmt[c] = st.column_config.TextColumn(c)
        elif c == "Volume":
            fmt[c] = st.column_config.NumberColumn(c, format="%d")
        elif c in ERROR_COLS:
            fmt[c] = st.column_config.NumberColumn(c, format="%.4f")
        else:
            fmt[c] = st.column_config.NumberColumn(c, format="%.4f")

    st.dataframe(
        d.sort_values("Date", ascending=False).reset_index(drop=True),
        use_container_width=True, height=340, column_config=fmt,
    )

    # Colour legend note
    st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{P['muted']};margin-top:6px">
  📌 Excel font colours on error columns —
  <b style="color:{P['err_red']}">RED</b> = over-predicted / revised up &nbsp;|&nbsp;
  <b style="color:{P['err_grn']}">GREEN</b> = under-predicted / revised down
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇ Export CSV", data=d.to_csv(index=False).encode(),
                       file_name=f"{ticker}_data.csv", mime="text/csv")


def render_sidebar(data_dir: str):
    st.sidebar.markdown(f"""
<div style="padding:13px 0 4px;font-family:'JetBrains Mono',monospace">
  <div style="font-size:1.05rem;font-weight:700;color:{P['accent']};letter-spacing:.06em">⬡ NIFTY 100</div>
  <div style="font-size:.57rem;color:{P['muted']};letter-spacing:.13em;text-transform:uppercase">Quant Research Terminal · v1.0.0</div>
</div>
<hr style="border-color:{P['border']};margin:8px 0 13px"/>""", unsafe_allow_html=True)

    tickers = list_tickers(data_dir)
    if not tickers:
        st.sidebar.error(f"No .xlsx files found in:\n`{data_dir}`")
        st.stop()

    st.sidebar.markdown(f'<div class="sh">📊 Ticker</div>', unsafe_allow_html=True)
    ticker = st.sidebar.selectbox("Ticker", tickers, label_visibility="collapsed")

    df_full = load_ticker(ticker, data_dir)
    if df_full is not None and not df_full.empty:
        min_d = df_full["Date"].min().date()
        max_d = df_full["Date"].max().date()
        def_s = max(min_d, (datetime.now() - timedelta(days=180)).date())
    else:
        min_d = (datetime.now() - timedelta(days=365)).date()
        max_d = datetime.now().date()
        def_s = min_d

    st.sidebar.markdown(f'<div class="sh">📅 Date Range</div>', unsafe_allow_html=True)
    start_d = st.sidebar.date_input("Start", value=def_s, min_value=min_d, max_value=max_d)
    end_d   = st.sidebar.date_input("End",   value=max_d, min_value=min_d, max_value=max_d)

    st.sidebar.markdown(f'<div class="sh">🔀 Compare</div>', unsafe_allow_html=True)
    compare = st.sidebar.multiselect(
        "Compare tickers", tickers,
        default=[tickers[0]] if tickers else [],
        max_selections=8, label_visibility="collapsed",
    )

    st.sidebar.markdown(f'<div class="sh">⚙ Options</div>', unsafe_allow_html=True)
    show_verbose = st.sidebar.checkbox("All feature columns in data table", False)
    drift_win    = st.sidebar.slider("E2 drift window (days)", 5, 60, 20)
    roll_acc_win = st.sidebar.slider("Rolling Dir-Acc window (days)", 10, 60, 20)

    if df_full is not None:
        has_p  = "Today_Pred_Mean" in df_full and df_full["Today_Pred_Mean"].notna().any()
        has_e1 = "Error1_Pct" in df_full and df_full["Error1_Pct"].notna().any()
        has_e2 = "Error2_Pct" in df_full and df_full["Error2_Pct"].notna().any()
        st.sidebar.markdown(f"""
<div style="background:{P['card']};border:1px solid {P['border']};border-radius:8px;padding:11px;margin-top:11px">
  <div style="font-family:'JetBrains Mono',monospace;font-size:.57rem;color:{P['muted']};letter-spacing:.1em;text-transform:uppercase;margin-bottom:7px">Dataset</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.70rem;color:{P['text']};line-height:1.7">
    📁 {ticker}.xlsx<br>
    📅 {len(df_full)} total rows<br>
    🤖 Predictions: {'✓' if has_p else '✗'}<br>
    ⚡ Error 1: {'✓' if has_e1 else '✗'}<br>
    🎯 Error 2: {'✓' if has_e2 else '✗'}<br>
    📐 Cols: {len(df_full.columns)}<br>
    🗓 {str(min_d)} → {str(max_d)}
  </div>
</div>""", unsafe_allow_html=True)

    return (
        ticker, compare,
        datetime.combine(start_d, datetime.min.time()),
        datetime.combine(end_d,   datetime.max.time()),
        show_verbose, drift_win, roll_acc_win,
    )


def render_model_health(ticker: str, models_dir: str) -> None:
    stem = ticker.replace(".NS", "").replace("/", "_")
    meta = load_model_meta(stem, models_dir)

    if meta is None:
        st.info(f"No model health file found for **{ticker}**. Run the tracker at least once.")
        return

    st.markdown(f'<div class="sh">🏥 Model Health Report · {ticker}</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    mape    = meta.get("mape_at_train")
    dir_acc = meta.get("directional_acc")
    bias    = meta.get("bias_offset", 0)
    drift   = meta.get("drift_detected", False)
    runs    = meta.get("run_count", 0)
    mode    = meta.get("training_mode", "—")
    calib   = meta.get("calibration_on", False)
    epochs  = meta.get("epochs_run", 0)
    val_l   = meta.get("best_val_loss", 0)
    n_hist  = meta.get("n_history_rows", 0)
    trained = meta.get("trained_on", "—")
    last    = meta.get("last_run", "—")

    c1.markdown(mc_card("MAPE at Training", fpc(mape), "", "p" if mape and mape < 3 else "n"),
                unsafe_allow_html=True)
    c2.markdown(mc_card("Dir Accuracy", fpc(dir_acc), "",
                         "p" if dir_acc and dir_acc >= 55 else "n"),
                unsafe_allow_html=True)
    c3.markdown(mc_card("Bias Offset", f"₹{bias:+.2f}" if bias else "—",
                         "Calibrated ✓" if calib else "Not calibrated", "w"),
                unsafe_allow_html=True)
    c4.markdown(mc_card("Total Runs", str(runs), f"Mode: {mode}", "m"),
                unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(mc_card("Drift Detected", "⚠ YES" if drift else "✓ NO", "",
                         "n" if drift else "p"), unsafe_allow_html=True)
    c6.markdown(mc_card("Best Val Loss", f"{val_l:.6f}" if val_l else "—", f"Epochs: {epochs}", "m"),
                unsafe_allow_html=True)
    c7.markdown(mc_card("History Rows", str(n_hist), f"Trained: {trained}", "m"),
                unsafe_allow_html=True)
    c8.markdown(mc_card("Last Run", last[:10] if last and last != "—" else "—", "", "m"),
                unsafe_allow_html=True)

    with st.expander("📄 Raw metadata JSON"):
        st.json(meta)


def render_portfolio(data_dir: str, models_dir: str,
                     start: datetime, end: datetime) -> None:
    st.markdown(f'<div class="sh">📋 Portfolio Summary — All Tickers</div>',
                unsafe_allow_html=True)
    tickers = list_tickers(data_dir)
    rows    = []
    prog    = st.progress(0, text="Loading tickers…")
    for i, t in enumerate(tickers):
        df = load_ticker(t, data_dir)
        if df is None or df.empty:
            continue
        df2 = date_filter(df, start, end)
        if df2.empty:
            continue
        s = compute_stats(_df_to_json(df2))
        rows.append({
            "ticker":    t,
            "latest":    s.get("price"),
            "pct_change":s.get("pct"),
            "e1":        s.get("e1p_mean"),
            "e2":        s.get("e2p_mean"),
            "dir_acc":   s.get("dir_acc"),
            "over_under":(s.get("over", 0) or 0, s.get("under", 0) or 0),
            "rows":      len(df2),
        })
        prog.progress((i + 1) / max(len(tickers), 1), text=f"Loaded {t}…")
    prog.empty()

    if not rows:
        st.warning("No data found in the selected date range.")
        return

    _pc(fig_portfolio_table(rows), key="port_table")

    df_s = pd.DataFrame(rows)
    c1, c2 = st.columns(2)

    for col, label, is_e2, colref, bk in [
        ("e1", "Error 1 Avg% (Stability)", False, c1, "port_e1"),
        ("e2", "Error 2 Avg% (Accuracy)",  True,  c2, "port_e2"),
    ]:
        if col in df_s.columns:
            dfe = df_s.dropna(subset=[col]).sort_values(col)
            bar_c = []
            for v in dfe[col]:
                thr_g, thr_a = (2, 5) if is_e2 else (1, 3)
                bar_c.append(P["err_grn"] if v < thr_g else (P["amber"] if v < thr_a else P["err_red"]))
            f = go.Figure(go.Bar(x=dfe["ticker"], y=dfe[col], marker_color=bar_c))
            f.update_layout(**_L(title=label, height=260))
            colref.plotly_chart(f, use_container_width=True, key=bk)

    if "dir_acc" in df_s.columns:
        dfd = df_s.dropna(subset=["dir_acc"]).sort_values("dir_acc", ascending=False)
        fd  = go.Figure(go.Bar(
            x=dfd["ticker"], y=dfd["dir_acc"],
            marker_color=[P["err_grn"] if v >= 55 else (P["amber"] if v >= 48 else P["err_red"])
                          for v in dfd["dir_acc"]],
        ))
        fd.add_hline(y=50, line_dash="dash", line_color=P["muted"])
        fd.add_hline(y=55, line_dash="dot",  line_color=P["green"],
                     annotation_text="55% edge", annotation_font=dict(color=P["green"], size=9))
        fd.update_layout(**_L(title="Directional Accuracy by Ticker", height=270))
        _pc(fd, key="port_dir_acc")

    ranked = rank_opportunities(rows)
    if not ranked.empty:
        st.markdown(f'<div class="sh">Stock Ranking Engine</div>', unsafe_allow_html=True)
        show_cols = [
            c for c in ["ticker", "side", "score", "confidence", "pct_change", "e2", "dir_acc"]
            if c in ranked.columns
        ]
        st.dataframe(
            ranked[show_cols].round(4),
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    sectors = sector_rotation(rows)
    if not sectors.empty:
        st.markdown(f'<div class="sh">Sector Rotation</div>', unsafe_allow_html=True)
        st.dataframe(sectors.round(4), use_container_width=True, hide_index=True, height=260)

    price_map = {}
    for t in tickers[:40]:
        d = load_ticker(t, data_dir)
        if d is not None and not d.empty:
            price_map[t] = date_filter(d, start, end).set_index("Date")
    alloc = optimize_portfolio(price_map, method="mean_variance")
    if not alloc.empty:
        st.markdown(f'<div class="sh">Portfolio Optimization · Mean Variance</div>',
                    unsafe_allow_html=True)
        st.dataframe(alloc.round(4), use_container_width=True, hide_index=True, height=280)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    # ── Resolve data dir ─────────────────────────────────────────────────────
    if "data_dir" not in st.session_state:
        try:
            idx = sys.argv.index("--")
            pa  = argparse.ArgumentParser()
            pa.add_argument("--data-dir",    default="nifty100_data")
            pa.add_argument("--models-dir",  default="nifty100_models")
            psd, _ = pa.parse_known_args(sys.argv[idx + 1:])
            st.session_state["data_dir"]   = psd.data_dir
            st.session_state["models_dir"] = psd.models_dir
        except (ValueError, SystemExit):
            st.session_state["data_dir"]   = "nifty100_data"
            st.session_state["models_dir"] = "nifty100_models"

    data_dir   = st.session_state["data_dir"]
    models_dir = st.session_state["models_dir"]

    (ticker, compare, start, end,
     show_verbose, drift_win, roll_acc_win) = render_sidebar(data_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {ticker}…"):
        df_full = load_ticker(ticker, data_dir)
    if df_full is None or df_full.empty:
        st.error(f"No data for **{ticker}**")
        st.stop()

    df = date_filter(df_full, start, end)
    if df.empty:
        st.warning("No data in selected range — showing full history.")
        df = df_full.copy()

    # Load model meta (cached, doesn't block)
    stem = ticker.replace(".NS", "").replace("/", "_")
    meta = load_model_meta(stem, models_dir)

    stats = compute_stats(_df_to_json(df))

    # ── Hero + KPIs ───────────────────────────────────────────────────────────
    render_hero(ticker, stats, df, meta)
    render_kpi(stats)
    st.markdown("<div style='height:14px'/>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Predictions",
        "⚡ Error Analysis",
        "🕯 Price & Technical",
        "🔭 Feature Explorer",
        "🏥 Model Health",
        "🌐 Multi-Ticker",
        "📋 Portfolio",
        "🗃 Data",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 0 · Predictions
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f'<div class="sh">Actual vs Predicted · {ticker}</div>',
                    unsafe_allow_html=True)
        _pc(fig_pred_overlay(df, ticker), key="t0_pred")

        c1, c2 = st.columns(2)
        with c1:
            _pc(fig_cumulative_return(df, ticker), key="t0_cumret")
        with c2:
            _pc(fig_mc_bands(df, ticker), key="t0_mc")

        _pc(fig_rolling_dir_acc(df, ticker, window=roll_acc_win), key="t0_rollacc")
        _pc(fig_scatter_pred(df, ticker), key="t0_scatter")
        bt = simple_backtest(df)
        if bt:
            st.markdown(f'<div class="sh">Walk-Forward Style Signal Backtest</div>',
                        unsafe_allow_html=True)
            bcols = st.columns(6)
            for col, (label, key) in zip(
                bcols,
                [
                    ("CAGR", "cagr"),
                    ("Sharpe", "sharpe"),
                    ("Sortino", "sortino"),
                    ("Max DD", "max_drawdown"),
                    ("Win Rate", "win_rate"),
                    ("Calmar", "calmar"),
                ],
            ):
                suffix = "%" if key in {"cagr", "max_drawdown", "win_rate"} else ""
                col.markdown(mc_card(label, f"{bt.get(key, 0)}{suffix}"), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 · Error Analysis
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        render_error_legend()
        _pc(fig_error_overlay(df, ticker), key="t1_overlay")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="sh" style="color:{P["orange"]}">⚡ Error 1 — Forecast Stability</div>',
                        unsafe_allow_html=True)
            _pc(fig_error1_bar(df, ticker),  key="t1_e1_bar")
            _pc(fig_error_hist(df, "Error1_Pct", "E1 %", P["orange"], ticker), key="t1_e1_hist")
        with c2:
            st.markdown(f'<div class="sh" style="color:{P["red"]}">🎯 Error 2 — Prediction Accuracy</div>',
                        unsafe_allow_html=True)
            _pc(fig_error2_bar(df, ticker),  key="t1_e2_bar")
            _pc(fig_error_hist(df, "Error2_Pct", "E2 %", P["red"], ticker),    key="t1_e2_hist")

        c3, c4 = st.columns([1.4, 1])
        with c3:
            _pc(fig_drift(df, ticker, drift_win), key="t1_drift")
        with c4:
            _pc(fig_bias_pie(df, ticker), key="t1_pie")

        _pc(fig_prediction_calendar(df, ticker), key="t1_calendar")

        # Summary stats table
        st.markdown(f'<div class="sh">Error Statistics Summary</div>', unsafe_allow_html=True)
        err_rows = [
            ["Definition",    "|YestPred − TodayPred| / TodayPred × 100",
                              "|TodayPred − Actual| / Actual × 100"],
            ["When computed", "Daily (no Actual needed)",
                              "Post-market close (needs Actual_Price)"],
            ["Excel colour",  "🔴 revised up  /  🟢 revised down",
                              "🔴 over-pred  /  🟢 under-pred"],
            ["Mean %",        fpc(stats.get("e1p_mean")),   fpc(stats.get("e2p_mean"))],
            ["Median %",      fpc(stats.get("e1p_median")), fpc(stats.get("e2p_median"))],
            ["Mean Abs (₹)",  fp(stats.get("e1a_mean")),    fp(stats.get("e2a_mean"))],
            ["Max Abs (₹)",   fp(stats.get("e1a_max")),     fp(stats.get("e2a_max"))],
            ["Std Dev (₹)",   fp(stats.get("e1a_std")),     fp(stats.get("e2a_std"))],
        ]
        st.dataframe(
            pd.DataFrame(err_rows, columns=["Metric", "Error 1 (Stability)", "Error 2 (Accuracy)"]),
            use_container_width=True, hide_index=True, height=280,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 · Price & Technical
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True,
                               key="chart_type")
        if chart_type == "Candlestick":
            _pc(fig_candlestick(df, ticker), key="t2_candle")
        else:
            _pc(fig_pred_overlay(df, ticker), key="t2_line")

        _pc(fig_technical(df, ticker), key="t2_tech")
        _pc(fig_corr_heatmap(df, ticker), key="t2_heatmap")

        st.markdown(f'<div class="sh">Latest Feature Snapshot</div>', unsafe_allow_html=True)
        if not df.empty:
            last = df.iloc[-1]
            snap = {c: round(float(last[c]), 4) if pd.notna(last.get(c)) else None
                    for c in TECH_COLS + QUANT_COLS if c in df.columns}
            st.dataframe(
                pd.DataFrame(list(snap.items()), columns=["Feature", "Value"]),
                use_container_width=True, hide_index=True, height=260,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 · Feature Explorer
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown(f'<div class="sh">🔭 Feature Explorer · {ticker}</div>',
                    unsafe_allow_html=True)

        all_cols = [c for c in df.columns if c != "Date" and df[c].notna().sum() > 5]
        groups   = {
            "📉 OHLCV":            [c for c in OHLCV_COLS   if c in all_cols],
            "📈 Technical":        [c for c in TECH_COLS    if c in all_cols],
            "🔢 Quant":            [c for c in QUANT_COLS   if c in all_cols],
            "🤖 Predictions":      [c for c in PRED_COLS    if c in all_cols],
            "⚡ Error 1":          [c for c in ERROR1_COLS  if c in all_cols],
            "🎯 Error 2":          [c for c in ERROR2_COLS  if c in all_cols],
        }
        flat  = [c for g in groups.values() for c in g]
        gtag  = {c: g for g, cs in groups.items() for c in cs}
        def label(c): return f"{gtag.get(c, '?')[2:]} · {c}"

        def_pri = ["Close"]      if "Close"      in flat else flat[:1]
        def_sec = ["Error2_Pct"] if "Error2_Pct" in flat else []

        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.68rem;'
                    f'color:{P["muted"]};margin-bottom:9px">Select columns for left (primary) '
                    f'and right (secondary) Y-axes. Mix prices, indicators and errors freely.</div>',
                    unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            primary = st.multiselect("Primary axis (left Y)", flat, default=def_pri,
                                      format_func=label, key="fe_pri")
        with fc2:
            secondary = st.multiselect("Secondary axis (right Y)",
                                        [c for c in flat if c not in primary],
                                        default=[c for c in def_sec if c not in primary],
                                        format_func=label, key="fe_sec")

        if primary or secondary:
            _pc(fig_feature_explorer(df, ticker, primary, secondary), key="fe_main")
        else:
            st.info("Select at least one column to plot.")

        # Quick presets
        st.markdown(f'<div class="sh">Quick Presets</div>', unsafe_allow_html=True)
        pcols = st.columns(5)
        presets = {
            "Price + BB":         (["Close", "BB_Upper", "BB_Lower"], []),
            "Pred vs Actual":     (["Actual_Price", "Today_Pred_Mean"], ["Error2_Pct"]),
            "E1 vs E2":           (["Error1_Pct"], ["Error2_Pct"]),
            "RSI + MACD":         (["RSI"], ["MACD", "MACD_Signal"]),
            "Beta + Sharpe":      (["Beta_60", "Sharpe_60"], ["Vol_Ratio"]),
        }
        for (name, (pc, sc)), col in zip(presets.items(), pcols):
            if col.button(name, key=f"pre_{name.replace(' ', '_')}"):
                ap = [c for c in pc if c in all_cols]
                as_ = [c for c in sc if c in all_cols]
                if ap or as_:
                    _pc(fig_feature_explorer(df, ticker, ap, as_),
                        key=f"fe_pre_{name.replace(' ', '_')}")
                else:
                    st.warning(f"Columns unavailable for preset: {name}")

        with st.expander("📐 Stats for Selected Features"):
            sel = list(dict.fromkeys(primary + secondary))
            if sel:
                st.dataframe(
                    df[[c for c in sel if c in df.columns]].describe().round(4),
                    use_container_width=True,
                )
            else:
                st.info("Select columns above.")

        with st.expander("📋 All Available Columns"):
            rows_g = []
            for g, cs in groups.items():
                for c in cs:
                    cd = df[c].dropna()
                    rows_g.append({
                        "Group": g, "Column": c, "Non-Null": len(cd),
                        "Mean": round(float(cd.mean()), 4) if not cd.empty else None,
                        "Min":  round(float(cd.min()),  4) if not cd.empty else None,
                        "Max":  round(float(cd.max()),  4) if not cd.empty else None,
                    })
            if rows_g:
                st.dataframe(pd.DataFrame(rows_g), use_container_width=True,
                             hide_index=True, height=340)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 · Model Health
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        render_model_health(ticker, models_dir)

        # Rolling accuracy chart here too — it's a direct health indicator
        st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
        _pc(fig_rolling_dir_acc(df, ticker, window=roll_acc_win), key="t4_rollacc")

        # Error 2 trend with calibration context
        if meta and meta.get("bias_offset") is not None:
            bias = float(meta.get("bias_offset", 0))
            n    = meta.get("n_history_rows", 0)
            st.markdown(f"""
<div style="background:{P['card']};border:1px solid {P['border']};border-radius:8px;
            padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:.72rem;
            color:{P['text']};margin:12px 0">
  <span style="color:{P['muted']};text-transform:uppercase;letter-spacing:.1em;font-size:.60rem">
    Calibration detail</span><br>
  Bias offset: <b style="color:{'#' + ('1A8F4C' if bias < 0 else 'D92B2B')}">{bias:+.4f} ₹</b>
  &nbsp;·&nbsp; based on {n} historical rows
  &nbsp;·&nbsp; {"active ✓" if meta.get('calibration_on') else "not yet active (need ≥ 10 rows)"}
</div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 · Multi-Ticker
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        if not compare:
            st.info("Select tickers in the sidebar **Compare** section.")
        else:
            metric = st.selectbox(
                "Metric to compare",
                ["Close", "Actual_Price", "Today_Pred_Mean", "Error1_Pct", "Error2_Pct"],
                index=0, key="mt_metric",
            )
            dfs_cmp: dict = {}
            prog = st.progress(0, text="Loading comparison tickers…")
            for i, t in enumerate(compare):
                d = load_ticker(t, data_dir)
                if d is not None and not d.empty:
                    dfs_cmp[t] = date_filter(d, start, end)
                prog.progress((i + 1) / max(len(compare), 1))
            prog.empty()

            if dfs_cmp:
                _pc(fig_multi_ticker(dfs_cmp, metric), key="t5_multi")

                st.markdown(f'<div class="sh">Error Distribution by Ticker</div>',
                            unsafe_allow_html=True)
                bc1, bc2 = st.columns(2)
                for ecol, lbl, color, col, bk in [
                    ("Error1_Pct", "E1 % (Stability)", P["orange"], bc1, "t5_e1_box"),
                    ("Error2_Pct", "E2 % (Accuracy)",  P["red"],    bc2, "t5_e2_box"),
                ]:
                    fb = go.Figure()
                    for t, d in dfs_cmp.items():
                        if ecol in d.columns:
                            fb.add_trace(go.Box(y=d[ecol].dropna(), name=t, boxmean=True,
                                                marker_color=color, line_color=color))
                    fb.update_layout(**_L(title=f"{lbl} Distribution", height=280, yaxis_title=lbl))
                    col.plotly_chart(fb, use_container_width=True, key=bk)

                # Heatmap of E2 across tickers
                e2_rows = {}
                for t, d in dfs_cmp.items():
                    if "Error2_Pct" in d.columns:
                        e2_rows[t] = d["Error2_Pct"].values[:50]  # last 50 days
                if e2_rows:
                    max_len = max(len(v) for v in e2_rows.values())
                    mat = np.full((len(e2_rows), max_len), np.nan)
                    for i, (t, vals) in enumerate(e2_rows.items()):
                        mat[i, :len(vals)] = vals
                    fh = go.Figure(go.Heatmap(
                        z=mat,
                        y=list(e2_rows.keys()),
                        colorscale=[[0, P["err_grn"]], [0.5, P["amber"]], [1, P["err_red"]]],
                        colorbar=dict(title="E2 %", thickness=11),
                        hovertemplate="Ticker: %{y}<br>Day: %{x}<br>E2: %{z:.2f}%<extra></extra>",
                    ))
                    fh.update_layout(**_L(title="Error 2 Heatmap — Last 50 Days",
                                          height=max(200, 40 * len(e2_rows) + 80),
                                          xaxis_title="Trading Day (recent →)",
                                          yaxis=dict(gridcolor=P["border"])))
                    _pc(fh, key="t5_e2_heatmap")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 · Portfolio
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        render_portfolio(data_dir, models_dir, start, end)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 7 · Data
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[7]:
        render_signed_error_table(df, ticker, show_verbose)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="margin-top:30px;padding:12px 0;border-top:1px solid {P['border']};
            font-family:'JetBrains Mono',monospace;font-size:.60rem;color:{P['muted']};
            text-align:center">
  NIFTY 100 Quant Research Terminal · v1.0.0 ·
  CNN-BiLSTM-Attention + Monte Carlo Dropout ·
  Data: Yahoo Finance · For research purposes only
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
