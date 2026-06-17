# 📈 Ticker-Teller v1.1

> **Neural Market Intelligence** — A full-stack stock forecasting system powered by a hybrid deep-learning
> model, real-time SSE streaming, **Autoencoder Deep Clustering (IDEC) risk analysis**, and comprehensive
> macro-economic overlays.

---

## Table of Contents

1. [Overview](#1-overview)
2. [What Changed from v1 → v1.1](#2-what-changed-from-v1.0--v1.1)
3. [System Architecture](#3-system-architecture)
4. [Feature Engineering — 31 Input Features](#4-feature-engineering--31-input-features)
5. [Prediction Model — CNN → BiLSTM → Attention](#5-prediction-model--cnn--bilstm--attention)
6. [Loss Function](#6-loss-function)
7. [Training Pipeline](#7-training-pipeline)
8. [Monte Carlo Uncertainty Estimation](#8-monte-carlo-uncertainty-estimation)
9. [Autoencoder Deep Clustering — IDEC Risk Analysis](#9-autoencoder-deep-clustering--idec-risk-analysis)
10. [Macro-Economic Panels](#10-macro-economic-panels)
11. [Frontend — React Dashboard](#11-frontend--react-dashboard)
12. [API Reference](#12-api-reference)
13. [Project Structure](#13-project-structure)
14. [Installation & Setup](#14-installation--setup)
15. [Configuration](#15-configuration)
16. [Tech Stack](#16-tech-stack)
17. [Disclaimer](#17-disclaimer)

---

## 1. Overview

Ticker-Teller is a research-grade, full-stack financial analysis application that combines classical
quantitative finance features with modern deep-learning architectures.

**What it does end-to-end:**

1. Fetches live OHLCV and macro data from Yahoo Finance for any ticker
2. Engineers **31 features** spanning price action, momentum indicators, bonds, and commodities
3. Runs **IDEC autoencoder deep clustering** across all 31 features to produce a data-driven, year-wise market risk assessment — replacing the older Fuzzy C-Means approach
4. Trains a hybrid **CNN → BiLSTM → Multi-Head Attention** model in real time, streaming every epoch to the browser via Server-Sent Events (SSE)
5. Produces a **Monte Carlo price forecast** with ±1σ confidence bands using MC-Dropout (80 passes)
6. Displays everything in a dark-mode React dashboard with interactive charts, macro panels, and a live training monitor

**Supported tickers:** Any Yahoo Finance symbol — US equities (`AAPL`), Indian stocks (`RELIANCE.NS`), crypto (`BTC-USD`), ETFs (`SPY`), indices (`^NSEI`), and more.

---

## 2. What Changed from v1.0 → v1.1

| Area | v1.0 | v1.1 |
|---|---|---|
| Risk Analysis | Fuzzy C-Means (FCM), 2 clusters, 3 hand-picked features | **IDEC Autoencoder Deep Clustering**, 4 clusters, all 31 features |
| Risk Features | Volatility, Beta, Drawdown only | Full 31-column feature matrix (price + macro) |
| Cluster Count | 2 (high / low) | **4** (graduated risk spectrum) |
| Representation | No latent space — raw 3D space | **Learned latent space** (AE encoder, dim 4–10) |
| Assignment | Hard (k-Means) OR soft (FCM, m=2) | **Soft** — Student-t distribution Q matrix |
| Optimisation | FCM: alternating U and C updates | **Joint** — reconstruction loss + KL divergence simultaneously |
| Frontend panel | `FcmRiskPanel.jsx` | `AdcRiskPanel.jsx` (same visual structure, 4-cluster volatility pills) |
| API key | `fcm_risk` | `adc_risk` |
| `stock.sh` | Vite output hidden in `.frontend.log` | **`tee`** — output goes to both log AND stdout so Codespaces auto-detects port 5173 |
| `vite.config.js` | No `host` setting | `host: '0.0.0.0'` — binds all interfaces for Codespaces forwarding |

---

## 3. System Architecture

```
Browser (React + Vite, port 5173)
        │
        │  HTTP  ─── GET /api/commodities, /api/gdp, /api/bonds, /api/ticker-info
        │  SSE   ─── POST /api/analyze   (epoch stream → result payload)
        │  (Vite dev proxy handles /api/* → localhost:8000)
        ▼
FastAPI Backend (Python 3.11+, port 8000)
        │
        ├── /api/analyze        POST — trains model, streams SSE events
        ├── /api/commodities    GET  — Holt-Winters commodity forecast
        ├── /api/gdp            GET  — ETF-based GDP proxy + forecast
        ├── /api/bonds          GET  — US Treasuries + ETF yield proxies
        ├── /api/ticker-info    GET  — Sector / Industry metadata
        └── /api/health         GET  — Version + device info
        │
        ├── src/data.py   ─── Data fetching, feature engineering,
        │                     IDEC deep clustering, Holt forecasting
        └── src/model.py  ─── PyTorch model definition, training loop,
        │                      MC forecast, evaluation metrics
        │
        └── Yahoo Finance (yfinance) ── OHLCV, ETF, futures, treasury data
```

**Concurrency model:** Two `ThreadPoolExecutor` pools — a 2-worker pool for CPU/GPU training
(prevents resource contention), and an 8-worker pool for concurrent data-fetch endpoints.
A `threading.Event` cancellation token propagates client-disconnect signals into the training loop,
stopping work the moment the browser tab is closed.

**In-memory cache:** All slow data fetches are cached per `"type:ticker:period"` key with a 1-hour TTL
(`_TTL = 3600s`), avoiding redundant Yahoo Finance calls within a session.

---

## 4. Feature Engineering — 31 Input Features

All 31 features are computed in `src/data.py` and then normalised to `[0, 1]` via `MinMaxScaler`
before entering the model. `ffill → bfill → fillna(0)` is applied to handle market closure gaps.

### Price & Technical Features (17)

| Feature | Description |
|---|---|
| `Open, High, Low, Close, Volume` | Raw OHLCV from Yahoo Finance |
| `SMA_10, SMA_50` | 10-day and 50-day Simple Moving Averages |
| `MACD, MACD_Signal` | 12/26 EMA difference and its 9-period EMA |
| `RSI` | 14-period Relative Strength Index (Wilder smoothing) |
| `ROC_10` | 10-day Rate of Change (%) |
| `Stoch_K` | 14-period Stochastic Oscillator %K |
| `BB_Upper, BB_Lower` | 20-day Bollinger Bands (±2σ around SMA_20) |
| `ATR` | 14-period Average True Range |
| `OBV_norm` | On-Balance Volume, z-scored over a 20-day window |
| `RVI` | Relative Volatility Index — RSI applied to daily range |

### Quantitative & Macro Features (14)

| Feature | Description |
|---|---|
| `Beta_60` | 60-day rolling beta vs SPY (OLS regression on log-returns) |
| `Alpha_60` | 60-day rolling Jensen's alpha (annualised) |
| `Momentum_12_1` | 12-month minus 1-month return — classic momentum factor |
| `Sharpe_60` | 60-day rolling annualised Sharpe ratio |
| `Vol_Ratio` | Short-term (10d) vs long-term (60d) volatility ratio |
| `Mkt_Return` | SPY daily log-return |
| `Bond_5Y, Bond_10Y` | US Treasury yields (^FVX, ^TNX via Yahoo Finance) |
| `Gold, Silver, Copper, Aluminium, Brent_Oil` | Commodity futures (GC=F, SI=F, HG=F, ALI=F, BZ=F) |
| `Buffett_Proxy` | SPY price / 252-day SMA × 100 — market valuation level |

---

## 5. Prediction Model — CNN → BiLSTM → Attention

The `HybridModel` in `src/model.py` takes a rolling window of `seq_length` days × 31 features
and predicts the next day's log-return (a scalar).

```
Input: [Batch, SeqLen, 31]
         │
         ▼
┌──────────────────────────────────────────────────┐
│  TemporalConvBlock  (Inception-style dual CNN)   │
│  ├── Conv1d(kernel=3) → BatchNorm → GELU         │
│  ├── Conv1d(kernel=5) → BatchNorm → GELU         │
│  ├── Concatenate → Conv1d(1×1) → BatchNorm → GELU│
│  └── Residual projection shortcut + Dropout      │
│  Output: [Batch, SeqLen, CNN_Channels]           │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│  Bidirectional LSTM                              │
│  hidden_size=128, num_layers=2                   │
│  Output: [Batch, SeqLen, 256]  (128 × 2 dirs)    │
└──────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│  Multi-Head Self-Attention                       │
│  d_model=256, num_heads=4                        │
│  Residual connection + LayerNorm                 │
│  Output: [Batch, SeqLen, 256]                    │
└──────────────────────────────────────────────────┘
         │
         ▼
  Soft temporal attention pooling
  (learned linear score per time step → weighted sum)
  Output: [Batch, 256]
         │
         ▼
┌──────────────────────────────────────────────────┐
│  MLP Head                                        │
│  Dropout → Linear(256→128) → GELU                │
│  → Dropout → Linear(128→1)                       │
└──────────────────────────────────────────────────┘
         │
         ▼
  Predicted log-return (scalar)
```

### Key Design Choices

| Choice | Rationale |
|---|---|
| **Log-return targets** | Stationary signal; avoids price-scale sensitivity across tickers |
| **Dual CNN kernels (3 + 5)** | Captures short and medium-range local patterns simultaneously |
| **Bidirectional LSTM** | Encodes both forward and backward temporal context |
| **Soft temporal pooling** | Learns *which* time steps in the window matter most |
| **Inception-style residual** | Avoids vanishing gradients in the CNN block |
| **MC-Dropout** | BatchNorm frozen, only Dropout stochastic — correct inference behaviour |

---

## 6. Loss Function

$L_{total} = L_{huber}(δ=0.01) + 0.4 × L_{directional}$

### Huber Loss
Robust to outlier log-returns from earnings surprises or flash crashes.
With δ=0.01 it operates in the linear regime for most typical daily moves
$(|return| > 0.01 → linear; |return| ≤ 0.01 → quadratic).$

### Directional Penalty
$L_{directional} = mean( relu( -sign(pred) * sign(target) ) )$

Fires a constant penalty of **1.0** whenever the model predicts the wrong
direction (up vs down), regardless of magnitude. The sign-based formulation
ensures the penalty is always meaningful — a naive `pred × target` product
produces near-zero values for small log-returns, making the weight useless
on low-volatility assets.

---

## 7. Training Pipeline

```
 1.  Fetch OHLCV history + macro data (bonds, commodities, market index)
 2.  Engineer all 31 features (data.py)
 3.  Run IDEC autoencoder deep clustering → adc_risk dict
 4.  Scale all features to [0,1] with MinMaxScaler
 5.  Build overlapping sequences of length seq_length
 6.  Time-ordered split: 70% train / 15% val / 15% test  (no shuffling)
 7.  DataLoaders (batch_size=32, pin_memory if CUDA available)
 8.  Train with AdamW + CosineAnnealingWarmRestarts scheduler
     ├── Gradient clipping: max_norm = 1.0
     ├── Early stopping (patience=10; best-val-loss weights restored)
     └── Every epoch → SSE event pushed to browser (loss, LR, step)
 9.  Evaluate on held-out test set
     └── Convert predicted log-returns → prices → MAE, RMSE, MAPE, Dir.Acc.
10.  Monte Carlo forecast for next N days (80 stochastic passes per step)
11.  Package full result → SSE "result" event → frontend renders
```

**No data leakage:** The `MinMaxScaler` is fitted only on the training split and
applied to val and test — refit at inference time is avoided.

---

## 8. Monte Carlo Uncertainty Estimation

The model is put into `eval()` mode to freeze BatchNorm running statistics,
then all `nn.Dropout` layers are explicitly switched back to `train()` mode:

```python
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()
```

**Why not just call `model.train()`?** That would re-enable BatchNorm's
batch-statistics mode. At inference time the rolling forecast window has
batch size = 1, making batch statistics meaningless and corrupting the
layer's output. The above approach is the correct MC-Dropout technique.

**80 stochastic forward passes** are run per forecast step. The mean
log-return predicts the next price; the standard deviation gives the ±1σ
confidence band. The forecast rolls forward autoregressively — each predicted
close becomes part of the next input window.

---

## 9. Autoencoder Deep Clustering — IDEC Risk Analysis

This is the core algorithmic upgrade in v1.1. The section explains the
theoretical foundations and the exact implementation in `src/data.py`.

---

### 9.1 Why Autoencoders for Risk Clustering?

The previous Fuzzy C-Means (FCM) approach had three structural weaknesses:

1. **Manual feature selection** — only 3 of 31 features were used (volatility,
   beta, drawdown). Macro signals (bond yields, commodities, Buffett proxy)
   were entirely ignored in the risk model.

2. **Fixed membership function** — FCM's fuzzifier parameter `m=2` defines
   the shape of cluster boundaries rigidly. The data never gets to influence
   what "high risk" looks like.

3. **Curse of dimensionality** — clustering in raw high-dimensional space
   produces unreliable results as distances become increasingly uniform.
   An autoencoder first compresses the data into a meaningful low-dimensional
   latent space before clustering occurs.

An **autoencoder** addresses all three:

- It reads **all 31 features** simultaneously
- It learns which combinations of features actually separate market regimes
- It creates a **latent space** where similar market conditions are nearby
  and different ones are far apart — making clustering far more reliable

---

### 9.2 What Is an Autoencoder?

An autoencoder is a neural network trained to compress input data into a
low-dimensional representation (the **latent space** or **embedding**) and
then reconstruct the original input from that representation.

```
Input x  ──>  Encoder f  ──>  Latent z  ──>  Decoder g  ──>  Reconstruction x̂
              (compress)                      (expand)
```

$Loss = ||x - x̂||²$   (MSE — how well does x̂ match x?)


The encoder is forced to learn which features matter most, because the latent
space `z` has far fewer dimensions than `x`. Irrelevant features are naturally
discarded. Related features are fused.

In our implementation:

```
Encoder: d → 256 → 128 → 64 → latent_dim   (BatchNorm + GELU activations)
Decoder: latent_dim → 64 → 128 → 256 → d   (mirrored, symmetric)
```

Where `d = 31` (number of features) and `latent_dim ∈ [4, 10]` (chosen
adaptively as `min(10, max(4, d // 4))`).

---

### 9.3 From AE to Deep Clustering — the DEC / IDEC Framework

Training just the autoencoder gives a good latent representation, but does
not optimise it for clustering. **DEC (Deep Embedded Clustering, Xie et al.
2016)** and its improvement **IDEC (Improved DEC, Guo et al. 2017)** add a
clustering objective on top of the reconstruction objective.

The key idea: define a probability distribution over cluster assignments
in the latent space, then train the network to make high-confidence
assignments even more confident (self-reinforcing).

---

### 9.4 The Student-t Soft Assignment (Q Distribution)

For each data point $z_i$ in the latent space and each cluster centre $μ_j$,
the **soft assignment** $q_{i,j}$ measures how likely point $i$ belongs to
cluster $j$. It uses a **Student-t distribution kernel** (same as t-SNE):

$q_{i,j} = \frac{(1 + ||z_i - μ_j||² / α)^{-(α+1)/2}}{Σ_j' (1 + ||z_i - μ_j'||² / α)^{-(α+1)/2}}$

- `α = 1` (degrees of freedom — standard choice)
- The result is normalised so all $q_{i,j}$ sum to 1 for each point `i`
- Heavy tails of the Student-t distribution allow points far from all centres
  to still receive meaningful (not near-zero) assignments

This is **soft clustering** — every trading day gets a continuous
membership score in [0, 1] for *each* of the 4 clusters simultaneously.

---

### 9.5 The Target Distribution (P Distribution)

IDEC introduces a **target distribution** `P` that sharpens the soft
assignments — it takes the current `Q` and amplifies high-confidence
assignments while suppressing uncertain ones:
         
$p_{i,j} = \frac{q²_{i,j} / f_j}{Σ_j' (q²_{i,j'} / f_j')}$

Where $f_j = Σ_i q_{i,j}$ is the soft cluster size (prevents large
clusters from dominating the loss).

**Intuition:** If a point is already 80% assigned to cluster 2, `P` makes
it look like 95%. If a point is uncertain (25% across 4 clusters), `P`
keeps it uncertain. This iteratively pulls points toward their dominant
cluster, creating cleaner separation.

---

### 9.6 The Joint IDEC Loss

IDEC optimises both objectives simultaneously:

$L_{total} = λ · L_{rec}  +  (1 - λ) · KL(P ‖ Q)$


Where:
- $L_{rec} = MSE(x̂, x)$ — reconstruction loss, keeps the embedding
  meaningful and prevents cluster collapse
- $KL(P ‖ Q) = Σ_i Σ_j p_{i,j} · log(p_{i,j} / q_{i,j})$ — KL
  divergence from target P to soft assignment Q, pushes the model
  to match high-confidence assignments
- `λ = 0.1` — reconstruction has a smaller weight; clustering drives
  most of the gradient

This is a **simultaneous** optimisation strategy (per the survey taxonomy):
both the autoencoder weights and the cluster centres $μ_j$ are updated
together in every gradient step.

---

### 9.7 The Three Training Phases

```
╔═══════════════════════════════════════════════════════════════╗
║  PHASE 1 — AE Pre-training                                    ║
║                                                               ║
║  Train the autoencoder on MSE reconstruction loss only.       ║
║  No clustering yet. This gives the encoder a meaningful,      ║
║  general-purpose latent space to start from.                  ║
║                                                               ║
║  Loss: L = MSE(x̂, x)                                          ║
║  Epochs: min(60, max(20, n_samples // 10))                    ║
╚═══════════════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════════════╗
║  PHASE 2 — k-Means Initialisation                            ║
║                                                              ║
║  Encode the full dataset → Z = encoder(X)                    ║
║  Run k-Means (k=4, n_init=20) in the latent space Z.         ║
║  Use the resulting 4 centroids as initial μ values.          ║
║                                                              ║
║  Why k-Means init? Random init of cluster centres leads      ║
║  to poor convergence. Starting from k-Means gives IDEC       ║
║  a sensible starting point before the joint optimisation.    ║
╚══════════════════════════════════════════════════════════════╝
                        │
                        ▼
╔══════════════════════════════════════════════════════════════╗
║  PHASE 3 — Joint IDEC Optimisation                           ║
║                                                              ║
║  Every 5 epochs: recompute Q and update frozen target P.     ║
║  Every batch: compute L_total = λ·L_rec + (1-λ)·KL(P‖Q)      ║
║              backpropagate through AE weights AND μ.         ║
║                                                              ║
║  Both the autoencoder and cluster centres are learnable      ║
║  parameters updated simultaneously.                          ║
║  Epochs: min(80, max(30, n_samples // 8))                    ║
╚══════════════════════════════════════════════════════════════╝
```

---

### 9.8 High-Risk Cluster Identification

After training, we need to label which of the 4 clusters represents
"high risk". We do this with **volatility-based identification** —
no human labelling required:

```python
# 21-day annualised volatility for every trading day
vol_21 = log_returns.rolling(21).std() * sqrt(252)

# Mean volatility of all days assigned to each cluster
cluster_mean_vol[k] = mean(vol_21[labels == k])

# The cluster with highest mean volatility is high risk
high_risk_cluster = argmax(cluster_mean_vol)
```

This is automatic and data-driven: if the market regime shifts over
time and volatility patterns change, the clustering adapts at the
next run.

---

### 9.9 The Risk Score

Each trading day's **risk score** is its soft membership `q_{i, high_risk}`
— a continuous value in `[0, 1]`.

- Score **≥ 0.65** → 🔴 **High Risk**
- Score **≥ 0.40** → 🟡 **Moderate Risk**
- Score **< 0.40** → 🟢 **Low Risk**

Year-wise risk evolution:
```
yearly[year] = mean(q_{i, high_risk}  for all trading days i in year)
current_score = q_{last_day, high_risk}
```

The ADC panel in the frontend shows:
- Current score + risk label (colour-coded)
- Overall mean across all years
- 4-cluster annualised volatility pills (sorted low → high)
- Year-wise bar chart (each bar coloured by its risk level)
- Year-wise table with exact scores
- Full methodology explanation inline

---

### 9.10 ADC vs FCM — Side-by-Side Comparison

| Property | FCM (v1.0) | IDEC / ADC (v1.1) |
|---|---|---|
| Features used | 3 (volatility, beta, drawdown) | All 31 |
| Cluster count | 2 | 4 |
| Latent space | None (raw feature space) | Learned by autoencoder |
| Cluster shape | Spherical (Euclidean distance) | Arbitrary (learned geometry) |
| Assignment type | Soft (fuzzy membership) | Soft (Student-t probability) |
| Optimisation | Alternating U and C updates | Joint AE + clustering gradient |
| High-risk labelling | Higher volatility centroid | Highest mean 21d ann. vol. cluster |
| Adapts to data | Limited (fixed fuzzifier) | Yes (learned encoder transforms) |
| Dependencies | Pure NumPy | PyTorch + scikit-learn (KMeans init) |
| Theoretical basis | Bezdek 1981 | DEC (Xie 2016) + IDEC (Guo 2017) |

---

### 9.11 Why IDEC Specifically, Not Plain DEC?

Original DEC (Xie et al. 2016) sets `λ = 0` — it discards the reconstruction
loss entirely after pre-training, using only the KL divergence. This causes
**embedding distortion**: the latent space can collapse (all points pulled to
their cluster centres), destroying the meaningful geometry the autoencoder
learned.

IDEC (Guo et al. 2017) keeps `λ > 0` throughout clustering — the
reconstruction loss acts as a **regulariser**, preventing collapse and
preserving local structure. In practice this consistently outperforms DEC
on accuracy metrics while keeping the embedding interpretable.

In our implementation: `λ = 0.1` (clustering dominates but reconstruction
is never fully ignored).

---

## 10. Macro-Economic Panels

### Commodities Panel

Fetches 5 commodity futures via yfinance (GC=F, SI=F, HG=F, ALI=F, BZ=F)
and applies **Holt's Linear (Double) Exponential Smoothing** (α=0.3, β=0.1)
to produce a multi-day forecast with ±1σ confidence bands.

The trend is initialised using the global slope `(s[-1] - s[0]) / (n-1)`
rather than a naive single-step initialisation, which produces stable
and smooth trend estimates even on noisy commodity series.

Each commodity card includes a 🏭 **Industries** button that shows
sector-sensitivity impact analysis for the 5 most relevant industries
(e.g. Airlines for Brent Oil, Solar/EV for Silver). Impact prices are
clearly labelled as **illustrative simulations** — not live quotes.

### Bond Yields Panel

- **US 5Y / 10Y**: fetched directly (^FVX, ^TNX)
- **India, Germany, UK, Japan**: ETF-anchored synthetic yields derived from
  ETF log-return drift normalised and anchored to fixed reference rates
  (e.g. India 6.80%, Germany 2.45%) with US 10Y sensitivity factor
- **Yield curve inversion alert**: fires when US 10Y < US 5Y

### GDP Proxy Panel

Annualised 252-day ETF returns (SPY, INDA, FXI, EZU, EWJ, EWU, EWZ) used
as directional economic momentum proxies. IMF consensus estimates shown
alongside. Holt smoothing for 30-day directional forecast.

> ⚠ ETF returns ≠ GDP. SPY +36% does not mean US GDP grew 36%.
> ETF values are directional proxies only.

### Correlations Panel

Pearson correlation between each of the 31 features and the Close price,
computed on the full engineered dataframe post-analysis.

---

## 11. Frontend — React Dashboard

Built with React 18 + Vite 5 + Recharts 2. Single-page application with
tab-based navigation and hamburger menu for macro panels.

### Tabs

| Tab | Contents |
|---|---|
| **Price & Forecast** | Historical OHLCV chart (BB bands, RSI, volume), test-set predictions, MC forecast with ±1σ bands, range buttons (1M/3M/6M/1Y/2Y/ALL) |
| **Model Performance** | ADC Risk panel, metric cards (MAE/RMSE/MAPE/Dir.Acc.), training loss curves, data split indicators |
| **More Analysis** (☰) | Bond yields, commodity forecasts, GDP proxies, feature correlations |

### Live Training via SSE

The `POST /api/analyze` endpoint returns a `text/event-stream` response.
`api.js` consumes it with the browser's `EventSource`-compatible fetch:

```
SSE event: "status"  → status bar update
SSE event: "epoch"   → training progress panel, loss sparkline
SSE event: "result"  → full payload, all charts rendered
SSE event: "error"   → error display with traceback
```

### ADC Risk Panel (`AdcRiskPanel.jsx`)

Displays:
- **Current risk score** — last trading day's high-risk soft membership
- **Overall mean** — mean score across all years
- **4-cluster volatility pills** — annualised vol % at each cluster centre,
  sorted low→high, colour-coded (green → blue → amber → red)
- **Year-wise bar chart** — each bar coloured by its risk level (green/amber/red)
- **Year table** — exact scores + risk label per year
- **Methodology card** — inline explanation of the IDEC phases

---

## 12. API Reference

### `POST /api/analyze`

Streams Server-Sent Events. Request body:

```json
{
  "ticker":        "AAPL",
  "period":        730,
  "seq_length":    60,
  "forecast_days": 5,
  "epochs":        50,
  "patience":      10,
  "hidden_size":   128,
  "cnn_channels":  64,
  "num_heads":     4,
  "dropout":       0.2
}
```

SSE event types:

| Type | Key fields |
|---|---|
| `status` | `message: str` |
| `epoch` | `epoch, total, train_loss, val_loss, lr` |
| `result` | See schema below |
| `error` | `message, traceback` |

Result payload (abbreviated):

```json
{
  "ticker": "AAPL",
  "currency": "$",
  "device": "cpu",
  "features": { "total": 31, "price": 17, "macro": 14 },
  "splits": { "n_train": 420, "n_val": 90, "n_test": 90 },
  "adc_risk": {
    "yearly":               { "2020": 0.71, "2021": 0.44 },
    "overall_mean":         0.56,
    "current_score":        0.68,
    "risk_label":           "High Risk",
    "cluster_centers":      [11.2, 19.7, 33.4, 58.1],
    "high_risk_centre_idx": 3,
    "n_clusters":           4,
    "latent_dim":           10,
    "error":                ""
  },
  "summary": {
    "last_price": 189.30,
    "next_price": 192.14,
    "pct_change": 1.50,
    "signal": "BUY"
  },
  "metrics": {
    "mae": 3.42, "rmse": 4.87,
    "mape": 1.92, "directional_accuracy": 58.3
  },
  "charts": {
    "historical": [...],
    "test":       [...],
    "forecast":   [...],
    "loss":       [...],
    "correlations": [...]
  }
}
```

### `GET /api/commodities`

| Param | Type | Default |
|---|---|---|
| `period` | int | 730 |
| `forecast_days` | int | 10 |

### `GET /api/gdp`

| Param | Type | Default |
|---|---|---|
| `period` | int | 730 |
| `forecast_days` | int | 30 |

### `GET /api/bonds`

| Param | Type | Default |
|---|---|---|
| `period` | int | 730 |

### `GET /api/ticker-info`

| Param | Type |
|---|---|
| `ticker` | str |

### `GET /api/health`

Returns `{ "status": "ok", "version": "4.0.0", "device": "cpu|cuda", "features": 31 }`.

---

## 13. Project Structure

```
stockmarket_dl_adc/
│
├── stock.sh                      # One-command launcher (start/stop/restart/logs)
│                                 # Fixed: --host 0.0.0.0, tee stdout, 90s wait
│
├── backend/
│   ├── main.py                   # FastAPI app — all endpoints, SSE stream,
│   │                             # ThreadPoolExecutor, cancellation token
│   └── src/
│       ├── __init__.py           # Package version
│       ├── data.py               # Feature engineering, data fetching,
│       │                         # IDEC autoencoder deep clustering,
│       │                         # Holt forecasting, GDP/bond proxies
│       └── model.py              # HybridModel (CNN→BiLSTM→Attention),
│                                 # CombinedLoss, training loop, MC forecast
│
└── frontend/
    ├── index.html                # Vite entry point
    ├── package.json              # npm dependencies
    ├── package-lock.json         # Lockfile
    ├── vite.config.js            # Dev proxy /api/* → :8000, host: 0.0.0.0
    └── src/
        ├── main.jsx              # React root mount
        ├── App.jsx               # Root component, SSE state machine,
        │                         # tab routing, hamburger nav
        ├── api.js                # fetch wrappers + SSE stream client
        ├── styles/
        │   └── globals.css       # Full design system — tokens, layout,
        │                         # components, dark-mode, animations
        └── components/
            ├── Sidebar.jsx           # Config panel, sliders, run button
            ├── TrainingProgress.jsx  # Live epoch stats + loss sparkline
            ├── MetricCards.jsx       # MAE/RMSE/MAPE/Dir.Acc. KPI cards
            ├── PriceChart.jsx        # Price/BB/RSI/Volume + range buttons
            ├── LossAndMacdCharts.jsx # Train+val loss curve + MACD chart
            ├── AdcRiskPanel.jsx      # IDEC risk panel (replaces FcmRiskPanel)
            ├── CommodityPanel.jsx    # Commodity cards + Holt forecast
            ├── GdpPanel.jsx          # GDP proxy cards + ETF trend + IMF bars
            ├── BondPanel.jsx         # Yield cards + history + spread chart
            └── CorrelationAndAbout.jsx  # Correlation chart + About modal
```

---

## 14. Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- (Optional) CUDA-capable GPU with PyTorch CUDA build

### Quickstart (Linux / macOS / GitHub Codespaces)

```bash
# 1. Clone / unzip and enter the project
cd stockmarket_dl_adc

# 2. Create Python virtual environment and install dependencies
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..

# 3. Run everything with the launcher script
chmod +x stock.sh
./stock.sh
```

The script will:
- Start the FastAPI backend on port 8000
- Run `npm install` if `node_modules` is missing
- Start the Vite dev server on port 5173
- Print both URLs when ready

### GitHub Codespaces

After `./stock.sh` shows `✔ All systems up!`:

1. Open the **Ports** tab in the VS Code bottom panel
2. Port **5173** should be listed and auto-forwarded (Vite now prints to stdout)
3. If it's not listed: click **Add Port → 5173**, right-click → **Port Visibility → Public**
4. Click the 🌐 globe icon next to 5173

> The app is on **port 5173** (React). Port 8000 is the raw API — it shows
> `{"detail":"Not Found"}` on the root path, which is normal.

### Manual Start (without stock.sh)

```bash
# Terminal 1 — Backend
cd backend
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

### GPU Support

```bash
# CUDA 12.x
pip install torch>=2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

The backend auto-detects CUDA via `torch.cuda.is_available()` and reports
the device in `/api/health` and the `result` payload.

---

## 15. Configuration

All parameters are set from the **Sidebar** in the UI. Defaults:

| Parameter | Default | Range | Description |
|---|---|---|---|
| `ticker` | AAPL | Any Yahoo Finance symbol | Stock ticker |
| `period` | 730 | 365 – 1825 days | Days of price history to download |
| `seq_length` | 60 | 10 – 120 | Lookback window (trading days) per sample |
| `forecast_days` | 5 | 1 – 30 | Future business days to forecast |
| `epochs` | 50 | 5 – 200 | Max training epochs |
| `patience` | 10 | 3 – 30 | Early-stopping patience |
| `hidden_size` | 128 | 32 / 64 / 128 / 256 | BiLSTM hidden units |
| `cnn_channels` | 64 | 32 / 64 / 128 | CNN output channels |
| `num_heads` | 4 | 2 / 4 / 8 | Multi-head attention heads |
| `dropout` | 0.2 | 0.0 – 0.5 | Dropout rate (also sets MC band width) |

### Currency Auto-Detection

| Ticker suffix | Symbol |
|---|---|
| `.NS`, `.BO` | ₹ Indian Rupee |
| `.L` | £ British Pound |
| `.HK` | HK$ Hong Kong Dollar |
| `.DE`, `.PA`, `.EU`, `.AS`, `.MI` | € Euro |
| (default) | $ US Dollar |

---

## 16. Tech Stack

### Backend

| Library | Version | Purpose |
|---|---|---|
| FastAPI | ≥0.111 | Async REST + SSE streaming |
| PyTorch | ≥2.3 | HybridModel + IDEC autoencoder |
| yfinance | ≥0.2.40 | Live market data |
| pandas | ≥2.2 | Time-series manipulation |
| NumPy | ≥1.26 | Numerical computing |
| scikit-learn | ≥1.5 | MinMaxScaler + KMeans (IDEC Phase 2) |
| Pydantic | ≥2.7 | Request validation |
| uvicorn | ≥0.30 | ASGI server |
| httpx | ≥0.27 | HTTP client (yfinance dependency) |

### Frontend

| Library | Version | Purpose |
|---|---|---|
| React | 18.3 | UI framework |
| Vite | 5.3 | Dev server + bundler |
| Recharts | 2.12 | All charts |
| Lucide React | 0.383 | Icons |
| Axios | 1.7 | HTTP client |
| clsx | 2.1 | Conditional class names |

---

## 17. Disclaimer

> ⚠ **This application is for educational and research purposes only.**
> Nothing in Ticker-Teller constitutes financial advice. All forecasts
> carry significant model risk and uncertainty. Do not use as the sole
> basis for any financial or investment decision.

**Technical limitations to be aware of:**

- **Yahoo Finance data quality**: gaps, split/dividend adjustments, and
  occasional erroneous ticks are possible. `ffill/bfill/fillna(0)` are
  applied as fallbacks but cannot guarantee clean data for all tickers.
- **GDP panel**: ETF returns are directional proxies — not official GDP
  statistics. IMF consensus estimates are shown alongside for comparison.
- **Bond yield proxies**: Non-US yields (India, Germany, UK, Japan) are
  synthetic, anchored to fixed reference rates. They are indicative only.
- **Industry impact modal**: All impact prices are illustrative simulations
  seeded by date — not live market quotes.
- **MC-Dropout calibration**: Bands reflect parameter uncertainty (epistemic)
  only, not data uncertainty (aleatoric).
- **No model persistence**: Each run re-trains from scratch. Add
  checkpointing for production use.
- **In-memory cache**: Lost on restart. Use Redis for multi-worker deployments.
- **IDEC runtime**: The three-phase IDEC training adds ~5–15 seconds to the
  analysis pipeline depending on dataset size and CPU speed. On GPU the
  overhead is negligible.

---

*Built with PyTorch · FastAPI · React · Recharts*
*IDEC implementation based on: Xie et al. 2016 (DEC) + Guo et al. 2017 (IDEC)*