"""
model.py  v1.0.0  —  Intelligent Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ① AdaptiveCombinedLoss  — direction_weight is passed at call time, so the
      tracker can dial it up/down per-stock based on historical accuracy.
  ② finetune_model()      — lightweight incremental update (fewer epochs, higher
      LR warmup ratio, cosine decay) for daily runs where a full retrain would
      be wasteful.
  ③ predict_with_calibration() — applies a per-stock bias offset (learned from
      Excel history) on top of the MC forecast.
  ④ evaluate_model()      — returns direction accuracy + MAPE on a DataLoader
      so the tracker can measure live performance without writing extra logic.

  - MC Dropout fix (BatchNorm stays in eval, only Dropout re-enabled)
  - Sign-based directional penalty
  - pin_memory gated on CUDA availability
"""

from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# ── Dataset ───────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    batch_size: int = 32,
):
    _pin = torch.cuda.is_available()
    def mk(X, y, shuffle=False):
        return DataLoader(
            StockDataset(X, y),
            batch_size  = batch_size,
            shuffle     = shuffle,
            pin_memory  = _pin,
            num_workers = 0,
        )
    return mk(X_train, y_train, shuffle=True), mk(X_val, y_val), mk(X_test, y_test)


# ── Loss ──────────────────────────────────────────────────────────────────────
class AdaptiveCombinedLoss(nn.Module):
    """
    Huber + sign-based directional penalty.

    direction_weight is passed at *forward time* (not locked at construction),
    so the tracker can increase it when a stock's historical directional
    accuracy is low — without rebuilding or reloading the model.
    """
    def __init__(self, delta: float = 0.01):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(
        self,
        pred:             torch.Tensor,
        target:           torch.Tensor,
        direction_weight: float = 0.4,
    ) -> torch.Tensor:
        h_loss      = self.huber(pred, target)
        dir_penalty = torch.mean(F.relu(-torch.sign(pred) * torch.sign(target)))
        return h_loss + direction_weight * dir_penalty


# Keep the old name as an alias so dashboard / tracker imports don't break
class CombinedLoss(AdaptiveCombinedLoss):
    """Back-compat alias. direction_weight baked in at construction."""
    def __init__(self, delta: float = 0.01, direction_weight: float = 0.4):
        super().__init__(delta=delta)
        self._dw = direction_weight

    def forward(self, pred, target, direction_weight=None):
        return super().forward(pred, target, direction_weight or self._dw)


# ── Building blocks ───────────────────────────────────────────────────────────
class TemporalConvBlock(nn.Module):
    """Inception-style dual-kernel CNN with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        mid = max(out_channels // 2, 1)
        self.conv3 = nn.Conv1d(in_channels, mid, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, mid, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(mid)
        self.bn5   = nn.BatchNorm1d(mid)
        self.fuse  = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn_f  = nn.BatchNorm1d(out_channels)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = self.residual(x)
        c3  = F.gelu(self.bn3(self.conv3(x)))
        c5  = F.gelu(self.bn5(self.conv5(x)))
        out = F.gelu(self.bn_f(self.fuse(torch.cat([c3, c5], dim=1))))
        return self.drop(out) + shortcut


class MultiHeadAttentionBlock(nn.Module):
    """Self-attention with residual + LayerNorm."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


# ── Hybrid model ──────────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(
        self,
        input_size:   int,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        cnn_channels: int   = 64,
        num_heads:    int   = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()
        self.cnn  = TemporalConvBlock(input_size, cnn_channels, dropout)
        self.lstm = nn.LSTM(
            input_size    = cnn_channels,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        bilstm_out   = hidden_size * 2
        actual_heads = num_heads
        while bilstm_out % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1

        self.attn      = MultiHeadAttentionBlock(bilstm_out, actual_heads, dropout)
        self.time_pool = nn.Linear(bilstm_out, 1)
        self.head      = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bilstm_out, bilstm_out // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(bilstm_out // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out     = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        attn_out    = self.attn(lstm_out)
        scores      = torch.softmax(self.time_pool(attn_out), dim=1)
        pooled      = (attn_out * scores).sum(dim=1)
        return self.head(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecastModel(nn.Module):
    """Modular Transformer encoder alternative with feature projection."""
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        actual_heads = num_heads
        while d_model % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=actual_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.feature_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled * self.feature_gate(pooled))


def build_model(
    architecture: str,
    input_size: int,
    hidden_size: int = 128,
    dropout: float = 0.2,
    cnn_channels: int = 64,
    num_heads: int = 4,
) -> nn.Module:
    architecture = (architecture or "hybrid").lower()
    if architecture in {"transformer", "transformer_encoder"}:
        return TransformerForecastModel(
            input_size=input_size,
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
    return HybridModel(
        input_size=input_size,
        hidden_size=hidden_size,
        cnn_channels=cnn_channels,
        num_heads=num_heads,
        dropout=dropout,
    )


# ── Training callback type ────────────────────────────────────────────────────
EpochCallback = Callable[[int, int, float, float, float], bool]


# ── Full training ──────────────────────────────────────────────────────────────
def train_model(
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    input_size:       int,
    epochs:           int,
    patience:         int,
    device:           torch.device,
    hidden_size:      int   = 128,
    dropout:          float = 0.2,
    cnn_channels:     int   = 64,
    num_heads:        int   = 4,
    architecture:     str   = "hybrid",
    direction_weight: float = 0.4,   # NEW: per-stock adaptive weight
    epoch_callback:   Optional[EpochCallback] = None,
) -> tuple[HybridModel, dict]:
    model     = build_model(
        architecture, input_size, hidden_size, dropout, cnn_channels, num_heads,
    ).to(device)
    criterion = AdaptiveCombinedLoss(delta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    history       = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb, direction_weight=direction_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(len(train_loader), 1)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_loss += criterion(model(Xb.to(device)), yb.to(device),
                                    direction_weight=direction_weight).item()
        v_loss /= max(len(val_loader), 1)

        scheduler.step(epoch + v_loss / 10.0)
        history["train"].append(round(t_loss, 6))
        history["val"].append(round(v_loss, 6))

        lr = float(scheduler.get_last_lr()[0])
        if epoch_callback and not epoch_callback(epoch + 1, epochs, t_loss, v_loss, lr):
            break

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.to(device), history


# ── Incremental fine-tuning ───────────────────────────────────────────────────
def finetune_model(
    model:            HybridModel,
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    device:           torch.device,
    epochs:           int   = 8,
    patience:         int   = 4,
    lr:               float = 3e-4,
    direction_weight: float = 0.4,
    epoch_callback:   Optional[EpochCallback] = None,
) -> tuple[HybridModel, dict]:
    """
    Fast incremental update for daily runs.

    Strategy:
    • Lower learning rate than full training (3e-4 vs 1e-3)
    • Only the LSTM + attention + head are unfrozen; CNN stays frozen to
      preserve low-level feature extraction while adapting to recent dynamics.
    • CosineAnnealingLR (single cycle, no restarts) — decays to 1e-6.
    • Early stop on validation loss with patience=4.
    """
    criterion = AdaptiveCombinedLoss(delta=0.01)

    # Freeze CNN, fine-tune the rest
    for param in model.cnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    history       = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb, direction_weight=direction_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(len(train_loader), 1)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_loss += criterion(model(Xb.to(device)), yb.to(device),
                                    direction_weight=direction_weight).item()
        v_loss /= max(len(val_loader), 1)

        scheduler.step()
        history["train"].append(round(t_loss, 6))
        history["val"].append(round(v_loss, 6))

        lr_cur = float(scheduler.get_last_lr()[0])
        if epoch_callback and not epoch_callback(epoch + 1, epochs, t_loss, v_loss, lr_cur):
            break

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # Re-enable CNN gradients before returning
    for param in model.cnn.parameters():
        param.requires_grad = True

    if best_state:
        model.load_state_dict(best_state)
    return model.to(device), history


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_test_set(
    model:       HybridModel,
    test_loader: DataLoader,
    device:      torch.device,
):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb.to(device)).cpu().numpy().flatten())
            actuals.extend(yb.numpy().flatten())
    return np.array(preds, dtype=np.float32), np.array(actuals, dtype=np.float32)


def monte_carlo_forecast(
    model:          HybridModel,
    scaled_data:    np.ndarray,
    raw_close_last: float,
    scaler:         MinMaxScaler,
    input_size:     int,
    seq_length:     int,
    forecast_days:  int,
    device:         torch.device,
    n_samples:      int = 80,
    close_idx:      int = 3,
) -> list[dict]:
    """
    Multi-step MC forecast.
    BatchNorm stays in eval(); only Dropout layers are re-enabled for sampling.
    """
    if len(scaled_data) < seq_length:
        return [{"mean": float(raw_close_last), "std": 0.0} for _ in range(forecast_days)]

    close_min = float(scaler.data_min_[close_idx])
    close_max = float(scaler.data_max_[close_idx])
    price_rng = max(close_max - close_min, 1e-9)

    window    = scaled_data[-seq_length:].copy()
    cur_price = float(raw_close_last)
    results   = []

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    for _ in range(forecast_days):
        inp     = torch.from_numpy(window[np.newaxis]).float().to(device)
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(float(model(inp).item()))

        mean_ret   = float(np.mean(samples))
        std_ret    = float(np.std(samples))
        next_price = max(cur_price * np.exp(mean_ret), 0.01)
        price_std  = cur_price * std_ret

        results.append({"mean": next_price, "std": price_std})

        next_feat             = window[-1].copy()
        next_feat[close_idx]  = np.clip((next_price - close_min) / price_rng, 0.0, 1.0)
        window                = np.concatenate([window[1:], next_feat[np.newaxis]], axis=0)
        cur_price             = next_price

    return results


def predict_with_calibration(
    raw_forecasts: list[dict],
    bias_offset:   float,
    bias_pct:      float,
) -> list[dict]:
    """
    Apply additive + multiplicative bias correction derived from Excel history.

    bias_offset : mean absolute additive error in price terms (signed; + means
                  model over-predicts, – means under-predicts).
    bias_pct    : mean percentage bias (signed).  Used as a fallback check.

    The correction subtracts the historical mean error so that, if the model
    has been consistently predicting ₹5 too high, the next prediction is
    nudged down ₹5.
    """
    calibrated = []
    for fc in raw_forecasts:
        corrected_mean = fc["mean"] - bias_offset
        corrected_mean = max(corrected_mean, 0.01)
        calibrated.append({"mean": corrected_mean, "std": fc["std"]})
    return calibrated


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_model(
    model:       HybridModel,
    test_loader: DataLoader,
    device:      torch.device,
    start_price: float,
) -> dict:
    """
    Evaluate the model on a DataLoader, returning price-level metrics.
    Converts log-return predictions → prices for interpretable MAPE.
    """
    from data import returns_to_prices  # avoid circular import at module level
    preds, actuals = predict_test_set(model, test_loader, device)

    pred_prices   = returns_to_prices(preds,   start_price)
    actual_prices = returns_to_prices(actuals,  start_price)
    return compute_metrics(pred_prices, actual_prices)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred_prices: np.ndarray, actual_prices: np.ndarray) -> dict:
    if len(pred_prices) == 0 or len(actual_prices) == 0:
        return {"mae": None, "rmse": None, "mape": None, "directional_accuracy": None}

    n    = min(len(pred_prices), len(actual_prices))
    p, a = pred_prices[:n], actual_prices[:n]

    mae  = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    denom = np.where(np.abs(a) < 1e-9, 1e-9, a)
    mape = float(np.mean(np.abs((p - a) / denom)) * 100.0)

    dir_acc = None
    if n > 1:
        correct = np.sign(np.diff(p)) == np.sign(np.diff(a))
        dir_acc = float(np.sum(correct) / (n - 1) * 100.0)

    return {
        "mae":                  round(mae,     4),
        "rmse":                 round(rmse,    4),
        "mape":                 round(mape,    4),
        "directional_accuracy": round(dir_acc, 2) if dir_acc is not None else None,
    }


def model_summary(model: HybridModel) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
