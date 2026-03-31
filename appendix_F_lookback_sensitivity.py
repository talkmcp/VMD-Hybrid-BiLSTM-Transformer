"""
Appendix F: Hyperparameter Sensitivity — Lookback Window L
===========================================================
Based on step6_stage3_v2.py training pipeline.
Trains VMD-Hybrid BiLSTM-Transformer for each L in {10, 20, 30, 45, 60}
and reports out-of-sample metrics on the Stage 3 test set.

Output : appendix_F_lookback_sensitivity.csv + appendix_F_lookback_sensitivity.png
Runtime: ~20-40 min on GPU (5 models x ~100 epochs each)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings, time
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0.  PATHS & CONFIG  (same as step6_stage3_v2.py)
# ─────────────────────────────────────────────
BASE_DIR   = r'C:\Users\montc\Dropbox\PythonProj\Forecasting'
IMF_FILE   = os.path.join(BASE_DIR, 'vmd_output', 'stage3_imfs.csv')
TRAIN_FILE = os.path.join(BASE_DIR, 'stage3_train.csv')
VAL_FILE   = os.path.join(BASE_DIR, 'stage3_val.csv')
TEST_FILE  = os.path.join(BASE_DIR, 'stage3_test.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'model_output', 'appendix_F')
os.makedirs(OUTPUT_DIR, exist_ok=True)

L_VALUES     = [10, 20, 30, 45, 60]
L_BASELINE   = 30
BATCH_SIZE   = 32
MAX_EPOCHS   = 300
LR           = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 30
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
N_HEADS      = 4
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET       = 'rss3_FOBm1_diff'
SEED         = 42

FEATURES_S3 = [
    'rss3_FOBm1_diff', 'rss3_FOBm2_diff',
    'str20_fobm1_diff', 'str20_fobm2_diff',
    'latex_fobm1_diff', 'latex_fobm2_diff',
    'rss3_JPXm1_diff', 'rss3_shfe_m1_diff', 'rss3_shfe_m2_diff',
    'rss3_sgxsett_diff', 'tsr20_sgxsett_diff',
    'CupLump_diff', 'uss',
    'usd_thb_diff', 'cny_thb_diff', 'usd_cny_diff',
    'brent_usd_diff', 'wti_usd_diff', 'brent_return', 'brent_lag1_diff',
    'china_pmi_mfg', 'bdi',
    'enso_oni_diff', 'covid_period_diff',
]

print("=" * 60)
print("APPENDIX F: Lookback Window Sensitivity")
print(f"L values: {L_VALUES}")
print(f"Device  : {DEVICE}")
print("=" * 60)


# ─── Dataset (identical to step6) ────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feat_cols, target_col, lookback):
        self.X        = df[feat_cols].values.astype(np.float32)
        self.y        = df[target_col].values.astype(np.float32)
        self.lookback = lookback
    def __len__(self):
        return max(0, len(self.X) - self.lookback)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx:idx + self.lookback]),
                torch.tensor(self.y[idx + self.lookback]))


# ─── Model (identical to step6) ──────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 n_heads, dropout, seq_len):
        super().__init__()
        d_model = hidden_size

        self.bilstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.lstm_attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        fusion_in = hidden_size * 2 + d_model
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        attn_w      = torch.softmax(self.lstm_attn(lstm_out), dim=1)
        lstm_ctx    = (lstm_out * attn_w).sum(dim=1)
        t_in        = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        t_out       = self.transformer(t_in)
        t_ctx       = t_out[:, -1, :]
        combined    = torch.cat([lstm_ctx, t_ctx], dim=1)
        return self.fusion(combined).squeeze(-1)


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(actual, predicted):
    actual    = np.array(actual,    dtype=np.float64)
    predicted = np.array(predicted, dtype=np.float64)
    mae       = float(np.mean(np.abs(actual - predicted)))
    rmse      = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    corr, _   = pearsonr(actual, predicted)
    stdr      = float(np.std(predicted, ddof=1) / np.std(actual, ddof=1))
    mask      = actual != 0
    da        = float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))
    return {"DA_pct": round(da*100,1), "Corr": round(float(corr),4),
            "MAE": round(mae,4), "RMSE": round(rmse,4), "StdR": round(stdr,4)}


# ─── Load data once ──────────────────────────────────────────────────────────
print("\n[1] Loading data...")
imf_df   = pd.read_csv(IMF_FILE,   parse_dates=['date'])
train_df = pd.read_csv(TRAIN_FILE, parse_dates=['date'])
val_df   = pd.read_csv(VAL_FILE,   parse_dates=['date'])
test_df  = pd.read_csv(TEST_FILE,  parse_dates=['date'])

imf_cols = [c for c in imf_df.columns if c.startswith('IMF_')]
print(f"  Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
print(f"  IMFs: {len(imf_cols)}  Economic: {len(FEATURES_S3)}")

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
n_tr, n_va = len(train_df), len(val_df)
min_len  = min(len(full_df), len(imf_df))
full_df  = full_df.iloc[:min_len].copy()
for col in imf_cols:
    full_df[col] = imf_df[col].values[:min_len]

feat_cols = [f for f in FEATURES_S3 if f in full_df.columns]
for f in [f for f in FEATURES_S3 if f not in full_df.columns]:
    full_df[f] = 0.0
    feat_cols.append(f)

all_feat_cols = feat_cols + imf_cols
for col in all_feat_cols:
    full_df[col] = full_df[col].ffill().bfill().fillna(0)

train_full = full_df.iloc[:n_tr].copy()
val_full   = full_df.iloc[n_tr:n_tr + n_va].copy()
test_full  = full_df.iloc[n_tr + n_va:].copy()
N_FEATURES = len(all_feat_cols)
print(f"  Total input features: {N_FEATURES}")


# ─── Train & Evaluate each L ─────────────────────────────────────────────────
results = []

for L in L_VALUES:
    print(f"\n{'='*60}")
    print(f"  Training  L = {L}  (baseline = {L_BASELINE})")
    print(f"{'='*60}")
    t_start = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = TimeSeriesDataset(train_full, all_feat_cols, TARGET, L)
    val_ds   = TimeSeriesDataset(val_full,   all_feat_cols, TARGET, L)
    test_ds  = TimeSeriesDataset(test_full,  all_feat_cols, TARGET, L)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, drop_last=False)

    model = HybridModel(
        input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, n_heads=N_HEADS,
        dropout=DROPOUT, seq_len=L,
    ).to(DEVICE)

    criterion = nn.HuberLoss(delta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    best_val, best_state, patience_c = float('inf'), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        ep_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        avg_tr = ep_loss / max(len(train_loader), 1)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                va_loss += criterion(model(xb.to(DEVICE)),
                                     yb.to(DEVICE)).item()
        avg_va = va_loss / max(len(val_loader), 1)
        scheduler.step(avg_va)

        if avg_va < best_val:
            best_val   = avg_va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d} | Train={avg_tr:.5f} "
                  f"Val={avg_va:.5f} LR={optimizer.param_groups[0]['lr']:.1e}")

        if patience_c >= PATIENCE:
            print(f"    Early stop epoch {epoch+1} | best_val={best_val:.5f}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUTPUT_DIR, f'model_L{L}.pt'))

    # Predict
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in DataLoader(test_ds, BATCH_SIZE, shuffle=False):
            preds.append(model(xb.to(DEVICE)).cpu().numpy())

    y_pred   = np.concatenate(preds)
    y_actual = test_full[TARGET].values[L:L + len(y_pred)]

    m            = compute_metrics(y_actual, y_pred)
    m["L"]       = L
    m["Runtime_min"] = round((time.time() - t_start) / 60, 1)
    results.append(m)

    star = " ★" if L == L_BASELINE else ""
    print(f"\n  Result L={L}: DA={m['DA_pct']}%  Corr={m['Corr']}  "
          f"MAE={m['MAE']}  StdR={m['StdR']}{star}")


# ─── Print table ─────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)[
    ["L", "DA_pct", "Corr", "MAE", "RMSE", "StdR", "Runtime_min"]]

print("\n")
print("=" * 75)
print("Table F1. Lookback Window Sensitivity — VMD-Hybrid BiLSTM-Transformer")
print("=" * 75)
print(df_res.to_string(index=False))
print()
for _, row in df_res.iterrows():
    star = " ★" if row["L"] == L_BASELINE else ""
    print(f"  L={int(row['L'])}: DA={row['DA_pct']}%  "
          f"Corr={row['Corr']}  MAE={row['MAE']}  StdR={row['StdR']}{star}")


# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (metric, label, color) in zip(axes, [
    ("Corr", "Pearson r",         "#2166AC"),
    ("StdR", "StdR",              "#D6604D"),
    ("MAE",  "MAE (norm. units)", "#4DAC26"),
]):
    ax.plot(df_res["L"], df_res[metric], "o-",
            color=color, linewidth=2, markersize=8)
    ax.axvline(L_BASELINE, color="red", linestyle="--",
               alpha=0.6, label=f"L={L_BASELINE} (selected)")
    ax.set_xlabel("Lookback Window L (trading days)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs L")
    ax.set_xticks(L_VALUES)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle(
    "Figure F1. Hyperparameter Sensitivity: Lookback Window L\n"
    "VMD-Hybrid BiLSTM-Transformer — Stage 3 Test Set",
    fontsize=11)
plt.tight_layout()
for path in ["appendix_F_lookback_sensitivity.png",
             os.path.join(OUTPUT_DIR, "appendix_F_lookback_sensitivity.png")]:
    plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()

# ─── Save CSV ─────────────────────────────────────────────────────────────────
for path in ["appendix_F_lookback_sensitivity.csv",
             os.path.join(OUTPUT_DIR, "appendix_F_lookback_sensitivity.csv")]:
    df_res.to_csv(path, index=False)

print("\nSaved: appendix_F_lookback_sensitivity.csv")
print("Saved: appendix_F_lookback_sensitivity.png")
print(f"Model checkpoints: {OUTPUT_DIR}")
