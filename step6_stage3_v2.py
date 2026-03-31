"""
Step 6 v2: VMD as Input Features → Predict TARGET Directly
===========================================================
เปลี่ยน paradigm:
  ❌ เดิม: predict ทีละ IMF แล้ว sum → variance ต่ำ
  ✅ ใหม่: ใช้ IMF เป็น input features เพิ่มเข้าไปใน model
           แล้ว predict rss3_FOBm1_diff โดยตรง

Architecture: BiLSTM-Transformer Hybrid
  Input = [Original features (24)] + [VMD IMFs (5)] = 29 features
  → BiLSTM branch  (long-term memory)
  → Transformer branch (short-term attention)
  → Concat → Dense → prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = r'C:\Users\montc\Dropbox\PythonProj\Forecasting'
IMF_FILE   = os.path.join(BASE_DIR, 'vmd_output', 'stage3_imfs.csv')
TRAIN_FILE = os.path.join(BASE_DIR, 'stage3_train.csv')
VAL_FILE   = os.path.join(BASE_DIR, 'stage3_val.csv')
TEST_FILE  = os.path.join(BASE_DIR, 'stage3_test.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'model_output', 'stage3_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

LOOKBACK     = 30
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
print("STEP 6 v2: VMD AS INPUT FEATURES → DIRECT PREDICTION")
print(f"Device: {DEVICE}")
print("=" * 60)

# ─── Metrics ─────────────────────────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2    = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    safe  = denom > 1e-8
    smape = float(np.mean(np.abs(y_true[safe]-y_pred[safe])/denom[safe])*100) if safe.sum()>0 else 999
    da    = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    corr  = float(np.corrcoef(y_true, y_pred)[0,1]) if len(y_true)>1 else 0
    return {'MAE':mae,'RMSE':rmse,'R2':r2,'SMAPE':smape,'DA':da,'Corr':corr}

# ─── Dataset ─────────────────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feat_cols, target_col, lookback):
        self.X = df[feat_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.lookback = lookback
    def __len__(self):
        return max(0, len(self.X) - self.lookback)
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx:idx+self.lookback]),
                torch.tensor(self.y[idx+self.lookback]))

# ─── Model: BiLSTM-Transformer Hybrid ───────────────────────────────────────
class HybridModel(nn.Module):
    """
    Dual-path architecture:
    Path 1: BiLSTM with attention  → captures long-term sequential patterns
    Path 2: Transformer encoder    → captures short-term multi-scale patterns
    Fusion: concat + MLP
    """
    def __init__(self, input_size, hidden_size, num_layers, n_heads, dropout, seq_len):
        super().__init__()
        d_model = hidden_size

        # ── Path 1: BiLSTM ──
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

        # ── Path 2: Transformer ──
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # ── Fusion ──
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
        # Path 1: BiLSTM
        lstm_out, _ = self.bilstm(x)
        attn_w  = torch.softmax(self.lstm_attn(lstm_out), dim=1)
        lstm_ctx = (lstm_out * attn_w).sum(dim=1)      # (B, hidden*2)

        # Path 2: Transformer
        t_in  = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        t_out = self.transformer(t_in)
        t_ctx = t_out[:, -1, :]                         # (B, d_model)

        # Fusion
        combined = torch.cat([lstm_ctx, t_ctx], dim=1)
        return self.fusion(combined).squeeze(-1)

# ─── 1. Load & Prepare Data ──────────────────────────────────────────────────
print("\n[1] Loading Stage 3 data + VMD IMFs...")
imf_df   = pd.read_csv(IMF_FILE,   parse_dates=['date'])
train_df = pd.read_csv(TRAIN_FILE, parse_dates=['date'])
val_df   = pd.read_csv(VAL_FILE,   parse_dates=['date'])
test_df  = pd.read_csv(TEST_FILE,  parse_dates=['date'])

imf_cols = [c for c in imf_df.columns if c.startswith('IMF_')]
print(f"    Original features : {len(FEATURES_S3)}")
print(f"    VMD IMF features  : {len(imf_cols)}  ← used as INPUT")
print(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Merge IMFs onto splits
full_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)
n_tr, n_va = len(train_df), len(val_df)
min_len  = min(len(full_df), len(imf_df))
full_df  = full_df.iloc[:min_len].copy()
for col in imf_cols:
    full_df[col] = imf_df[col].values[:min_len]

# Build combined feature set: original + IMFs
feat_cols = [f for f in FEATURES_S3 if f in full_df.columns]
missing   = [f for f in FEATURES_S3 if f not in full_df.columns]
for f in missing:
    full_df[f] = 0.0
    feat_cols.append(f)

# Add IMF cols as input features
all_feat_cols = feat_cols + imf_cols
for col in all_feat_cols:
    full_df[col] = full_df[col].ffill().bfill().fillna(0)

train_full = full_df.iloc[:n_tr].copy()
val_full   = full_df.iloc[n_tr:n_tr+n_va].copy()
test_full  = full_df.iloc[n_tr+n_va:].copy()

print(f"    Combined input size: {len(all_feat_cols)} features")
print(f"    Target: {TARGET} (predicted directly)")

# ─── 2. Build & Train Model ──────────────────────────────────────────────────
print(f"\n[2] Training Hybrid BiLSTM-Transformer...")

train_ds = TimeSeriesDataset(train_full, all_feat_cols, TARGET, LOOKBACK)
val_ds   = TimeSeriesDataset(val_full,   all_feat_cols, TARGET, LOOKBACK)
test_ds  = TimeSeriesDataset(test_full,  all_feat_cols, TARGET, LOOKBACK)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, drop_last=False)

model = HybridModel(
    input_size  = len(all_feat_cols),
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    n_heads     = N_HEADS,
    dropout     = DROPOUT,
    seq_len     = LOOKBACK,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"    Total parameters: {total_params:,}")

criterion = nn.HuberLoss(delta=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

best_val, best_state, patience_c = float('inf'), None, 0
train_losses, val_losses = [], []

for epoch in range(MAX_EPOCHS):
    # Train
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

    # Validate
    model.eval()
    va_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            va_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
    avg_va = va_loss / max(len(val_loader), 1)
    scheduler.step(avg_va)

    train_losses.append(avg_tr)
    val_losses.append(avg_va)

    if avg_va < best_val:
        best_val   = avg_va
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_c = 0
    else:
        patience_c += 1

    if (epoch + 1) % 50 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"Train={avg_tr:.6f} | Val={avg_va:.6f} | LR={lr_now:.2e}")

    if patience_c >= PATIENCE:
        print(f"    Early stop epoch {epoch+1} | Best val={best_val:.6f}")
        break

model.load_state_dict(best_state)
torch.save(best_state, os.path.join(OUTPUT_DIR, 'hybrid_model.pt'))
print(f"    Model saved → hybrid_model.pt")

# ─── 3. Evaluate ─────────────────────────────────────────────────────────────
print("\n[3] Evaluating on Test set...")

model.eval()
preds = []
with torch.no_grad():
    for xb, _ in DataLoader(test_ds, BATCH_SIZE, shuffle=False):
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
final_pred = np.concatenate(preds)
actuals    = test_full[TARGET].values[LOOKBACK:LOOKBACK+len(final_pred)]

print(f"\n  Prediction stats:")
print(f"    Actual  — mean={actuals.mean():.4f} | std={actuals.std():.4f} | "
      f"min={actuals.min():.4f} | max={actuals.max():.4f}")
print(f"    Predict — mean={final_pred.mean():.4f} | std={final_pred.std():.4f} | "
      f"min={final_pred.min():.4f} | max={final_pred.max():.4f}")
print(f"    Std ratio (pred/actual): {final_pred.std()/actuals.std():.3f}  "
      f"(ideal = 1.0)")

metrics = calc_metrics(actuals, final_pred)
print(f"\n  Test Results:")
for k, v in metrics.items():
    print(f"    {k:8s}: {v:.4f}")

# Compare vs Stage 1
s1_path = os.path.join(BASE_DIR, 'model_output', 'stage1', 'stage1_test_results.csv')
if os.path.exists(s1_path):
    s1_res     = pd.read_csv(s1_path)
    s1_metrics = calc_metrics(s1_res['actual'].values, s1_res['predicted'].values)
    print(f"\n  Comparison vs Stage 1 BiLSTM:")
    print(f"  {'Metric':8s} | {'Stage 1':>10s} | {'Stage 3':>10s} | Change")
    print(f"  {'-'*52}")
    for m in ['MAE','RMSE','R2','SMAPE','DA','Corr']:
        s1v, s3v = s1_metrics[m], metrics[m]
        better = (m in ['MAE','RMSE','SMAPE'] and s3v < s1v) or \
                 (m in ['R2','DA','Corr'] and s3v > s1v)
        print(f"  {m:8s} | {s1v:>10.4f} | {s3v:>10.4f} | {'✅ better' if better else '❌ worse'}")

# ─── 4. Plots ─────────────────────────────────────────────────────────────────
print("\n[4] Generating plots...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Loss curve
ax1 = axes[0]
ax1.plot(train_losses, label='Train', color='#2980b9', lw=1)
ax1.plot(val_losses,   label='Val',   color='#e74c3c', lw=1)
best_ep = int(np.argmin(val_losses))
ax1.axvline(best_ep, color='green', linestyle='--', alpha=0.5, label=f'Best ep={best_ep}')
ax1.set_title('Training Curve — Hybrid BiLSTM-Transformer', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Huber Loss')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Prediction vs Actual
ax2 = axes[1]
x_axis = range(len(actuals))
ax2.plot(x_axis, actuals,    label='Actual',    color='#2c3e50', lw=1.0)
ax2.plot(x_axis, final_pred, label='Predicted', color='#e74c3c', lw=1.0, alpha=0.85)
ax2.fill_between(x_axis, actuals, final_pred, alpha=0.1, color='#e74c3c')
ax2.set_title(
    f'Stage 3 — VMD Hybrid Direct Prediction (Test set)\n'
    f'MAE={metrics["MAE"]:.4f} | RMSE={metrics["RMSE"]:.4f} | '
    f'R²={metrics["R2"]:.4f} | DA={metrics["DA"]:.1f}% | Corr={metrics["Corr"]:.4f}',
    fontsize=10
)
ax2.set_ylabel('rss3_FOBm1_diff (normalized)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'stage3_v2_results.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actuals, final_pred, alpha=0.3, s=10, color='steelblue')
lims = [min(actuals.min(), final_pred.min()), max(actuals.max(), final_pred.max())]
ax.plot(lims, lims, 'r--', lw=1, label='Perfect prediction')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title(f'Actual vs Predicted\nCorr={metrics["Corr"]:.4f} | R²={metrics["R2"]:.4f}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'stage3_v2_scatter.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ─── 5. Save ──────────────────────────────────────────────────────────────────
pd.DataFrame({
    'actual':    actuals,
    'predicted': final_pred,
    'error':     actuals - final_pred,
    'abs_error': np.abs(actuals - final_pred),
}).to_csv(os.path.join(OUTPUT_DIR, 'stage3_v2_test_results.csv'), index=False)

with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump({
        'model':         'VMD-Hybrid BiLSTM-Transformer (direct prediction)',
        'paradigm':      'VMD IMFs used as INPUT features, not targets',
        'input_features': all_feat_cols,
        'n_features':    len(all_feat_cols),
        'n_imf_features': len(imf_cols),
        'lookback':      LOOKBACK,
        'hidden_size':   HIDDEN_SIZE,
        'test_metrics':  {k: round(v,6) for k,v in metrics.items()},
    }, f, indent=2)

print("\n" + "=" * 60)
print("STAGE 3 v2 COMPLETE")
print("=" * 60)
print(f"  MAE   : {metrics['MAE']:.6f}")
print(f"  RMSE  : {metrics['RMSE']:.6f}")
print(f"  R²    : {metrics['R2']:.4f}")
print(f"  SMAPE : {metrics['SMAPE']:.2f}%")
print(f"  DA    : {metrics['DA']:.2f}%")
print(f"  Corr  : {metrics['Corr']:.4f}")
print(f"  Std ratio: {final_pred.std()/actuals.std():.3f}  (target: >0.5)")
print(f"\n  Output: {OUTPUT_DIR}")
print("\n✅ Complete — check Std ratio and Corr vs previous versions")
