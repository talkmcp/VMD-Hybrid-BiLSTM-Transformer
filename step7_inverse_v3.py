"""
Step 7 v3: Inverse Transform — Final Fix (ffill NaN in actual prices)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

BASE_DIR     = r'C:\Users\montc\Dropbox\PythonProj\Forecasting'
RAW_FILE     = BASE_DIR + r'\rubber_data_combined_Update.xlsx'
RESULTS_FILE = BASE_DIR + r'\model_output\stage3_v2\stage3_v2_test_results.csv'
OUTPUT_DIR   = BASE_DIR + r'\model_output\final_evaluation_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR + r'\plots', exist_ok=True)

TARGET_ORIG  = 'rss3_FOBm1'
TARGET_DIFF  = 'rss3_FOBm1_diff'
STAGE3_START = '2018-05-07'
STAGE3_END   = '2026-02-27'
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1
LOOKBACK     = 30

FEATURES_S3 = [
    'rss3_FOBm1_diff', 'rss3_FOBm2_diff',
    'str20_fobm1_diff', 'str20_fobm2_diff',
    'latex_fobm1_diff', 'latex_fobm2_diff',
    'rss3_JPXm1_diff', 'rss3_shfe_m1_diff', 'rss3_shfe_m2_diff',
    'rss3_sgxsett_diff', 'tsr20_sgxsett_diff',
    'CupLump_diff', 'uss',
    'usd_thb_diff', 'cny_thb_diff', 'usd_cny_diff',
    'brent_usd_diff', 'wti_usd_diff', 'brent_return', 'brent_lag1_diff',
    'china_pmi_mfg', 'bdi', 'enso_oni_diff', 'covid_period_diff',
]
NON_STATIONARY = [
    'CupLump', 'usd_thb', 'enso_oni', 'rss3_shfe_m2', 'latex_fobm2',
    'wti_usd', 'covid_period', 'tsr20_sgxsett', 'str20_fobm1', 'rss3_FOBm2',
    'brent_lag1', 'usd_cny', 'rss3_shfe_m1', 'brent_usd', 'rss3_JPXm1',
    'str20_fobm2', 'rss3_sgxsett', 'rss3_FOBm1', 'latex_fobm1', 'cny_thb',
]

print("=" * 60)
print("STEP 7 v3: INVERSE TRANSFORM — FINAL")
print("=" * 60)

# ── 1. Load & prep raw data ───────────────────────────────────────────────────
print("\n[1] Loading raw data...")
df = pd.read_excel(RAW_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ffill original price first (before diff)
df[TARGET_ORIG] = df[TARGET_ORIG].ffill()

num_cols = [c for c in df.columns if c != 'date']
df[num_cols] = df[num_cols].ffill(limit=3)
for col in NON_STATIONARY:
    if col in df.columns:
        df[f'{col}_diff'] = df[col].diff()

# Stage 3 slice
s3 = df[(df['date'] >= STAGE3_START) & (df['date'] <= STAGE3_END)].copy().reset_index(drop=True)
for f in FEATURES_S3:
    if f not in s3.columns:
        s3[f] = 0.0
s3[FEATURES_S3] = s3[FEATURES_S3].ffill().bfill().interpolate(limit=10).fillna(0)

n       = len(s3)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * (TRAIN_RATIO + VAL_RATIO))

# ── 2. Refit scaler on train ──────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(s3.iloc[:n_train][FEATURES_S3].values)
target_idx = FEATURES_S3.index(TARGET_DIFF)
feat_min   = scaler.data_min_[target_idx]
feat_max   = scaler.data_max_[target_idx]
print(f"    Scaler target range: [{feat_min:.4f}, {feat_max:.4f}] → [-1,1]")

# ── 3. Load predictions ───────────────────────────────────────────────────────
print("\n[2] Loading predictions...")
results     = pd.read_csv(RESULTS_FILE)
pred_norm   = results['predicted'].values
actual_norm = results['actual'].values
n_pred      = len(pred_norm)
print(f"    {n_pred} predictions")

# ── 4. Inverse transform → Baht/kg diff ──────────────────────────────────────
def inv(v, lo, hi, slo=-1, shi=1):
    return (v - slo) / (shi - slo) * (hi - lo) + lo

pred_diff   = inv(pred_norm,   feat_min, feat_max)
actual_diff = inv(actual_norm, feat_min, feat_max)

# ── 5. Get actual price series ────────────────────────────────────────────────
print("\n[3] Getting actual price series...")
# Use s3 which already has rss3_FOBm1 (ffilled)
test_s3 = s3.iloc[n_val:].copy().reset_index(drop=True)
price_series = test_s3[TARGET_ORIG].ffill().values  # extra safety ffill
print(f"    Price series length: {len(price_series)}")
print(f"    NaN remaining: {np.isnan(price_series).sum()}")
print(f"    Range: [{np.nanmin(price_series):.2f}, {np.nanmax(price_series):.2f}] Baht/kg")

# Align with predictions (offset by LOOKBACK)
if LOOKBACK > 0 and len(price_series) > LOOKBACK:
    last_known    = price_series[LOOKBACK - 1]
    actual_prices = price_series[LOOKBACK: LOOKBACK + n_pred]
else:
    last_known    = price_series[0]
    actual_prices = price_series[:n_pred]

# Trim to matching length
n_use = min(n_pred, len(actual_prices))
actual_prices = actual_prices[:n_use]
pred_diff     = pred_diff[:n_use]
actual_diff   = actual_diff[:n_use]
pred_norm     = pred_norm[:n_use]
actual_norm   = actual_norm[:n_use]

print(f"    last_known price: {last_known:.2f} Baht/kg")
print(f"    Actual prices NaN: {np.isnan(actual_prices).sum()}")
print(f"    Sample: {actual_prices[:5]}")

# ── 6. Reconstruct price level ────────────────────────────────────────────────
print("\n[4] Reconstructing prices...")

# A: Cumulative (biased — for comparison only)
cum_pred    = np.zeros(n_use)
cum_actual  = np.zeros(n_use)
cum_pred[0]   = last_known + pred_diff[0]
cum_actual[0] = last_known + actual_diff[0]
for i in range(1, n_use):
    cum_pred[i]   = cum_pred[i-1]   + pred_diff[i]
    cum_actual[i] = cum_actual[i-1] + actual_diff[i]

# B: Rolling 1-step (correct — use actual price as anchor each day)
roll_pred = np.zeros(n_use)
for i in range(n_use):
    anchor = last_known if i == 0 else actual_prices[i-1]
    if np.isnan(anchor):
        # fallback: use last valid price
        valid = actual_prices[:i]
        anchor = valid[~np.isnan(valid)][-1] if np.any(~np.isnan(valid)) else last_known
    roll_pred[i] = anchor + pred_diff[i]

print(f"    Actual  prices: [{np.nanmin(actual_prices):.2f}, {np.nanmax(actual_prices):.2f}] Baht/kg")
print(f"    Cumsum  pred  : [{cum_pred.min():.2f}, {cum_pred.max():.2f}] Baht/kg")
print(f"    Rolling pred  : [{roll_pred.min():.2f}, {roll_pred.max():.2f}] Baht/kg")

# ── 7. Metrics ────────────────────────────────────────────────────────────────
def calc(y_true, y_pred, label):
    mask  = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae   = float(np.mean(np.abs(yt - yp)))
    rmse  = float(np.sqrt(np.mean((yt - yp)**2)))
    ss_r  = np.sum((yt - yp)**2)
    ss_t  = np.sum((yt - np.mean(yt))**2)
    r2    = float(1 - ss_r/ss_t) if ss_t > 0 else 0
    mape  = float(np.mean(np.abs((yt - yp)/yt)) * 100)
    da    = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))) * 100)
    corr  = float(np.corrcoef(yt, yp)[0,1])
    mean_ = float(np.mean(yt))
    print(f"\n  [{label}]  n={mask.sum()}")
    print(f"    MAE   : {mae:.2f} Baht/kg  ({mae/mean_*100:.2f}% of mean)")
    print(f"    RMSE  : {rmse:.2f} Baht/kg")
    print(f"    MAPE  : {mape:.2f}%")
    print(f"    R²    : {r2:.4f}")
    print(f"    DA    : {da:.2f}%")
    print(f"    Corr  : {corr:.4f}")
    return {'MAE':mae,'RMSE':rmse,'R2':r2,'MAPE':mape,'DA':da,'Corr':corr,'mean':mean_}

print("\n[5] Metrics:")
m_roll = calc(actual_prices, roll_pred, 'Rolling 1-Step (Report This)')
m_cum  = calc(actual_prices, cum_pred,  'Cumulative (Reference only)')

# Diff space
mask_d = ~np.isnan(actual_diff) & ~np.isnan(pred_diff)
ad, pd_ = actual_diff[mask_d], pred_diff[mask_d]
da_diff   = float(np.mean(np.sign(ad) == np.sign(pd_)) * 100)
corr_diff = float(np.corrcoef(ad, pd_)[0,1])
mae_diff  = float(np.mean(np.abs(ad - pd_)))
rmse_diff = float(np.sqrt(np.mean((ad - pd_)**2)))
print(f"\n  [Diff Space — model's direct output]")
print(f"    MAE   : {mae_diff:.4f} Baht/kg/day")
print(f"    RMSE  : {rmse_diff:.4f} Baht/kg/day")
print(f"    DA    : {da_diff:.2f}%")
print(f"    Corr  : {corr_diff:.4f}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
print("\n[6] Generating plots...")
x = np.arange(n_use)
mean_price = m_roll['mean']

fig, axes = plt.subplots(3, 1, figsize=(16, 15))

ax = axes[0]
ax.plot(x, actual_prices, label='Actual',         color='#2c3e50', lw=1.5)
ax.plot(x, roll_pred,     label='Rolling 1-step', color='#e74c3c', lw=1.2, alpha=0.85)
ax.fill_between(x, roll_pred - m_roll['RMSE'], roll_pred + m_roll['RMSE'],
                alpha=0.15, color='#e74c3c', label=f'±RMSE ({m_roll["RMSE"]:.2f} Baht/kg)')
ax.set_title(
    f'RSS3 FOB — Rolling 1-Step Reconstruction\n'
    f'MAE={m_roll["MAE"]:.2f} Baht/kg | MAPE={m_roll["MAPE"]:.2f}% | '
    f'R²={m_roll["R2"]:.4f} | DA={m_roll["DA"]:.1f}% | Corr={m_roll["Corr"]:.4f}',
    fontsize=10, fontweight='bold')
ax.set_ylabel('Price (Baht/kg)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x, actual_prices, label='Actual',          color='#2c3e50', lw=1.5)
ax.plot(x, cum_pred,      label='Cumulative pred', color='#f39c12', lw=1.2, alpha=0.85)
ax.set_title(
    f'Cumulative Reconstruction (Drift visible)\n'
    f'MAE={m_cum["MAE"]:.2f} | R²={m_cum["R2"]:.4f} | Corr={m_cum["Corr"]:.4f}',
    fontsize=9)
ax.set_ylabel('Price (Baht/kg)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(x, actual_diff, label='Actual Δprice',    color='#2c3e50', lw=1.0, alpha=0.8)
ax.plot(x, pred_diff,   label='Predicted Δprice', color='#3498db', lw=1.0, alpha=0.8)
ax.axhline(0, color='gray', lw=0.8, linestyle='--')
ax.set_title(f'Daily Price Change (Diff Space)\nDA={da_diff:.2f}% | Corr={corr_diff:.4f}',
             fontsize=9)
ax.set_ylabel('Δ Price (Baht/kg/day)'); ax.set_xlabel('Test Days')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + r'\plots\final_prediction_v3.png', dpi=150, bbox_inches='tight')
plt.close()

# Scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, yp, m, ttl in [
    (axes[0], roll_pred, m_roll, 'Rolling 1-Step'),
    (axes[1], cum_pred,  m_cum,  'Cumulative'),
]:
    mask = ~np.isnan(actual_prices) & ~np.isnan(yp)
    ax.scatter(actual_prices[mask], yp[mask], alpha=0.35, s=10, color='steelblue')
    lims = [min(actual_prices[mask].min(), yp[mask].min())*0.97,
            max(actual_prices[mask].max(), yp[mask].max())*1.03]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_title(f'{ttl}\nCorr={m["Corr"]:.4f} | R²={m["R2"]:.4f} | MAPE={m["MAPE"]:.2f}%')
    ax.set_xlabel('Actual (Baht/kg)'); ax.set_ylabel('Predicted (Baht/kg)')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + r'\plots\scatter_v3.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 9. Save ───────────────────────────────────────────────────────────────────
pd.DataFrame({
    'actual_price':       actual_prices,
    'rolling_pred_price': roll_pred,
    'cumsum_pred_price':  cum_pred,
    'actual_diff_baht':   actual_diff,
    'pred_diff_baht':     pred_diff,
    'residual_rolling':   actual_prices - roll_pred,
    'residual_cumsum':    actual_prices - cum_pred,
}).to_csv(OUTPUT_DIR + r'\final_predictions_v3.csv', index=False)

with open(OUTPUT_DIR + r'\final_evaluation_v3.json', 'w') as f:
    json.dump({
        'model': 'VMD-Hybrid BiLSTM-Transformer (Stage 3)',
        'rolling_1step': {k: round(v,4) for k,v in m_roll.items()},
        'cumulative':    {k: round(v,4) for k,v in m_cum.items()},
        'diff_space':    {'DA': round(da_diff,2), 'Corr': round(corr_diff,4),
                          'MAE': round(mae_diff,4), 'RMSE': round(rmse_diff,4)},
        'mean_price_baht': round(mean_price, 2),
        'n': n_use,
    }, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETE")
print("=" * 60)
print()
print("  ┌─── Rolling 1-Step ─────────────────────────────────┐")
print(f"  │  MAE    : {m_roll['MAE']:7.2f} Baht/kg  ({m_roll['MAE']/mean_price*100:.2f}% of mean)   │")
print(f"  │  RMSE   : {m_roll['RMSE']:7.2f} Baht/kg                     │")
print(f"  │  MAPE   : {m_roll['MAPE']:7.2f} %                           │")
print(f"  │  R²     : {m_roll['R2']:7.4f}                              │")
print(f"  │  DA     : {m_roll['DA']:7.2f} % (daily direction)           │")
print(f"  │  Corr   : {m_roll['Corr']:7.4f}                              │")
print("  └─────────────────────────────────────────────────────┘")
print()
print("  ┌─── Diff Space (Model Direct Output) ───────────────┐")
print(f"  │  DA     : {da_diff:7.2f} % (sign of Δprice correct)      │")
print(f"  │  Corr   : {corr_diff:7.4f}                              │")
print("  └─────────────────────────────────────────────────────┘")
print()
print(f"  Mean price (test): {mean_price:.2f} Baht/kg")
print(f"  Output: {OUTPUT_DIR}")
