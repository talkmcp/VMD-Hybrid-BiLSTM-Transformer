"""
Step 3B: VMD Decomposition
==========================
Variational Mode Decomposition (VMD) on target variable per stage.

Pipeline:
  1. Load train splits per stage
  2. Grid search for optimal K (number of IMFs) using energy ratio
  3. Decompose target series into K IMFs
  4. Visualize IMFs per stage
  5. Save IMF components + decomposition metadata
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from vmdpy import VMD
import warnings, json, os
warnings.filterwarnings('ignore')

INPUT_DIR  = r'C:\Users\montc\Dropbox\PythonProj\Forecasting'
OUTPUT_DIR = r'C:\Users\montc\Dropbox\PythonProj\Forecasting\vmd_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

TARGET = 'rss3_FOBm1_diff'

# ─── VMD Parameters ──────────────────────────────────────────────────────────
ALPHA   = 2000   # bandwidth constraint (moderate)
TAU     = 0      # noise-tolerance (0 = no noise)
DC      = 0      # no DC component
INIT    = 1      # initialize omegas uniformly
TOL     = 1e-7   # convergence tolerance
K_RANGE = [4, 5, 6, 7, 8]  # candidate number of modes

# ─── Helper: Energy ratio to select optimal K ────────────────────────────────
def energy_ratio(imfs):
    """Compute energy ratio: sum of IMF energies / total signal energy."""
    total = np.sum(np.sum(imfs, axis=0) ** 2)
    if total == 0:
        return 0
    imf_energies = np.array([np.sum(imf**2) for imf in imfs])
    return np.sum(imf_energies) / total

def reconstruction_error(signal, imfs):
    """RMSE between original signal and sum of IMFs."""
    recon = np.sum(imfs, axis=0)
    return np.sqrt(np.mean((signal - recon)**2))

def find_optimal_K(signal, k_range):
    """Find K that minimizes reconstruction error while maximizing energy ratio."""
    results = []
    for k in k_range:
        try:
            u, _, _ = VMD(signal, ALPHA, TAU, k, DC, INIT, TOL)
            err  = reconstruction_error(signal, u)
            eratio = energy_ratio(u)
            results.append({'K': k, 'rmse': err, 'energy_ratio': eratio})
            print(f"    K={k}: RMSE={err:.6f} | Energy Ratio={eratio:.4f}")
        except Exception as e:
            print(f"    K={k}: FAILED ({e})")
    if not results:
        return 5, []
    # Normalize scores: minimize RMSE, maximize energy ratio
    df_r = pd.DataFrame(results)
    df_r['rmse_norm']   = (df_r['rmse'] - df_r['rmse'].min()) / (df_r['rmse'].max() - df_r['rmse'].min() + 1e-10)
    df_r['eratio_norm'] = (df_r['energy_ratio'] - df_r['energy_ratio'].min()) / (df_r['energy_ratio'].max() - df_r['energy_ratio'].min() + 1e-10)
    df_r['score'] = -df_r['rmse_norm'] + df_r['eratio_norm']
    best_k = int(df_r.loc[df_r['score'].idxmax(), 'K'])
    return best_k, results

# ─── Helper: Plot IMFs ────────────────────────────────────────────────────────
def plot_imfs(signal, imfs, dates, stage_name, k):
    n = len(imfs)
    fig = plt.figure(figsize=(16, 3 * (n + 1)))
    gs  = gridspec.GridSpec(n + 1, 1, hspace=0.5)

    # Original signal
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(dates, signal, color='#2c3e50', linewidth=0.8)
    ax0.set_title(f'{stage_name.upper()} — Original Signal ({TARGET})', fontsize=11, fontweight='bold')
    ax0.set_ylabel('Normalized\nDiff', fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(axis='x', labelsize=7)

    colors = plt.cm.tab10(np.linspace(0, 0.9, n))
    freq_labels = []
    energies = [np.sum(imf**2) for imf in imfs]
    total_e = sum(energies)

    for i, (imf, color) in enumerate(zip(imfs, colors)):
        ax = fig.add_subplot(gs[i + 1])
        ax.plot(dates, imf, color=color, linewidth=0.8)
        pct = energies[i] / total_e * 100
        # Frequency label
        if i == 0:
            freq_lbl = 'High-freq (noise)'
        elif i == n - 1:
            freq_lbl = 'Trend (low-freq)'
        elif i == 1:
            freq_lbl = 'High-freq'
        else:
            freq_lbl = f'Mid-freq'
        freq_labels.append(freq_lbl)
        ax.set_title(f'IMF {i+1} — {freq_lbl} | Energy {pct:.1f}%', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=7)

    recon = np.sum(imfs, axis=0)
    rmse  = np.sqrt(np.mean((signal - recon)**2))
    fig.suptitle(f'{stage_name.upper()} | K={k} optimal IMFs | Reconstruction RMSE={rmse:.6f}',
                 fontsize=12, fontweight='bold', y=1.005)
    plt.savefig(f'{OUTPUT_DIR}/plots/{stage_name}_vmd_K{k}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {stage_name}_vmd_K{k}.png")
    return freq_labels

# ─── Main: Process each stage ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 3B: VMD DECOMPOSITION")
print("=" * 60)

vmd_metadata = {}

for stage_name in ['stage1', 'stage2', 'stage3']:
    print(f"\n{'─'*50}")
    print(f"Processing {stage_name.upper()}...")

    # Load full stage data (train + val + test combined for decomposition)
    dfs = []
    for split in ['train', 'val', 'test']:
        path = f'{INPUT_DIR}/{stage_name}_{split}.csv'
        dfs.append(pd.read_csv(path))
    df_full = pd.concat(dfs, ignore_index=True)

    # Ensure date column
    if 'date' in df_full.columns:
        df_full['date'] = pd.to_datetime(df_full['date'])
    
    # Extract target series, drop NaN
    series_raw = df_full[TARGET].values.astype(float)
    dates_raw  = df_full['date'].values if 'date' in df_full.columns else np.arange(len(series_raw))
    
    # Remove NaN for VMD (VMD needs clean series)
    valid_mask = ~np.isnan(series_raw)
    series = series_raw[valid_mask]
    dates  = dates_raw[valid_mask]
    
    print(f"  Series length: {len(series):,} | NaN removed: {(~valid_mask).sum()}")
    # VMD requires even-length signal
    if len(series) % 2 != 0:
        series = series[:-1]
        dates  = dates[:-1]


    # ── Grid search for optimal K
    print(f"\n  Grid search K in {K_RANGE}:")
    best_k, k_results = find_optimal_K(series, K_RANGE)
    print(f"\n  ✅ Optimal K = {best_k}")

    # ── Final VMD with optimal K
    print(f"\n  Running VMD with K={best_k}...")
    u, u_hat, omega = VMD(series, ALPHA, TAU, best_k, DC, INIT, TOL)
    
    rmse_final = reconstruction_error(series, u)
    print(f"  Reconstruction RMSE: {rmse_final:.6f}")

    # ── Classify IMFs by dominant frequency
    # omega shape: (iterations, K) → use last row as converged frequencies
    dominant_freqs = omega[-1]  # normalized frequencies
    sorted_idx = np.argsort(dominant_freqs)  # low → high frequency
    imfs_sorted = u[sorted_idx]
    freqs_sorted = dominant_freqs[sorted_idx]

    freq_labels = []
    for i, f in enumerate(freqs_sorted):
        if f < 0.05:
            lbl = 'Trend'
        elif f < 0.15:
            lbl = 'Low-freq'
        elif f < 0.30:
            lbl = 'Mid-freq'
        else:
            lbl = 'High-freq'
        freq_labels.append(lbl)
    
    print(f"  IMF frequencies (normalized): {[round(f, 4) for f in freqs_sorted]}")
    print(f"  IMF labels: {freq_labels}")

    # ── Plot
    plot_imfs(series, imfs_sorted, dates, stage_name, best_k)

    # ── Save IMF components as CSV
    imf_df = pd.DataFrame({'date': dates})
    for i, imf in enumerate(imfs_sorted):
        imf_df[f'IMF_{i+1}_{freq_labels[i]}'] = imf
    imf_df['reconstructed'] = np.sum(imfs_sorted, axis=0)
    imf_df['original']      = series
    imf_df['residual']      = series - imf_df['reconstructed']
    imf_df.to_csv(f'{OUTPUT_DIR}/{stage_name}_imfs.csv', index=False)

    # ── Metadata
    energies = [float(np.sum(imf**2)) for imf in imfs_sorted]
    total_e  = sum(energies)
    vmd_metadata[stage_name] = {
        'optimal_K':     best_k,
        'recon_rmse':    float(rmse_final),
        'series_length': int(len(series)),
        'k_search_results': k_results,
        'imfs': [
            {
                'imf_index':   i + 1,
                'label':       freq_labels[i],
                'frequency':   float(freqs_sorted[i]),
                'energy':      float(energies[i]),
                'energy_pct':  float(energies[i] / total_e * 100),
            }
            for i in range(best_k)
        ]
    }
    print(f"  Saved: {stage_name}_imfs.csv")

# ─── Save metadata ────────────────────────────────────────────────────────────
with open(f'{OUTPUT_DIR}/vmd_metadata.json', 'w') as f:
    json.dump(vmd_metadata, f, indent=2)

# ─── Summary plot: energy distribution across stages ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('VMD Energy Distribution per Stage', fontsize=13, fontweight='bold')

colors_bar = plt.cm.tab10(np.linspace(0, 0.9, 8))
for ax, stage_name in zip(axes, ['stage1', 'stage2', 'stage3']):
    meta = vmd_metadata[stage_name]
    labels = [f"IMF{d['imf_index']}\n{d['label']}" for d in meta['imfs']]
    energies_pct = [d['energy_pct'] for d in meta['imfs']]
    bars = ax.bar(labels, energies_pct, color=colors_bar[:len(labels)])
    ax.set_title(f"{stage_name.upper()} (K={meta['optimal_K']})", fontsize=10, fontweight='bold')
    ax.set_ylabel('Energy (%)', fontsize=9)
    ax.set_ylim(0, max(energies_pct) * 1.2)
    for bar, pct in zip(bars, energies_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/energy_distribution_all_stages.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VMD DECOMPOSITION COMPLETE — SUMMARY")
print("=" * 60)
for stage_name, meta in vmd_metadata.items():
    print(f"\n  {stage_name.upper()}")
    print(f"    Optimal K    : {meta['optimal_K']}")
    print(f"    Recon RMSE   : {meta['recon_rmse']:.6f}")
    for d in meta['imfs']:
        print(f"    IMF {d['imf_index']} ({d['label']:12s}): freq={d['frequency']:.4f} | energy={d['energy_pct']:.1f}%")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files saved:")
print("  stage1_imfs.csv / stage2_imfs.csv / stage3_imfs.csv")
print("  plots/stage1_vmd_K*.png / stage2_vmd_K*.png / stage3_vmd_K*.png")
print("  plots/energy_distribution_all_stages.png")
print("  vmd_metadata.json")
print("\n✅ Ready for Step 4: Stage 1 BiLSTM Model")
