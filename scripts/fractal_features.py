"""
Fractal complexity features for maritime vessel RCS classification.
Tests whether RCS fluctuation complexity (Higuchi FD, Petrosian FD,
Sample Entropy, Hurst exponent) improves Cargo-Tanker discrimination
beyond the d=0.56 amplitude-feature ceiling.

Usage:
    python scripts/fractal_features.py --data /path/to/rcs_train_4class.csv
"""
import argparse, json, warnings, time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

FEATS     = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc',
             'aspect_ratio', 'footprint_m2',
             'range', 'down_range_extent', 'cross_range_extent']
N_FEAT    = len(FEATS)
CLASSES   = [30, 52, 70, 80]
CLS_NAMES = ['Fishing', 'Tug', 'Cargo', 'Tanker']
N_CLS     = 4
N_FOLDS   = 5
MIN_LEN   = 3

# Minimum track lengths for reliable estimation
MIN_PETROSIAN = 5
MIN_HIGUCHI   = 10
MIN_SAMPENT   = 10
MIN_HURST     = 20


# ── Fractal estimators ─────────────────────────────────────────────────────────

def petrosian_fd(x):
    """Petrosian FD: counts sign changes in first difference. Works from 5 points."""
    n = len(x)
    if n < MIN_PETROSIAN:
        return np.nan
    d = np.diff(x)
    nzc = int(np.sum(d[:-1] * d[1:] < 0))
    if nzc == 0:
        return 1.0   # flat / monotone = line (dimension 1)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nzc)))


def higuchi_fd(x, kmax=8):
    """
    Higuchi FD: slope of log(curve-length) vs log(1/k).
    FD ~ 1 = smooth; FD ~ 2 = maximally complex.
    Reliable from 10 points; best above 20.
    """
    n = len(x)
    if n < MIN_HIGUCHI:
        return np.nan
    kmax = min(kmax, n // 2)
    if kmax < 3:
        return np.nan
    L_vals, k_vals = [], []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, n, k)
            if len(idxs) < 2:
                continue
            nm = len(idxs) - 1          # number of steps
            norm = (n - 1) / (nm * k)
            Lk.append(np.sum(np.abs(np.diff(x[idxs]))) * norm)
        if Lk:
            L_vals.append(np.mean(Lk))
            k_vals.append(k)
    if len(k_vals) < 3:
        return np.nan
    L_arr = np.array(L_vals)
    k_arr = np.array(k_vals, dtype=float)
    valid = L_arr > 0
    if valid.sum() < 3:
        return np.nan
    # L(k) ∝ k^(-FD)  =>  slope of log(L) vs log(k) = -FD
    slope, _ = np.polyfit(np.log(k_arr[valid]), np.log(L_arr[valid]), 1)
    return float(-slope)


def sample_entropy(x, m=2, r_factor=0.2):
    """
    Sample entropy: unpredictability of the series.
    High SampEn = complex/irregular; low = regular/predictable.
    Capped at 60 points to keep runtime tractable.
    """
    n = len(x)
    if n < MIN_SAMPENT:
        return np.nan
    if n > 60:
        x = x[:60]; n = 60
    r = r_factor * np.std(x, ddof=1)
    if r == 0:
        return np.nan

    def _count_templates(tlen):
        segs = np.array([x[i:i + tlen] for i in range(n - tlen)])
        cnt = 0
        for i in range(len(segs)):
            cnt += int(np.sum(np.max(np.abs(segs - segs[i]), axis=1) <= r)) - 1
        return cnt

    B = _count_templates(m)
    A = _count_templates(m + 1)
    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else 2.0   # 2.0 = high entropy cap


def hurst_rs(x):
    """
    Hurst exponent via rescaled-range analysis.
    H > 0.5 = persistent (trending), H < 0.5 = anti-persistent (mean-reverting).
    Reliable from 20 points.
    """
    n = len(x)
    if n < MIN_HURST:
        return np.nan
    min_lag = max(4, n // 8)
    lag_cands = np.unique(np.round(np.geomspace(min_lag, n // 2, 12)).astype(int))
    lags, rs_vals = [], []
    for lag in lag_cands:
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = x[i * lag:(i + 1) * lag]
            dev = np.cumsum(chunk - np.mean(chunk))
            R = np.ptp(dev)
            S = np.std(chunk, ddof=1)
            if S > 0 and R > 0:
                rs_chunk.append(R / S)
        if len(rs_chunk) >= 2:
            lags.append(lag)
            rs_vals.append(np.mean(rs_chunk))
    if len(lags) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log(lags), np.log(rs_vals), 1)
    return float(slope)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to rcs_train_4class.csv')
    p.add_argument('--out-dir', default='.', help='Directory for output figure/JSON')
    return p.parse_args()

args = _args()

# ── Load and preprocess ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(args.data)
df = df[df['ObjID'].notna() & df['Type'].isin(CLASSES)].copy()
df['Type'] = df['Type'].astype(int)
df[FEATS]  = df[FEATS].fillna(0)
cls2idx    = {c: i for i, c in enumerate(CLASSES)}
df['label'] = df['Type'].map(cls2idx)

scaler = StandardScaler()
df[FEATS] = scaler.fit_transform(df[FEATS])

tracks_df  = df.groupby('ObjID').agg(
    label=('label', 'first'), n_det=('ObjID', 'count')).reset_index()
tracks_df  = tracks_df[tracks_df['n_det'] >= MIN_LEN].reset_index(drop=True)
all_ids    = tracks_df['ObjID'].values
all_labels = tracks_df['label'].values
print(f"Tracks: {len(tracks_df)}")
for i, name in enumerate(CLS_NAMES):
    print(f"  {name}: {(all_labels==i).sum()}")


# ── Extract fractal features ───────────────────────────────────────────────────
FRAC_BASE = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc']

print("\nExtracting fractal features (this may take ~30s)...")
t0 = time.time()
frac_rows = []
for oid in all_ids:
    sub = df[df['ObjID'] == oid]
    n   = len(sub)
    row = {'track_len': n}
    for feat in FRAC_BASE:
        x = sub[feat].values.astype(float)
        row[f'{feat}_petrosian'] = petrosian_fd(x)
        row[f'{feat}_higuchi']   = higuchi_fd(x)
        row[f'{feat}_sampent']   = sample_entropy(x)
    # Hurst only on the two RCS features (most physically grounded)
    row['lpk_hurst']  = hurst_rs(sub['log_peak_rcs'].values.astype(float))
    row['ltot_hurst'] = hurst_rs(sub['log_total_rcs'].values.astype(float))
    frac_rows.append(row)

frac_df   = pd.DataFrame(frac_rows)
frac_cols = [c for c in frac_df.columns if c != 'track_len']
print(f"  Done in {time.time()-t0:.1f}s   |   {len(frac_cols)} fractal features")


# ── Coverage report ────────────────────────────────────────────────────────────
print(f"\n{'Feature':<35}", end="")
for name in CLS_NAMES:
    print(f"  {name:>9}", end="")
print()
print("-" * 75)
for col in frac_cols:
    print(f"{col:<35}", end="")
    for i in range(N_CLS):
        mask = all_labels == i
        pct  = frac_df.loc[mask, col].notna().mean()
        print(f"  {pct:>8.0%} ", end="")
    print()


# ── Cohen's d analysis ─────────────────────────────────────────────────────────
def cohens_d(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) /
                 (len(a)+len(b)-2))
    return float(abs(np.mean(a) - np.mean(b)) / sp) if sp > 0 else np.nan

pair_labels = [('Fis','Tug'), ('Fis','Car'), ('Fis','Tan'),
               ('Tug','Car'), ('Tug','Tan'), ('Car','Tan')]
pair_idx    = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

print(f"\nCohen's d — fractal features (Car--Tan is the target pair):")
print(f"{'Feature':<35}", end="")
for a, b in pair_labels:
    print(f"  {a}--{b}", end="")
print()
print("-" * 80)

ct_d = {}   # Cargo-Tanker d per fractal feature
for col in frac_cols:
    vals = [frac_df.loc[all_labels == i, col].values for i in range(4)]
    ds   = [cohens_d(vals[i], vals[j]) for i, j in pair_idx]
    ct_d[col] = ds[-1]
    row_str = f"{col:<35}"
    for d in ds:
        row_str += f"  {'N/A':>7}" if np.isnan(d) else f"  {d:>7.2f}"
    # Highlight Car-Tan
    ct_val = ds[-1]
    marker = " <<<" if (not np.isnan(ct_val) and ct_val > 0.56) else ""
    print(row_str + marker)

valid_ct  = {k: v for k, v in ct_d.items() if not np.isnan(v)}
if valid_ct:
    best_col  = max(valid_ct, key=valid_ct.get)
    best_d    = valid_ct[best_col]
    print(f"\nBaseline ceiling (amplitude features): d = 0.56  (cross_range_extent)")
    print(f"Best fractal Car-Tanker d:             d = {best_d:.3f}  ({best_col})")
    delta = best_d - 0.56
    print(f"Delta vs baseline:                     {delta:+.3f}  "
          f"({'IMPROVEMENT' if delta>0 else 'no improvement'})")


# ── XGBoost: track stats only (baseline) vs + fractal ─────────────────────────
stats_cols = ['mean','std','min','max','q25','q75','range','skew','kurt']

def track_stats_features(df, oid_list):
    rows = []
    for oid in oid_list:
        sub = df[df['ObjID'] == oid][FEATS].values
        n   = len(sub)
        row = []
        for fi in range(N_FEAT):
            v  = sub[:, fi]
            sk = float(sp_stats.skew(v))      if n > 2 else 0.0
            ku = float(sp_stats.kurtosis(v))  if n > 3 else 0.0
            row.extend([np.mean(v), np.std(v), np.min(v), np.max(v),
                        np.percentile(v,25), np.percentile(v,75),
                        np.ptp(v), sk, ku])
        rng_ = sub[:, FEATS.index('range')]
        ar_  = sub[:, FEATS.index('aspect_ratio')]
        fp_  = sub[:, FEATS.index('footprint_m2')]
        lpk_ = sub[:, FEATS.index('log_peak_rcs')]
        row += [float(n),
                float(np.std(rng_)),
                float(np.mean(ar_) * np.mean(fp_)),
                float(np.std(lpk_) / (abs(np.mean(lpk_)) + 1e-6)),
                float(np.median(rng_))]
        rows.append(row)
    return np.array(rows, dtype=np.float32)

print("\nBuilding feature matrices...", flush=True)
X_stats = track_stats_features(df, all_ids)

# Fractal matrix — median-impute NaN
X_frac = frac_df[frac_cols].values.astype(np.float32)
for j in range(X_frac.shape[1]):
    col = X_frac[:, j]
    col[np.isnan(col)] = np.nanmedian(col)

X_combined = np.hstack([X_stats, X_frac])
print(f"  Baseline features : {X_stats.shape[1]}")
print(f"  Fractal features  : {X_frac.shape[1]}")
print(f"  Combined features : {X_combined.shape[1]}")

# XGBoost hyperparameters (same as tuned version)
XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.75, colsample_bytree=0.7, min_child_weight=2,
    gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=SEED, verbosity=0, n_jobs=-1
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def run_cv(X, y, tag):
    oof_preds = np.full(len(y), -1, dtype=int)
    accs, f1s = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        cw = compute_class_weight('balanced', classes=np.arange(N_CLS), y=y_tr)
        sw = np.array([cw[l] for l in y_tr])
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_te, y_te)], verbose=False)
        pred = clf.predict(X_te)
        oof_preds[te_idx] = pred
        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, average='macro'))
    oof_acc = accuracy_score(y, oof_preds)
    oof_f1  = f1_score(y, oof_preds, average='macro')
    print(f"\n  {tag}")
    print(f"    CV  Acc {np.mean(accs):.1%} ± {np.std(accs):.1%}   "
          f"F1 {np.mean(f1s):.1%} ± {np.std(f1s):.1%}")
    print(f"    OOF Acc {oof_acc:.1%}                    F1 {oof_f1:.1%}")
    return oof_preds, oof_acc, oof_f1, accs, f1s

y_all = all_labels
print("\n" + "="*65)
print("5-fold stratified CV")
print("="*65)
preds_base, base_acc, base_f1, base_accs, base_f1s = run_cv(
    X_stats,    y_all, "XGB — track stats only (baseline)")
preds_comb, comb_acc, comb_f1, comb_accs, comb_f1s = run_cv(
    X_combined, y_all, "XGB — track stats + fractal features")


# ── Cargo-Tanker breakdown ─────────────────────────────────────────────────────
def ct_report(preds, y, tag):
    mask = np.isin(y, [2, 3])
    y_ct = y[mask]; p_ct = preds[mask]
    cm   = confusion_matrix(y_ct, p_ct, labels=[2, 3])
    r0   = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0
    r1   = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0
    print(f"  {tag}")
    print(f"    Cargo  → Cargo {cm[0,0]:3d}  Tanker {cm[0,1]:3d}   recall={r0:.2f}")
    print(f"    Tanker → Cargo {cm[1,0]:3d}  Tanker {cm[1,1]:3d}   recall={r1:.2f}")

print("\nCargo-Tanker sub-confusion:")
ct_report(preds_base, y_all, "Baseline")
ct_report(preds_comb, y_all, "+Fractal")

print("\nFull classification reports:")
print("Baseline OOF:")
print(classification_report(y_all, preds_base, target_names=CLS_NAMES, zero_division=0))
print("+Fractal OOF:")
print(classification_report(y_all, preds_comb, target_names=CLS_NAMES, zero_division=0))


# ── Feature importance of fractal features (full-data model) ──────────────────
cw_full  = compute_class_weight('balanced', classes=np.arange(N_CLS), y=y_all)
sw_full  = np.array([cw_full[l] for l in y_all])
clf_full = xgb.XGBClassifier(**XGB_PARAMS)
clf_full.fit(X_combined, y_all, sample_weight=sw_full, verbose=False)

imp = clf_full.feature_importances_
frac_start = X_stats.shape[1]
frac_imp   = sorted(
    [(frac_cols[i], float(imp[frac_start + i])) for i in range(len(frac_cols))],
    key=lambda x: x[1], reverse=True
)
print("Fractal feature importances (trained on full data):")
for name, val in frac_imp:
    bar = '█' * int(val * 400)
    print(f"  {name:<35} {val:.4f}  {bar}")


# ── Save results ───────────────────────────────────────────────────────────────
results = {
    'baseline_oof_acc': float(base_acc),
    'baseline_oof_f1':  float(base_f1),
    'fractal_oof_acc':  float(comb_acc),
    'fractal_oof_f1':   float(comb_f1),
    'delta_acc':        float(comb_acc - base_acc),
    'delta_f1':         float(comb_f1  - base_f1),
    'best_ct_fractal_d':      float(best_d)   if valid_ct else None,
    'best_ct_fractal_feature': best_col        if valid_ct else None,
    'baseline_ct_d':    0.56,
    'fractal_ct_d_all': {k: (float(v) if not np.isnan(v) else None)
                         for k, v in ct_d.items()},
}
with open('/tmp/fractal_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved → /tmp/fractal_results.json")


# ── Figure ──────────────────────────────────────────────────────────────────────
def short(s):
    return (s.replace('log_peak_rcs','lpk')
             .replace('log_total_rcs','ltot')
             .replace('rcs_conc','conc')
             .replace('_rcs',''))

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

# (a) Cohen's d for Car-Tan per fractal feature
ct_sorted = sorted([(k, v) for k, v in valid_ct.items()], key=lambda x: x[1])
names_s   = [short(k) for k, _ in ct_sorted]
vals_s    = [v for _, v in ct_sorted]
colors_s  = ['#d73027' if v > 0.56 else '#4393c3' for v in vals_s]
axes[0,0].barh(range(len(names_s)), vals_s, color=colors_s)
axes[0,0].axvline(0.56, color='black', ls='--', lw=1.2, label='Amplitude ceiling (d=0.56)')
axes[0,0].set_yticks(range(len(names_s)))
axes[0,0].set_yticklabels(names_s, fontsize=7)
axes[0,0].set_xlabel("Cohen's d  (Cargo vs Tanker)")
axes[0,0].set_title("(a) Fractal features: Cargo–Tanker d")
axes[0,0].legend(fontsize=7)

# (b) Distribution of best fractal feature
if valid_ct:
    colors_cls = ['#1b7837','#762a83','#e66101','#5e3c99']
    for i, (cls, col) in enumerate(zip(CLS_NAMES, colors_cls)):
        mask = all_labels == i
        vals_i = frac_df.loc[mask, best_col].dropna().values
        if len(vals_i) > 2:
            axes[0,1].hist(vals_i, bins=20, alpha=0.6, label=f"{cls} (n={len(vals_i)})",
                           color=col, density=True)
    axes[0,1].set_xlabel(short(best_col))
    axes[0,1].set_title(f"(b) Best fractal feature distribution")
    axes[0,1].legend(fontsize=7)

# (c) Per-fold accuracy
x = np.arange(N_FOLDS); w = 0.35
axes[0,2].bar(x-w/2, base_accs, w, label='Baseline', color='#4dac26')
axes[0,2].bar(x+w/2, comb_accs, w, label='+Fractal', color='#2166ac')
axes[0,2].axhline(np.mean(base_accs), ls='--', color='#4dac26', lw=1.0)
axes[0,2].axhline(np.mean(comb_accs), ls='--', color='#2166ac', lw=1.0)
axes[0,2].set_xticks(x); axes[0,2].set_xticklabels([f'F{i+1}' for i in range(N_FOLDS)])
axes[0,2].set_ylim(0.4, 1.0)
axes[0,2].set_ylabel('Accuracy'); axes[0,2].set_title('(c) Per-fold accuracy')
axes[0,2].legend(fontsize=8)

# (d) Baseline confusion matrix
def plot_cm(ax, preds, y, title):
    cm = confusion_matrix(y, preds).astype(float)
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_n, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_xticklabels([n[:3] for n in CLS_NAMES], fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels([n[:3] for n in CLS_NAMES], fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm_n[i,j]:.2f}', ha='center', va='center',
                    fontsize=7.5, color='white' if cm_n[i,j]>0.6 else 'black')
    return im

plot_cm(axes[1,0], preds_base, y_all, f'(d) Baseline OOF  ({base_acc:.1%})')
plot_cm(axes[1,1], preds_comb, y_all, f'(e) +Fractal OOF  ({comb_acc:.1%})')

# (f) Fractal feature importances (top 11)
top_n  = min(11, len(frac_imp))
fn     = [short(n) for n, _ in frac_imp[:top_n]]
fv     = [v for _, v in frac_imp[:top_n]]
axes[1,2].barh(range(top_n), fv[::-1], color='#762a83')
axes[1,2].set_yticks(range(top_n))
axes[1,2].set_yticklabels(fn[::-1], fontsize=7)
axes[1,2].set_xlabel('Feature importance')
axes[1,2].set_title(f'(f) Fractal feature importance (top {top_n})')

fig.tight_layout(pad=0.8)
fig_path = f'{args.out_dir}/fig_fractal_analysis.pdf'
fig.savefig(fig_path, bbox_inches='tight', dpi=150)
print(f"Figure saved → {fig_path}")
