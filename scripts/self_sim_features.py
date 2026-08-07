"""
Per-detection self-similarity features via Local Intrinsic Dimensionality (LID)
and multiscale neighbourhood density (correlation dimension).

Hypothesis: Cargo and Tanker detections occupy geometrically distinct regions
of the 8D feature space with different local fractal structure — even though
their marginal distributions overlap (amplitude ceiling d=0.56).

Two evaluations:
  1. Cohen's d on raw LID/density features (does self-similarity separate classes?)
  2. GBT per-detection soft-vote  — baseline vs +self-sim (proper CV: LID for
     test detections computed using ONLY training detections as reference)

Usage:
    python scripts/self_sim_features.py --data /path/to/rcs_train_4class.csv
"""
import argparse, json, warnings, time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED      = 42
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
K_VALS    = [5, 10, 20, 40]     # LID at 4 neighbourhood scales
BATCH     = 512                  # rows per cdist chunk (~25 MB each)


# ── Self-similarity feature engine ────────────────────────────────────────────

def _lid_mle(dists_sorted, k):
    """MLE estimator for Local Intrinsic Dimensionality (Amsaleg et al. 2015)."""
    d = dists_sorted[:k]
    d = d[d > 1e-12]
    if len(d) < 3:
        return np.nan
    d_max = d[-1]
    if d_max < 1e-12:
        return np.nan
    # Use d[:-1] to exclude the boundary point (log(d_k/d_k)=0 biases mean)
    log_ratios = np.log(d[:-1] / d_max)
    if len(log_ratios) == 0:
        return np.nan
    mean_lr = np.mean(log_ratios)
    if mean_lr >= -1e-10:   # degenerate: all neighbours at same distance
        return np.nan
    lid = float(-1.0 / mean_lr)
    return lid if np.isfinite(lid) else np.nan


def compute_self_sim(X_query, X_ref, k_vals, radii):
    """
    For each row in X_query, compute:
      - LID at each k in k_vals   (local fractal dimension via MLE)
      - N(r) for each r in radii  (neighbour count at each scale)
      - corr_dim                  (slope of log N(r) vs log r — correlation dimension)

    X_ref is the reference point cloud (training detections in CV mode).
    X_query points that exist in X_ref have their self-distance excluded
    automatically (distances < 1e-12 are ignored).

    Returns
    -------
    X_ss : ndarray (n_query, n_features)
    names : list[str]
    """
    n_q    = len(X_query)
    n_k    = len(k_vals)
    n_r    = len(radii)
    log_r  = np.log(np.array(radii, dtype=float))
    max_k  = max(k_vals)

    lid_arr  = np.full((n_q, n_k), np.nan, dtype=np.float32)
    dens_arr = np.full((n_q, n_r), np.nan, dtype=np.float32)
    dim_arr  = np.full(n_q,        np.nan, dtype=np.float32)

    for start in range(0, n_q, BATCH):
        end   = min(start + BATCH, n_q)
        D     = cdist(X_query[start:end], X_ref, metric='euclidean')

        for bi in range(end - start):
            drow = np.sort(D[bi])
            drow = drow[drow > 1e-12]           # exclude self-distance

            # LID at multiple k
            for ki, k in enumerate(k_vals):
                lid_arr[start + bi, ki] = _lid_mle(drow, k)

            # Neighbourhood counts at multiple radii
            cnts = np.array([(drow <= r).sum() for r in radii], dtype=float)
            dens_arr[start + bi] = cnts

            # Correlation dimension: slope of log(N(r)+1) vs log(r)
            valid = cnts > 0
            if valid.sum() >= 3:
                slope, _ = np.polyfit(log_r[valid],
                                      np.log(cnts[valid] + 1), 1)
                dim_arr[start + bi] = float(slope)

    names = ([f'lid_k{k}' for k in k_vals] +
             [f'dens_r{i+1}' for i in range(n_r)] +
             ['corr_dim'])
    return np.hstack([lid_arr, dens_arr, dim_arr.reshape(-1, 1)]), names


def impute_median(X_tr, X_te=None):
    """In-place median imputation. Converts inf→nan first, then fills with median."""
    for arr in ([X_tr] if X_te is None else [X_tr, X_te]):
        arr[np.isinf(arr)] = np.nan
    for j in range(X_tr.shape[1]):
        med = np.nanmedian(X_tr[:, j])
        med = 0.0 if np.isnan(med) else float(med)
        X_tr[~np.isfinite(X_tr[:, j]), j] = med
        if X_te is not None:
            X_te[~np.isfinite(X_te[:, j]), j] = med


# ── CLI ───────────────────────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to rcs_train_4class.csv')
    p.add_argument('--out-dir', default='.', help='Directory for output figure/JSON')
    return p.parse_args()

args = _args()

# ── Load and standardise ───────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(args.data)
df = df[df['ObjID'].notna() & df['Type'].isin(CLASSES)].copy()
df['Type']  = df['Type'].astype(int)
df[FEATS]   = df[FEATS].fillna(0)
df['label'] = df['Type'].map({c: i for i, c in enumerate(CLASSES)})

scaler = StandardScaler()
df[FEATS] = scaler.fit_transform(df[FEATS])

X_all      = df[FEATS].values.astype(np.float32)
y_all_det  = df['label'].values
oid_all    = df['ObjID'].values

print(f"Detections: {len(df)}")
for i, n in enumerate(CLS_NAMES):
    print(f"  {n}: {(y_all_det==i).sum()}")

# Track index
tracks_df  = df.groupby('ObjID').agg(
    label=('label','first'), n_det=('ObjID','count')).reset_index()
tracks_df  = tracks_df[tracks_df['n_det'] >= MIN_LEN].reset_index(drop=True)
all_ids    = tracks_df['ObjID'].values
all_labels = tracks_df['label'].values


# ── Data-adaptive radius grid ──────────────────────────────────────────────────
print("\nEstimating radius grid from 2000-point sample...")
idx_s  = np.random.choice(len(X_all), 2000, replace=False)
D_s    = cdist(X_all[idx_s], X_all[idx_s])
D_flat = D_s[D_s > 1e-12].ravel()
RADII  = [float(np.percentile(D_flat, p)) for p in [10, 25, 50, 75]]
print(f"  Radii (10th/25th/50th/75th pct): {[f'{r:.3f}' for r in RADII]}")


# ── Full-dataset self-sim (for Cohen's d only — no label leakage) ─────────────
print(f"\nComputing self-similarity for all {len(X_all)} detections...")
t0 = time.time()
X_ss_full, SS_NAMES = compute_self_sim(X_all, X_all, K_VALS, RADII)
impute_median(X_ss_full)
print(f"  Done in {time.time()-t0:.1f}s  |  {X_ss_full.shape[1]} features: {SS_NAMES}")


# ── Cohen's d ─────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3: return np.nan
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) /
                 (len(a)+len(b)-2))
    return float(abs(np.mean(a)-np.mean(b))/sp) if sp > 0 else np.nan

pair_labels = [('Fis','Tug'),('Fis','Car'),('Fis','Tan'),
               ('Tug','Car'),('Tug','Tan'),('Car','Tan')]
pair_idx    = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

print(f"\nCohen's d — self-similarity features  "
      f"(Car--Tan = target; amplitude ceiling = 0.56):")
print(f"{'Feature':<18}", end="")
for a, b in pair_labels: print(f"  {a}--{b}", end="")
print()
print("-"*78)

ct_d = {}
all_d_table = []
for fi, col in enumerate(SS_NAMES):
    vals = [X_ss_full[y_all_det==i, fi] for i in range(4)]
    ds   = [cohens_d(vals[i], vals[j]) for i,j in pair_idx]
    ct_d[col] = ds[-1]
    all_d_table.append(ds)
    row = f"{col:<18}"
    for d in ds:
        row += f"  {'N/A':>7}" if np.isnan(d) else f"  {d:>7.2f}"
    mark = "  <<<< EXCEEDS CEILING" if (not np.isnan(ds[-1]) and ds[-1]>0.56) else ""
    print(row + mark)

print(f"\n  [amplitude best]      "
      + "".join(f"  {'':>7}" for _ in range(5))
      + "     0.56")

valid_ct = {k:v for k,v in ct_d.items() if not np.isnan(v)}
best_col  = max(valid_ct, key=valid_ct.get)
best_d    = valid_ct[best_col]
print(f"\nBest self-similarity Car--Tan d : {best_d:.3f}  ({best_col})")
print(f"Amplitude ceiling               : 0.560")
print(f"Delta                           : {best_d-0.56:+.3f}  "
      f"({'IMPROVEMENT' if best_d>0.56 else 'no improvement'})")


# ── GBT per-detection soft-vote — proper CV ────────────────────────────────────
# In each fold: test detections' LID is computed using ONLY training detections
# as reference → no leakage from test fold into self-sim features.
print("\n" + "="*65)
print("GBT per-detection soft-vote — 5-fold CV (proper LID computation)")
print("="*65)

GBT_P = dict(n_estimators=300, max_depth=5, random_state=SEED)
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

base_accs, base_f1s = [], []
ss_accs,   ss_f1s   = [], []
oof_base = np.full(len(all_ids), -1, dtype=int)
oof_ss   = np.full(len(all_ids), -1, dtype=int)

for fold, (tr_idx, te_idx) in enumerate(skf.split(all_ids, all_labels)):
    tr_oids = all_ids[tr_idx]
    te_oids = all_ids[te_idx]

    tr_mask = np.isin(oid_all, tr_oids)
    te_mask = np.isin(oid_all, te_oids)

    X_tr = X_all[tr_mask];  y_tr = y_all_det[tr_mask]
    X_te = X_all[te_mask]
    oid_tr = oid_all[tr_mask]
    oid_te = oid_all[te_mask]

    # Self-sim: train uses itself as reference; test uses TRAIN as reference
    t0 = time.time()
    X_tr_ss, _ = compute_self_sim(X_tr, X_tr, K_VALS, RADII)
    X_te_ss, _ = compute_self_sim(X_te, X_tr, K_VALS, RADII)
    impute_median(X_tr_ss, X_te_ss)

    X_tr_aug = np.hstack([X_tr, X_tr_ss])
    X_te_aug = np.hstack([X_te, X_te_ss])
    # Final guard: clip any residual extreme values
    np.clip(X_tr_aug, -1e6, 1e6, out=X_tr_aug)
    np.clip(X_te_aug, -1e6, 1e6, out=X_te_aug)
    elapsed  = time.time() - t0

    # Train both models
    gbt_base = GradientBoostingClassifier(**GBT_P)
    gbt_base.fit(X_tr, y_tr)

    gbt_ss = GradientBoostingClassifier(**GBT_P)
    gbt_ss.fit(X_tr_aug, y_tr)

    # Predict per track via soft vote
    preds_base_fold, preds_ss_fold, trues_fold = [], [], []
    for oi, (oid, lbl) in enumerate(zip(te_oids, all_labels[te_idx])):
        idx = np.where(oid_te == oid)[0]
        if len(idx) == 0: continue

        pr_b = gbt_base.predict_proba(X_te[idx]).mean(axis=0)
        pr_s = gbt_ss.predict_proba(X_te_aug[idx]).mean(axis=0)

        pb, ps = int(np.argmax(pr_b)), int(np.argmax(pr_s))
        preds_base_fold.append(pb);  oof_base[te_idx[oi]] = pb
        preds_ss_fold.append(ps);    oof_ss[te_idx[oi]]   = ps
        trues_fold.append(lbl)

    fa = accuracy_score(trues_fold, preds_base_fold)
    fb = accuracy_score(trues_fold, preds_ss_fold)
    base_accs.append(fa);  ss_accs.append(fb)
    base_f1s.append(f1_score(trues_fold, preds_base_fold, average='macro'))
    ss_f1s.append(f1_score(trues_fold, preds_ss_fold,   average='macro'))
    print(f"  Fold {fold+1}  LID-time={elapsed:.1f}s  "
          f"base={fa:.3f}  +self-sim={fb:.3f}  "
          f"delta={fb-fa:+.3f}")

valid_mask = oof_base >= 0
oof_base_acc = accuracy_score(all_labels[valid_mask], oof_base[valid_mask])
oof_ss_acc   = accuracy_score(all_labels[valid_mask], oof_ss[valid_mask])
oof_base_f1  = f1_score(all_labels[valid_mask], oof_base[valid_mask], average='macro')
oof_ss_f1    = f1_score(all_labels[valid_mask], oof_ss[valid_mask],   average='macro')

print(f"\n{'─'*65}")
print(f"{'Method':<35}  {'CV Acc':>8}  {'OOF Acc':>8}  {'OOF F1':>8}")
print(f"{'─'*65}")
print(f"{'GBT soft-vote (baseline)':<35}  "
      f"{np.mean(base_accs):>7.1%}  {oof_base_acc:>8.1%}  {oof_base_f1:>8.1%}")
print(f"{'GBT + self-similarity':<35}  "
      f"{np.mean(ss_accs):>7.1%}  {oof_ss_acc:>8.1%}  {oof_ss_f1:>8.1%}")
print(f"{'Delta':<35}  "
      f"{np.mean(ss_accs)-np.mean(base_accs):>+7.1%}  "
      f"{oof_ss_acc-oof_base_acc:>+8.1%}  "
      f"{oof_ss_f1-oof_base_f1:>+8.1%}")


# ── Cargo-Tanker breakdown ─────────────────────────────────────────────────────
def ct_sub(preds, labels, tag):
    mask = np.isin(labels, [2,3]) & (preds >= 0)
    cm   = confusion_matrix(labels[mask], preds[mask], labels=[2,3])
    r0   = cm[0,0]/cm[0].sum() if cm[0].sum()>0 else 0
    r1   = cm[1,1]/cm[1].sum() if cm[1].sum()>0 else 0
    print(f"  {tag}")
    print(f"    Cargo  → Cargo {cm[0,0]:3d}  Tanker {cm[0,1]:3d}   recall={r0:.2f}")
    print(f"    Tanker → Cargo {cm[1,0]:3d}  Tanker {cm[1,1]:3d}   recall={r1:.2f}")

print("\nCargo-Tanker sub-confusion (OOF):")
ct_sub(oof_base, all_labels, "Baseline")
ct_sub(oof_ss,   all_labels, "+Self-sim")

print("\nFull classification report — baseline:")
print(classification_report(all_labels[valid_mask], oof_base[valid_mask],
      target_names=CLS_NAMES, zero_division=0))
print("Full classification report — +self-similarity:")
print(classification_report(all_labels[valid_mask], oof_ss[valid_mask],
      target_names=CLS_NAMES, zero_division=0))


# ── Save results ───────────────────────────────────────────────────────────────
results = {
    'gbt_base_cv_acc'  : float(np.mean(base_accs)),
    'gbt_base_oof_acc' : float(oof_base_acc),
    'gbt_base_oof_f1'  : float(oof_base_f1),
    'gbt_ss_cv_acc'    : float(np.mean(ss_accs)),
    'gbt_ss_oof_acc'   : float(oof_ss_acc),
    'gbt_ss_oof_f1'    : float(oof_ss_f1),
    'delta_oof_acc'    : float(oof_ss_acc - oof_base_acc),
    'delta_oof_f1'     : float(oof_ss_f1  - oof_base_f1),
    'best_ct_lid_d'    : float(best_d),
    'best_ct_feature'  : best_col,
    'amplitude_ceiling': 0.56,
    'ct_d_all'         : {k: (float(v) if not np.isnan(v) else None)
                          for k,v in ct_d.items()},
}
with open('/tmp/self_sim_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved → /tmp/self_sim_results.json")


# ── Figure ─────────────────────────────────────────────────────────────────────
def short(s):
    return s.replace('lid_k','LID-k').replace('dens_r','Dens-r').replace('corr_dim','CorrDim')

plt.rcParams.update({'font.family':'serif','font.size':9})
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

# (a) Cohen's d for Car-Tan — all self-sim features
ct_sorted = sorted(valid_ct.items(), key=lambda x: x[1])
names_s   = [short(k) for k,_ in ct_sorted]
vals_s    = [v for _,v in ct_sorted]
cols_s    = ['#d73027' if v>0.56 else '#4393c3' for v in vals_s]
axes[0,0].barh(range(len(names_s)), vals_s, color=cols_s)
axes[0,0].axvline(0.56, color='black', ls='--', lw=1.2,
                  label='Amplitude ceiling (d=0.56)')
axes[0,0].set_yticks(range(len(names_s)))
axes[0,0].set_yticklabels(names_s, fontsize=8)
axes[0,0].set_xlabel("Cohen's d  (Cargo vs Tanker)")
axes[0,0].set_title("(a) Self-similarity: Cargo–Tanker d")
axes[0,0].legend(fontsize=7)

# (b) Distribution of best self-sim feature by class
colors_cls = ['#1b7837','#762a83','#e66101','#5e3c99']
best_fi = SS_NAMES.index(best_col)
for i, (cls, col) in enumerate(zip(CLS_NAMES, colors_cls)):
    vals_i = X_ss_full[y_all_det==i, best_fi]
    vals_i = vals_i[~np.isnan(vals_i)]
    if len(vals_i) > 2:
        axes[0,1].hist(vals_i, bins=30, alpha=0.55, label=f"{cls} (n={len(vals_i)})",
                       color=col, density=True)
axes[0,1].set_xlabel(short(best_col))
axes[0,1].set_title(f"(b) Best feature distribution")
axes[0,1].legend(fontsize=7)

# (c) Cohen's d across ALL class pairs for best two self-sim features
top2     = sorted(valid_ct.items(), key=lambda x: x[1], reverse=True)[:2]
x_pairs  = np.arange(len(pair_labels)); w = 0.35
cl       = ['#2166ac','#d6604d']
for pi, (feat, _) in enumerate(top2):
    fi   = SS_NAMES.index(feat)
    d_vals = [cohens_d(X_ss_full[y_all_det==i, fi],
                       X_ss_full[y_all_det==j, fi]) for i,j in pair_idx]
    axes[0,2].bar(x_pairs + (pi-0.5)*w, d_vals, w,
                  label=short(feat), color=cl[pi], alpha=0.8)
axes[0,2].axhline(0.56, color='black', ls='--', lw=1.0, label='Amplitude ceiling')
axes[0,2].set_xticks(x_pairs)
axes[0,2].set_xticklabels([f'{a}–{b}' for a,b in pair_labels], fontsize=7)
axes[0,2].set_ylabel("Cohen's d"); axes[0,2].set_title("(c) Top-2 features all pairs")
axes[0,2].legend(fontsize=7)

# (d) Per-fold accuracy comparison
x = np.arange(N_FOLDS); w = 0.35
axes[1,0].bar(x-w/2, base_accs, w, label='Baseline',   color='#4dac26')
axes[1,0].bar(x+w/2, ss_accs,   w, label='+Self-sim',  color='#2166ac')
axes[1,0].axhline(np.mean(base_accs), ls='--', color='#4dac26', lw=1.0)
axes[1,0].axhline(np.mean(ss_accs),   ls='--', color='#2166ac', lw=1.0)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels([f'F{i+1}' for i in range(N_FOLDS)])
axes[1,0].set_ylim(0.4, 1.0)
axes[1,0].set_ylabel('Accuracy'); axes[1,0].set_title('(d) Per-fold accuracy')
axes[1,0].legend(fontsize=8)

# (e) Baseline confusion matrix
def plot_cm(ax, preds, labels, title):
    msk = preds >= 0
    cm  = confusion_matrix(labels[msk], preds[msk]).astype(float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_xticklabels([n[:3] for n in CLS_NAMES], fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels([n[:3] for n in CLS_NAMES], fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{cmn[i,j]:.2f}',ha='center',va='center',
                    fontsize=7.5,color='white' if cmn[i,j]>0.6 else 'black')

plot_cm(axes[1,1], oof_base, all_labels, f'(e) Baseline OOF ({oof_base_acc:.1%})')
plot_cm(axes[1,2], oof_ss,   all_labels, f'(f) +Self-sim OOF ({oof_ss_acc:.1%})')

fig.tight_layout(pad=0.8)
fig_path = f'{args.out_dir}/fig_self_sim_analysis.pdf'
fig.savefig(fig_path, bbox_inches='tight', dpi=150)
print(f"Figure saved → {fig_path}")
