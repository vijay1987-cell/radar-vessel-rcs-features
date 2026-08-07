"""
LID and multiscale density across feature subspaces.
Tests whether Cargo-Tanker neighbourhood separation improves
when LID is computed in amplitude-only, geometry-only, or 5-RCS subspace
vs the full 8D space.
"""
import warnings, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

ALL_FEATS = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc',
             'aspect_ratio', 'footprint_m2',
             'range', 'down_range_extent', 'cross_range_extent']

SUBSPACES = {
    'Amplitude (3D)'   : ['log_peak_rcs', 'log_total_rcs', 'rcs_conc'],
    'Geometry (4D)'    : ['aspect_ratio', 'footprint_m2',
                          'down_range_extent', 'cross_range_extent'],
    'RCS-5 (5D)'       : ['log_peak_rcs', 'log_total_rcs', 'rcs_conc',
                          'aspect_ratio', 'footprint_m2'],
    'No-range (7D)'    : ['log_peak_rcs', 'log_total_rcs', 'rcs_conc',
                          'aspect_ratio', 'footprint_m2',
                          'down_range_extent', 'cross_range_extent'],
    'Full (8D)'        : ALL_FEATS,
}

CLASSES   = [30, 52, 70, 80]
CLS_NAMES = ['Fishing', 'Tug', 'Cargo', 'Tanker']
K_VALS    = [5, 10, 20, 40]
BATCH     = 512


# ── Core estimators ────────────────────────────────────────────────────────────

def _lid_mle(drow, k):
    d = drow[:k]
    d = d[d > 1e-12]
    if len(d) < 3:
        return np.nan
    d_max = d[-1]
    if d_max < 1e-12:
        return np.nan
    log_ratios = np.log(d[:-1] / d_max)
    mean_lr = np.mean(log_ratios)
    if mean_lr >= -1e-10:
        return np.nan
    lid = -1.0 / mean_lr
    return float(lid) if np.isfinite(lid) else np.nan


def compute_ss(X_query, X_ref, k_vals, radii):
    n_q   = len(X_query)
    n_k   = len(k_vals)
    n_r   = len(radii)
    log_r = np.log(np.array(radii, dtype=float))

    lid  = np.full((n_q, n_k), np.nan, dtype=np.float32)
    dens = np.full((n_q, n_r), np.nan, dtype=np.float32)
    dim  = np.full(n_q,        np.nan, dtype=np.float32)

    for s in range(0, n_q, BATCH):
        e = min(s + BATCH, n_q)
        D = cdist(X_query[s:e], X_ref, metric='euclidean')
        for bi in range(e - s):
            drow = np.sort(D[bi])
            drow = drow[drow > 1e-12]
            for ki, k in enumerate(k_vals):
                lid[s + bi, ki] = _lid_mle(drow, k)
            cnts = np.array([(drow <= r).sum() for r in radii], dtype=float)
            dens[s + bi] = cnts
            valid = cnts > 0
            if valid.sum() >= 3:
                slope, _ = np.polyfit(log_r[valid], np.log(cnts[valid] + 1), 1)
                dim[s + bi] = float(slope)

    names = ([f'lid_k{k}' for k in k_vals] +
             [f'dens_r{i+1}' for i in range(n_r)] +
             ['corr_dim'])
    return np.hstack([lid, dens, dim.reshape(-1, 1)]), names


def cohens_d(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) /
                 (len(a)+len(b)-2))
    return float(abs(np.mean(a)-np.mean(b))/sp) if sp > 0 else np.nan


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('/home/iaxiom/Downloads/rcs_train_4class.csv')
df = df[df['ObjID'].notna() & df['Type'].isin(CLASSES)].copy()
df['Type']  = df['Type'].astype(int)
df[ALL_FEATS] = df[ALL_FEATS].fillna(0)
df['label'] = df['Type'].map({c: i for i, c in enumerate(CLASSES)})

scaler = StandardScaler()
df[ALL_FEATS] = scaler.fit_transform(df[ALL_FEATS])

y = df['label'].values
print(f"Detections: {len(df)}  |  "
      + "  ".join(f"{n}:{(y==i).sum()}" for i,n in enumerate(CLS_NAMES)))

pair_labels = [('Fis','Tug'),('Fis','Car'),('Fis','Tan'),
               ('Tug','Car'),('Tug','Tan'),('Car','Tan')]
pair_idx    = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]


# ── Run per subspace ───────────────────────────────────────────────────────────
results = {}   # subspace_name → {feature_name: [d_pair0..d_pair5]}

print(f"\n{'─'*72}")
print(f"{'Subspace':<18}  {'Feature':<15}"
      + "".join(f"  {a}--{b}" for a,b in pair_labels))
print(f"{'─'*72}")

for sp_name, feats in SUBSPACES.items():
    X = df[feats].values.astype(np.float32)

    # Data-adaptive radii for this subspace
    idx_s  = np.random.choice(len(X), 2000, replace=False)
    D_s    = cdist(X[idx_s], X[idx_s])
    D_flat = D_s[D_s > 1e-12].ravel()
    radii  = [float(np.percentile(D_flat, p)) for p in [10, 25, 50, 75]]

    t0 = time.time()
    X_ss, ss_names = compute_ss(X, X, K_VALS, radii)
    elapsed = time.time() - t0

    # Replace inf with nan
    X_ss[~np.isfinite(X_ss)] = np.nan

    sp_results = {}
    for fi, col in enumerate(ss_names):
        vals = [X_ss[y==i, fi] for i in range(4)]
        ds   = [cohens_d(vals[i], vals[j]) for i,j in pair_idx]
        sp_results[col] = ds

    results[sp_name] = sp_results

    # Print best LID and best density feature for this subspace
    ct_by_feat = {col: ds[-1] for col, ds in sp_results.items()
                  if not np.isnan(ds[-1])}
    best_feat  = max(ct_by_feat, key=ct_by_feat.get) if ct_by_feat else 'N/A'
    best_ds    = sp_results.get(best_feat, [np.nan]*6)

    print(f"\n  [{sp_name}]  ({len(feats)}D, {elapsed:.1f}s)  radii≈"
          f"[{radii[0]:.2f},{radii[1]:.2f},{radii[2]:.2f},{radii[3]:.2f}]")
    for col in ss_names:
        ds = sp_results[col]
        row = f"  {'':18}  {col:<15}"
        for d in ds:
            mark = '**' if d == ds[-1] and not np.isnan(d) and d > 0.56 else '  '
            row += f"  {mark}{d:5.2f}{mark}" if not np.isnan(d) else f"    N/A  "
        print(row)


# ── Summary table: best Car-Tan d per subspace ────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY — Best Cargo-Tanker Cohen's d per subspace and feature type")
print(f"{'='*65}")
print(f"{'Subspace':<18}  {'Best LID feat':<14}  {'d':>6}  "
      f"{'Best density feat':<18}  {'d':>6}  {'Best corr_dim d':>15}")
print(f"{'─'*85}")

for sp_name, sp_results in results.items():
    lid_feats  = {k: v[-1] for k,v in sp_results.items()
                  if k.startswith('lid') and not np.isnan(v[-1])}
    dens_feats = {k: v[-1] for k,v in sp_results.items()
                  if k.startswith('dens') and not np.isnan(v[-1])}
    dim_d      = sp_results.get('corr_dim', [np.nan]*6)[-1]

    best_lid  = max(lid_feats,  key=lid_feats.get)  if lid_feats  else 'N/A'
    best_dens = max(dens_feats, key=dens_feats.get) if dens_feats else 'N/A'
    bl_d = lid_feats.get(best_lid,  np.nan)
    bd_d = dens_feats.get(best_dens, np.nan)

    mark_l = ' <<<' if bl_d  > 0.56 else ''
    mark_d = ' <<<' if bd_d  > 0.56 else ''
    mark_c = ' <<<' if dim_d > 0.56 else ''

    print(f"  {sp_name:<18}  {best_lid:<14}  {bl_d:>6.3f}{mark_l}  "
          f"{best_dens:<18}  {bd_d:>6.3f}{mark_d}  "
          f"{dim_d:>10.3f}{mark_c}")

print(f"\n  Amplitude ceiling (cross_range_extent): d = 0.560")


# ── Figure: Car-Tan d heatmap across subspaces × features ─────────────────────
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: heatmap of Car-Tan d
sp_names  = list(results.keys())
feat_names = list(next(iter(results.values())).keys())
mat = np.array([[results[sp].get(f, [np.nan]*6)[-1]
                 for f in feat_names] for sp in sp_names])
mat_plot = np.where(np.isnan(mat), 0, mat)

im = axes[0].imshow(mat_plot, cmap='RdYlGn', vmin=0, vmax=0.8, aspect='auto')
axes[0].set_xticks(range(len(feat_names)))
axes[0].set_xticklabels(
    [f.replace('lid_k','LID-k').replace('dens_r','D-r').replace('corr_dim','CorrD')
     for f in feat_names], rotation=45, ha='right', fontsize=7)
axes[0].set_yticks(range(len(sp_names)))
axes[0].set_yticklabels(sp_names, fontsize=8)
axes[0].set_title("(a) Cargo–Tanker Cohen's d  per subspace × feature\n"
                  "(green = high separation; red line = 0.56 ceiling)")
plt.colorbar(im, ax=axes[0], label="Cohen's d")
# Add ceiling line annotation
axes[0].axhline(-0.5, color='none')  # padding
for i in range(len(sp_names)):
    for j in range(len(feat_names)):
        v = mat[i, j]
        txt = f'{v:.2f}' if not np.isnan(v) else 'N/A'
        col = 'white' if mat_plot[i,j] > 0.5 else 'black'
        axes[0].text(j, i, txt, ha='center', va='center', fontsize=6.5, color=col)

# Right: bar chart — best Car-Tan d per subspace (across all self-sim features)
best_per_sp = []
for sp_name in sp_names:
    ct_vals = [v[-1] for v in results[sp_name].values() if not np.isnan(v[-1])]
    best_per_sp.append(max(ct_vals) if ct_vals else 0.0)

colors = ['#d73027' if v > 0.56 else '#4393c3' for v in best_per_sp]
axes[1].barh(range(len(sp_names)), best_per_sp, color=colors)
axes[1].axvline(0.56, color='black', ls='--', lw=1.2,
                label='Amplitude ceiling (d=0.56)')
axes[1].set_yticks(range(len(sp_names)))
axes[1].set_yticklabels(sp_names, fontsize=8)
axes[1].set_xlabel("Best Cargo–Tanker Cohen's d")
axes[1].set_title("(b) Best self-similarity d per subspace")
axes[1].legend(fontsize=8)
for i, v in enumerate(best_per_sp):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

fig.tight_layout(pad=0.8)
fig_path = ('/home/iaxiom/projects/Hqnn/paper/submission_ready/'
            'figures/fig_lid_subspace.pdf')
fig.savefig(fig_path, bbox_inches='tight', dpi=150)
print(f"\nFigure saved → {fig_path}")
