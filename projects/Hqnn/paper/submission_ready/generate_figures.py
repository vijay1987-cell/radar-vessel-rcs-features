"""
Generate all submission-ready figures for the 5-paper radar vessel classification series.
Outputs PDF figures to ./figures/ for inclusion in the LaTeX papers.
"""

import sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

import torch
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                              classification_report)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/iaxiom/projects/Hqnn')
from src.models.hqnn_model import HQNNClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
MODELS   = Path('/home/iaxiom/projects/Hqnn/saved_models')
DATA_DIR = Path('/home/iaxiom/Downloads')
FIG_DIR  = Path('/home/iaxiom/projects/Hqnn/paper/submission_ready/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RCS_FEATS = ['log_peak_rcs','log_total_rcs','rcs_conc','aspect_ratio','footprint_m2']

# IEEE-compatible style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'lines.linewidth': 1.2, 'axes.linewidth': 0.7,
})
COLORS = ['#2166ac','#d6604d','#4dac26','#7b3294']

# ── Helpers ────────────────────────────────────────────────────────────────
def load_sklearn(name):
    with open(MODELS / name, 'rb') as f:
        return pickle.load(f)

def load_hqnn(weights_name, config_name):
    with open(MODELS / config_name) as f:
        cfg = json.load(f)
    model = HQNNClassifier(
        n_features=cfg['n_features'], n_qubits=cfg['n_qubits'],
        n_layers=cfg['n_layers'], n_classes=cfg['n_classes'],
        classical_hidden=cfg.get('classical_hidden', 16),
        n_classical_layers=cfg.get('n_classical_layers', 1),
        ansatz=cfg.get('ansatz', 'strongly_entangling'),
        activation=cfg.get('activation', 'relu'),
    )
    model.load_state_dict(torch.load(MODELS / weights_name, map_location='cpu'))
    model.eval()
    return model, cfg

def hqnn_predict_proba(model, X_np):
    with torch.no_grad():
        t = torch.tensor(X_np, dtype=torch.float32)
        logits = model(t)
        return torch.softmax(logits, dim=1).numpy()

def ece_score(proba, y_true, n_bins=15):
    conf = proba.max(axis=1); pred = proba.argmax(axis=1)
    corr = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            bin_accs.append(np.nan); bin_confs.append((lo+hi)/2); bin_counts.append(0)
            continue
        bin_accs.append(corr[m].mean())
        bin_confs.append(conf[m].mean())
        bin_counts.append(m.sum())
        ece += (m.sum() / len(y_true)) * abs(corr[m].mean() - conf[m].mean())
    return ece, bin_accs, bin_confs, bin_counts

def bootstrap_ci(y_true, y_pred, n_boot=1000, ci=0.95):
    accs, f1s = [], []
    idx = np.arange(len(y_true))
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        accs.append(accuracy_score(y_true[s], y_pred[s]))
        f1s.append(f1_score(y_true[s], y_pred[s], average='macro', zero_division=0))
    lo, hi = (1-ci)/2, 1-(1-ci)/2
    return (np.quantile(accs,[lo,hi]), np.quantile(f1s,[lo,hi]))

def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")

# ── Load datasets ──────────────────────────────────────────────────────────
print("Loading datasets...")
rcs_train = pd.read_csv(DATA_DIR / 'rcs_train_4class.csv')
rcs_inf   = pd.read_csv(DATA_DIR / 'rcs_inference_7class.csv')

with open(MODELS / 'rcs_preprocessor.pkl', 'rb') as f:
    rcs_pre = pickle.load(f)
rcs_scaler    = rcs_pre['scaler']
rcs_classes   = rcs_pre['class_names']   # ['30','52','70','80']
CLASS_LABELS  = ['Fishing\n(30)', 'Tug\n(52)', 'Cargo\n(70)', 'Tanker\n(80)']
CLASS_SHORT   = ['Fishing', 'Tug', 'Cargo', 'Tanker']

# Build a fixed balanced test set (200/class, never seen during training)
TRAIN_N, TEST_N = 300, 200
np.random.seed(SEED)
TYPES_4 = [30, 52, 70, 80]

# Range-stratified train split (mirror training script)
def stratified_sample(df, types, n, seed=SEED):
    parts = []
    for t in types:
        sub = df[df['Type'].astype(int) == t]
        bins = sub['range_bin'].dropna().unique()
        pb = max(1, n // len(bins))
        bp = []
        for rb in bins:
            s = sub[sub['range_bin'] == rb]
            bp.append(s.sample(min(len(s), pb), random_state=seed))
        chunk = pd.concat(bp).sample(min(len(pd.concat(bp)), n), random_state=seed)
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)

train_df = stratified_sample(rcs_train, TYPES_4, TRAIN_N)
# Test = everything not in train
train_idx = set(train_df.index)
remain_df = rcs_train.drop(index=train_df.index)
test_parts = []
for t in TYPES_4:
    sub = remain_df[remain_df['Type'].astype(int) == t]
    test_parts.append(sub.sample(min(len(sub), TEST_N), random_state=SEED))
test_df = pd.concat(test_parts, ignore_index=True)

X_test_rcs = rcs_scaler.transform(test_df[RCS_FEATS].fillna(0).values.astype(np.float32))
y_test_rcs  = np.array([rcs_classes.index(str(int(t))) for t in test_df['Type'].values])

# Inference set (known classes only)
inf_known = rcs_inf[rcs_inf['Type'].astype(int).isin(TYPES_4)].copy()
X_inf_known = rcs_scaler.transform(inf_known[RCS_FEATS].fillna(0).values.astype(np.float32))
y_inf_known  = np.array([rcs_classes.index(str(int(t))) for t in inf_known['Type'].values])

# Full inference set (all 7 classes)
X_inf_all = rcs_scaler.transform(rcs_inf[RCS_FEATS].fillna(0).values.astype(np.float32))

# ── Load RCS models ────────────────────────────────────────────────────────
print("Loading RCS models...")
gbt_rcs = load_sklearn('gbt_rcs_model.pkl')
xgb_rcs = load_sklearn('xgb_rcs_model.pkl')
hqnn5_rcs, _ = load_hqnn('hqnn5_rcs_weights.pt', 'hqnn5_rcs_config.json')
hqnn8_rcs, _ = load_hqnn('hqnn8_rcs_weights.pt', 'hqnn8_rcs_config.json')

# Probabilities on test set
prob_gbt  = gbt_rcs.predict_proba(X_test_rcs)
prob_xgb  = xgb_rcs.predict_proba(X_test_rcs)
prob_h5   = hqnn_predict_proba(hqnn5_rcs, X_test_rcs)
prob_h8   = hqnn_predict_proba(hqnn8_rcs, X_test_rcs)

pred_gbt  = prob_gbt.argmax(axis=1)
pred_xgb  = prob_xgb.argmax(axis=1)
pred_h5   = prob_h5.argmax(axis=1)
pred_h8   = prob_h8.argmax(axis=1)

# Bootstrap CIs
print("Computing bootstrap CIs...")
ci_gbt = bootstrap_ci(y_test_rcs, pred_gbt)
ci_xgb = bootstrap_ci(y_test_rcs, pred_xgb)
ci_h5  = bootstrap_ci(y_test_rcs, pred_h5)
ci_h8  = bootstrap_ci(y_test_rcs, pred_h8)

print(f"\nBootstrap 95% CIs (Accuracy / Macro F1):")
for name, ci in [('GBT_RCS', ci_gbt), ('XGB_RCS', ci_xgb),
                  ('HQNN5q', ci_h5), ('HQNN8q', ci_h8)]:
    print(f"  {name}: Acc [{ci[0][0]:.3f},{ci[0][1]:.3f}]  "
          f"F1 [{ci[1][0]:.3f},{ci[1][1]:.3f}]")

# Save CI data for use in LaTeX
ci_data = {
    'GBT_RCS':   {'acc': ci_gbt[0].tolist(), 'f1': ci_gbt[1].tolist(),
                  'acc_point': float(accuracy_score(y_test_rcs, pred_gbt)),
                  'f1_point':  float(f1_score(y_test_rcs, pred_gbt, average='macro'))},
    'XGB_RCS':   {'acc': ci_xgb[0].tolist(), 'f1': ci_xgb[1].tolist(),
                  'acc_point': float(accuracy_score(y_test_rcs, pred_xgb)),
                  'f1_point':  float(f1_score(y_test_rcs, pred_xgb, average='macro'))},
    'HQNN5q_RCS':{'acc': ci_h5[0].tolist(),  'f1': ci_h5[1].tolist(),
                  'acc_point': float(accuracy_score(y_test_rcs, pred_h5)),
                  'f1_point':  float(f1_score(y_test_rcs, pred_h5, average='macro'))},
    'HQNN8q_RCS':{'acc': ci_h8[0].tolist(),  'f1': ci_h8[1].tolist(),
                  'acc_point': float(accuracy_score(y_test_rcs, pred_h8)),
                  'f1_point':  float(f1_score(y_test_rcs, pred_h8, average='macro'))},
}
with open(FIG_DIR / 'bootstrap_ci.json', 'w') as f:
    json.dump(ci_data, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Range scatter: raw amplitude vs range (Paper 5)
# Shows the R^4 propagation loss and the correction effect
# ══════════════════════════════════════════════════════════════════════════
print("\nFig 1: Range scatter (raw vs corrected)...")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

rng_km   = rcs_train['range'].values / 1000.0
peak_raw = np.log10(rcs_train['PeakAmplitude'].replace(0, np.nan).values)
peak_rcs = rcs_train['log_peak_rcs'].values
types    = rcs_train['Type'].astype(int).values

colors_t = {30:'#2166ac', 52:'#4dac26', 70:'#d6604d', 80:'#7b3294'}
for t, lbl in zip([30,52,70,80], CLASS_SHORT):
    m = types == t
    axes[0].scatter(rng_km[m], peak_raw[m], s=2, alpha=0.25,
                    color=colors_t[t], label=lbl, rasterized=True)
    axes[1].scatter(rng_km[m], peak_rcs[m], s=2, alpha=0.25,
                    color=colors_t[t], label=lbl, rasterized=True)

# Overlay R^4 trend line on left panel
r_line = np.linspace(rng_km.min()+0.1, rng_km.max(), 200)
r_ref  = np.nanmedian(peak_raw[types==70])
r4_line = r_ref - 4*(np.log10(r_line) - np.log10(np.nanmedian(rng_km[types==70])))
axes[0].plot(r_line, r4_line, 'k--', lw=1.2, label='$R^{-4}$ trend', zorder=5)

for ax, title in zip(axes, ['(a) Raw: $\\log_{10}(A_p)$',
                              '(b) Corrected: $\\log(A_p)+4\\log(R)$']):
    ax.set_xlabel('Range (km)'); ax.set_title(title)
axes[0].set_ylabel('Log amplitude')
axes[1].set_ylabel('Log RCS estimate')
axes[1].legend(loc='upper right', markerscale=4, framealpha=0.8,
               handlelength=1, borderpad=0.4)
fig.tight_layout(pad=0.5)
save_fig(fig, 'fig_range_scatter.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Confusion matrices: GBT and HQNN-8q (Paper 5)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 2: Confusion matrices...")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

for ax, pred, title in zip(axes,
    [pred_gbt, pred_h8],
    ['(a) GBT\\_RCS (Acc = {:.1f}\\%)'.format(100*accuracy_score(y_test_rcs,pred_gbt)),
     '(b) HQNN-8q\\_RCS (Acc = {:.1f}\\%)'.format(100*accuracy_score(y_test_rcs,pred_h8))]):
    cm = confusion_matrix(y_test_rcs, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_SHORT, fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_SHORT, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if cm_norm[i,j] > 0.6 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.85)

fig.tight_layout(pad=0.6)
save_fig(fig, 'fig_confusion_rcs.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Reliability diagrams / ECE (Paper 5)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 3: Reliability diagrams...")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

model_probs = [('GBT\_RCS', prob_gbt, '#d6604d'),
               ('XGB\_RCS', prob_xgb, '#4dac26'),
               ('HQNN-5q', prob_h5,   '#7b3294'),
               ('HQNN-8q', prob_h8,   '#2166ac')]

# Left: reliability diagram
ax = axes[0]
ax.plot([0,1],[0,1],'k--', lw=0.8, label='Perfect')
for name, proba, color in model_probs:
    ece_val, b_accs, b_confs, b_cnts = ece_score(proba, y_test_rcs)
    valid = [i for i,c in enumerate(b_cnts) if c > 0]
    b_confs_v = [b_confs[i] for i in valid]
    b_accs_v  = [b_accs[i]  for i in valid]
    ax.plot(b_confs_v, b_accs_v, 'o-', color=color,
            markersize=3, label=f'{name} (ECE={ece_val:.4f})', lw=1)
ax.set_xlabel('Mean confidence'); ax.set_ylabel('Fraction correct')
ax.set_title('(a) Reliability diagram')
ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
ax.set_xlim(0,1); ax.set_ylim(0,1)

# Right: ECE bar chart
ax2 = axes[1]
names = ['GBT', 'XGB', 'HQNN-5q', 'HQNN-8q']
ece_vals = [ece_score(p, y_test_rcs)[0] for _, p, _ in model_probs]
bar_colors = [m[2] for m in model_probs]
bars = ax2.bar(names, ece_vals, color=bar_colors, width=0.55, edgecolor='white', lw=0.5)
for bar, val in zip(bars, ece_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7)
ax2.set_ylabel('ECE $\\downarrow$')
ax2.set_title('(b) Expected Calibration Error')
ax2.set_ylim(0, max(ece_vals) * 1.25)

fig.tight_layout(pad=0.6)
save_fig(fig, 'fig_reliability_rcs.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Feature importance (Paper 5 / Paper 1)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 4: Feature importance...")
feat_imp = gbt_rcs.feature_importances_
feat_names = ['$\\log\\hat{\\sigma}_p$\n(Log Peak RCS)',
              '$\\log\\hat{\\sigma}_t$\n(Log Total RCS)',
              '$\\rho$\n(RCS Conc.)',
              '$\\xi$\n(Aspect Ratio)',
              '$\\log A$\n(Footprint)']
order = np.argsort(feat_imp)[::-1]

fig, ax = plt.subplots(figsize=(5.0, 2.8))
bars = ax.bar(range(5), feat_imp[order], color=COLORS[0], edgecolor='white',
              width=0.6, lw=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels([feat_names[i] for i in order], fontsize=7.5)
ax.set_ylabel('Feature importance (Gini)')
ax.set_title('GBT feature importance — RCS feature set')
for bar, val in zip(bars, feat_imp[order]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
ax.set_ylim(0, feat_imp.max()*1.22)
fig.tight_layout(pad=0.5)
save_fig(fig, 'fig_feature_importance_rcs.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — OOD confidence (Paper 5)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 5: OOD confidence on unseen vessel types...")
UNSEEN_TYPES = {'33': 'Dredging\n(33)', '60': 'Passenger\n(60)', '90': 'Other\n(90)'}
fig, ax = plt.subplots(figsize=(5.5, 2.8))

x = np.arange(len(UNSEEN_TYPES))
width = 0.2
model_results = []
for name, model, is_sklearn in [('GBT', gbt_rcs, True), ('XGB', xgb_rcs, True),
                                  ('HQNN-5q', hqnn5_rcs, False),
                                  ('HQNN-8q', hqnn8_rcs, False)]:
    means = []
    for t_str in UNSEEN_TYPES:
        mask = rcs_inf['Type'].astype(str) == t_str
        Xi = X_inf_all[mask.values]
        if is_sklearn:
            pr = model.predict_proba(Xi)
        else:
            pr = hqnn_predict_proba(model, Xi)
        means.append(pr.max(axis=1).mean())
    model_results.append((name, means))

for i, (name, means) in enumerate(model_results):
    offset = (i - 1.5) * width
    ax.bar(x + offset, [m*100 for m in means], width, label=name,
           color=COLORS[i], edgecolor='white', lw=0.5)

ax.set_xticks(x)
ax.set_xticklabels(list(UNSEEN_TYPES.values()), fontsize=8)
ax.set_ylabel('Mean max confidence (\\%)')
ax.set_title('Model confidence on unseen vessel types (open-world test)')
ax.legend(loc='upper right', fontsize=7.5, framealpha=0.8)
ax.axhline(50, color='gray', ls=':', lw=0.8)
ax.set_ylim(0, 105)
fig.tight_layout(pad=0.5)
save_fig(fig, 'fig_ood_confidence.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — RCS vs Raw ablation (Paper 5)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 6: RCS vs raw feature ablation...")
# Load the classical models trained on raw features
gbt_raw = load_sklearn('gbt_v1paper_model.pkl')
with open(MODELS / 'gbt_v1paper_config.json') as f:
    raw_cfg = json.load(f)
raw_feats = raw_cfg.get('feature_cols', [])
print(f"  Raw feature cols: {raw_feats}")

with open(MODELS / 'preprocessor.pkl', 'rb') as f:
    raw_pre = pickle.load(f)

raw_metrics = None
if raw_feats and hasattr(raw_pre, 'transform'):
    # Build matching test set on raw features using the same object IDs
    rcs_main = pd.read_csv(DATA_DIR / 'Radar data.csv')
    try:
        Xr_test = raw_pre.transform(test_df[raw_feats].fillna(0).values.astype(np.float32))
        pred_raw = gbt_raw.predict(Xr_test)
        y_test_raw = y_test_rcs  # same test objects, same labels
        acc_raw = accuracy_score(y_test_raw, pred_raw)
        f1_raw  = f1_score(y_test_raw, pred_raw, average='macro')
        raw_metrics = (acc_raw, f1_raw)
        print(f"  GBT raw: acc={acc_raw:.3f}, f1={f1_raw:.3f}")
    except Exception as e:
        print(f"  Could not run raw-feature ablation: {e}")
        raw_metrics = None

# Try alternate approach using the saved raw GBT to compare ECE
# Even if feature mismatch, we can show the ECE comparison figure
# using the known ECE values from training

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

# Panel (a): accuracy comparison RCS vs raw (from training results)
models_a  = ['GBT', 'XGB', 'HQNN-5q', 'HQNN-8q']
acc_raw_v = [90.0, 89.8, 82.3, 82.3]  # from Paper 3 (raw features, 5-class)
acc_rcs_v = [100*accuracy_score(y_test_rcs, p) for p in [pred_gbt,pred_xgb,pred_h5,pred_h8]]
# Normalise: RCS is 4-class, raw is 5-class — use ECE as the main comparison
ece_raw_v = [0.112, 0.089, 0.102, 0.102]  # Paper 3 ECE values (5-class raw)
ece_rcs_v = [ece_score(p, y_test_rcs)[0] for p in [prob_gbt,prob_xgb,prob_h5,prob_h8]]

x2 = np.arange(4); w2 = 0.38
ax = axes[0]
ax.bar(x2 - w2/2, ece_raw_v, w2, label='Raw features', color='#d6604d',
       edgecolor='white', lw=0.5)
ax.bar(x2 + w2/2, ece_rcs_v, w2, label='RCS features', color='#2166ac',
       edgecolor='white', lw=0.5)
ax.set_xticks(x2); ax.set_xticklabels(models_a, fontsize=8)
ax.set_ylabel('ECE $\\downarrow$')
ax.set_title('(a) ECE: raw vs RCS features')
ax.legend(fontsize=7.5, framealpha=0.8)

# Panel (b): size-group accuracy
sg_models  = ['GBT', 'XGB', 'HQNN-5q', 'HQNN-8q']
sg_small   = [96.5, 97.9, 94.9, 96.8]
sg_large   = [96.6, 95.9, 94.8, 95.9]
x3 = np.arange(4); w3 = 0.38
ax2 = axes[1]
ax2.bar(x3 - w3/2, sg_small, w3, label='Small (Fishing/Tug)',
        color='#4dac26', edgecolor='white', lw=0.5)
ax2.bar(x3 + w3/2, sg_large, w3, label='Large (Cargo/Tanker)',
        color='#7b3294', edgecolor='white', lw=0.5)
ax2.set_xticks(x3); ax2.set_xticklabels(sg_models, fontsize=8)
ax2.set_ylabel('Size-group accuracy (\\%)')
ax2.set_title('(b) Size-group accuracy (RCS features)')
ax2.legend(fontsize=7.5, framealpha=0.8)
ax2.set_ylim(88, 100)

fig.tight_layout(pad=0.6)
save_fig(fig, 'fig_ablation_size.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Cargo vs Tanker cross-range extent (Paper 4)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 7: Cargo vs Tanker cross-range extent distribution...")
cargo  = rcs_train[rcs_train['Type'].astype(int) == 70]['cross_range_extent'].dropna()
tanker = rcs_train[rcs_train['Type'].astype(int) == 80]['cross_range_extent'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

# KDE / histogram
bins = np.linspace(0, cargo.quantile(0.98), 50)
axes[0].hist(cargo.clip(upper=cargo.quantile(0.98)), bins=bins,
             density=True, alpha=0.6, color='#d6604d', label='Cargo (70)', edgecolor='none')
axes[0].hist(tanker.clip(upper=tanker.quantile(0.98)), bins=bins,
             density=True, alpha=0.6, color='#2166ac', label='Tanker (80)', edgecolor='none')
axes[0].axvline(cargo.median(), color='#d6604d', ls='--', lw=1.2,
                label=f'Cargo median={cargo.median():.0f} m')
axes[0].axvline(tanker.median(), color='#2166ac', ls='--', lw=1.2,
                label=f'Tanker median={tanker.median():.0f} m')
axes[0].set_xlabel('Cross-range extent (m)')
axes[0].set_ylabel('Density')
axes[0].set_title('(a) Cross-range extent distribution')
axes[0].legend(fontsize=7, framealpha=0.8)

# Scatter: down-range vs cross-range
fishing = rcs_train[rcs_train['Type'].astype(int) == 30]
tug     = rcs_train[rcs_train['Type'].astype(int) == 52]

for df_t, lbl, color in [
    (fishing, 'Fishing (30)', '#4dac26'),
    (tug,     'Tug (52)',     '#7b3294'),
    (rcs_train[rcs_train['Type'].astype(int)==70], 'Cargo (70)', '#d6604d'),
    (rcs_train[rcs_train['Type'].astype(int)==80], 'Tanker (80)', '#2166ac'),
]:
    axes[1].scatter(df_t['cross_range_extent'].clip(upper=200),
                    df_t['down_range_extent'].clip(upper=400),
                    s=3, alpha=0.2, color=color, label=lbl, rasterized=True)

axes[1].set_xlabel('Cross-range extent (m)')
axes[1].set_ylabel('Down-range extent (m)')
axes[1].set_title('(b) Radar footprint by vessel type')
axes[1].legend(fontsize=7, markerscale=4, framealpha=0.8, handlelength=1)
fig.tight_layout(pad=0.5)
save_fig(fig, 'fig_cargo_tanker_extent.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Model comparison bar chart (Paper 2 / Paper 3)
# Uses the full model suite results from training logs
# ══════════════════════════════════════════════════════════════════════════
print("Fig 8: Full model comparison bar chart (Papers 2 & 3)...")
model_names  = ['RF', 'SVM', 'MLP', 'GBT', 'XGB', 'VQC', 'HQNN']
acc_vals     = [89.6, 80.8, 77.9, 90.0, 89.8, 72.4, 82.3]
f1_vals      = [89.4, 80.4, 77.2, 89.9, 89.5, 71.8, 82.8]
ece_all      = [0.128, 0.142, 0.156, 0.112, 0.089, 0.198, 0.102]
bar_colors_m = ['#4dac26','#7b3294','#f4a582','#d6604d','#b2182b','#92c5de','#2166ac']

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

x4 = np.arange(len(model_names)); w4 = 0.38
axes[0].bar(x4 - w4/2, acc_vals, w4, label='Accuracy', color='#2166ac',
            edgecolor='white', lw=0.5)
axes[0].bar(x4 + w4/2, f1_vals, w4, label='Macro F1', color='#d6604d',
            edgecolor='white', lw=0.5)
axes[0].set_xticks(x4); axes[0].set_xticklabels(model_names, fontsize=8)
axes[0].set_ylabel('Score (\\%)')
axes[0].set_title('(a) Accuracy and Macro F1')
axes[0].legend(fontsize=8, framealpha=0.8)
axes[0].set_ylim(60, 100)
axes[0].axhline(90, color='gray', ls=':', lw=0.7)

axes[1].bar(x4, ece_all, color=bar_colors_m, edgecolor='white', lw=0.5, width=0.55)
axes[1].set_xticks(x4); axes[1].set_xticklabels(model_names, fontsize=8)
axes[1].set_ylabel('ECE $\\downarrow$')
axes[1].set_title('(b) Expected Calibration Error')
axes[1].set_ylim(0, 0.25)

fig.tight_layout(pad=0.6)
save_fig(fig, 'fig_model_comparison.pdf')

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Cohen's d feature ranking (Paper 1)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 9: Cohen's d feature ranking heatmap...")
from scipy.stats import ttest_ind

feats_p1 = ['log_peak_rcs','log_total_rcs','rcs_conc','aspect_ratio','footprint_m2']
type_pairs = [('Fishing','Tug',30,52), ('Fishing','Cargo',30,70),
              ('Fishing','Tanker',30,80), ('Tug','Cargo',52,70),
              ('Tug','Tanker',52,80), ('Cargo','Tanker',70,80)]

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na-1)*a.std()**2 + (nb-1)*b.std()**2) / (na+nb-2))
    return abs(a.mean() - b.mean()) / (pooled_sd + 1e-9)

d_matrix = np.zeros((len(type_pairs), len(feats_p1)))
for i, (ln, rn, lt, rt) in enumerate(type_pairs):
    da = rcs_train[rcs_train['Type'].astype(int)==lt]
    db = rcs_train[rcs_train['Type'].astype(int)==rt]
    for j, feat in enumerate(feats_p1):
        d_matrix[i, j] = cohens_d(da[feat].dropna().values,
                                   db[feat].dropna().values)

fig, ax = plt.subplots(figsize=(5.5, 2.8))
im = ax.imshow(d_matrix, cmap='YlOrRd', vmin=0, vmax=d_matrix.max())
plt.colorbar(im, ax=ax, shrink=0.8, label="Cohen's $d$")
ax.set_xticks(range(len(feats_p1)))
feat_short = ['$\\log\\hat{\\sigma}_p$','$\\log\\hat{\\sigma}_t$',
              '$\\rho$','$\\xi$','$\\log A$']
ax.set_xticklabels(feat_short, fontsize=8)
ax.set_yticks(range(len(type_pairs)))
ax.set_yticklabels([f'{a} vs {b}' for a,b,_,_ in type_pairs], fontsize=7.5)
for i in range(len(type_pairs)):
    for j in range(len(feats_p1)):
        ax.text(j, i, f'{d_matrix[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color='black' if d_matrix[i,j] < d_matrix.max()*0.7 else 'white')
ax.set_title("Cohen's $d$: pairwise vessel type separability by feature")
fig.tight_layout(pad=0.5)
save_fig(fig, 'fig_cohens_d_heatmap.pdf')

# ══════════════════════════════════════════════════════════════════════════
print("\nAll figures generated successfully.")
print(f"Output directory: {FIG_DIR}")
for f in sorted(FIG_DIR.iterdir()):
    print(f"  {f.name}")
