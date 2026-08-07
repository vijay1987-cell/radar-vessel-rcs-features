import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

VQC_CSV  = '/home/iaxiom/Downloads/predictions_vqc_filtered_v2.csv'
HQNN_CSV = '/home/iaxiom/Downloads/predictions_hqnn_filtered_v2.csv'
CLASSES  = ['30', '33', '52', '70', '80']
PROB_COLS = [f'prob_{c}' for c in CLASSES]

vqc  = pd.read_csv(VQC_CSV)
hqnn = pd.read_csv(HQNN_CSV)

assert (vqc['true_label'].values == hqnn['true_label'].values).all(), \
    "Row mismatch between prediction files — test sets differ!"

y_true = vqc['true_label'].astype(str).values

# ── Soft voting: average probabilities ───────────────────────
vqc_proba  = vqc[PROB_COLS].values
hqnn_proba = hqnn[PROB_COLS].values
avg_proba  = (vqc_proba + hqnn_proba) / 2.0
y_pred     = np.array([CLASSES[i] for i in avg_proba.argmax(axis=1)])

# ── Metrics ──────────────────────────────────────────────────
acc      = accuracy_score(y_true, y_pred)
f1_mac   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
f1_wt    = f1_score(y_true, y_pred, average='weighted', zero_division=0)

class_idx = {c: i for i, c in enumerate(CLASSES)}
y_idx     = np.array([class_idx[c] for c in y_true])
y_bin     = label_binarize(y_idx, classes=list(range(len(CLASSES))))
try:
    auc = roc_auc_score(y_bin, avg_proba, average='macro', multi_class='ovr')
except Exception:
    auc = None

print("=" * 60)
print("  SOFT-VOTE ENSEMBLE  —  VQC_v2 + HQNN_v2  (filtered features)")
print("=" * 60)
print(f"  Accuracy    : {acc:.1%}")
print(f"  F1 Macro    : {f1_mac:.1%}")
print(f"  F1 Weighted : {f1_wt:.1%}")
print(f"  AUC Macro   : {auc:.3f}" if auc else "  AUC Macro   : N/A")

print("\n  Per-class report:")
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

print("  Confusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
cm_df = pd.DataFrame(cm,
    index=[f"true_{c}" for c in CLASSES],
    columns=[f"pred_{c}" for c in CLASSES])
print(cm_df.to_string())

# ── vs individual models ──────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPARISON")
print("=" * 60)
rows = []
for name, pred_col, proba in [
    ("VQC  (filtered v2)",  vqc['predicted_label'].astype(str),  vqc_proba),
    ("HQNN (filtered v2)", hqnn['predicted_label'].astype(str), hqnn_proba),
    ("Ensemble (soft)",    y_pred,                               avg_proba),
]:
    a  = accuracy_score(y_true, pred_col)
    fm = f1_score(y_true, pred_col, average='macro',    zero_division=0)
    fw = f1_score(y_true, pred_col, average='weighted', zero_division=0)
    try:
        au = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
    except Exception:
        au = None
    # per-class recall for 70
    mask70 = y_true == '70'
    r70 = (pred_col[mask70] == '70').mean() if mask70.sum() > 0 else 0
    rows.append({'Model': name, 'Accuracy': f'{a:.1%}', 'F1 Macro': f'{fm:.1%}',
                 'F1 Weighted': f'{fw:.1%}',
                 'AUC': f'{au:.3f}' if au else 'N/A',
                 'Class-70 Recall': f'{r70:.1%}'})

print(pd.DataFrame(rows).set_index('Model').to_string())

# ── Save predictions ──────────────────────────────────────────
out = pd.DataFrame({
    'true_label':      y_true,
    'predicted_label': y_pred,
    'correct':         y_true == y_pred,
    **{f'prob_{c}': avg_proba[:, i] for i, c in enumerate(CLASSES)},
})
out_path = '/home/iaxiom/Downloads/predictions_ensemble_filtered.csv'
out.to_csv(out_path, index=False)
print(f"\n  Predictions saved → {out_path}")
