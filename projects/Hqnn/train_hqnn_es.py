import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.data_processing import DataProcessor
from src.models.hqnn_model import HQNNClassifier
from src.trainer import TorchTrainer, evaluate_model

CSV_PATH     = '/home/iaxiom/Downloads/Radar_data_filtered.csv'
LABEL_COL    = 'Label'
KEEP_CLASSES = [30, 33, 52, 70, 80]
CLASSES      = ['30', '33', '52', '70', '80']
PROB_COLS    = [f'prob_{c}' for c in CLASSES]

# ── Data (same seed/split as all previous runs) ───────────────
df = pd.read_csv(CSV_PATH)
df = df[df[LABEL_COL].isin(KEEP_CLASSES)].reset_index(drop=True)
feature_cols = [c for c in df.columns if c != LABEL_COL]

dp = DataProcessor()
df_clean = dp.clean(df, feature_cols, LABEL_COL, {
    'remove_duplicates': True, 'missing_strategy': 'drop', 'outlier_method': 'none',
})
data = dp.preprocess(df_clean, feature_cols, LABEL_COL, {
    'normalization': 'standard', 'test_size': 0.2, 'random_state': 42,
    'stratify': True, 'use_pca': False, 'balancing': 'smote',
})

counts  = np.bincount(data['y_train'])
weights = data['n_classes'] / (counts + 1e-9)
weights = (weights / weights.sum() * data['n_classes']).tolist()

print(f"Train: {data['train_size']} (post-SMOTE) | Test: {data['test_size']}")
print(f"Classes: {data['class_names']}\n")

# ── HQNN with early stopping (patience=15, max 150 epochs) ───
model = HQNNClassifier(
    n_features=data['n_features'],
    n_qubits=data['n_features'],
    n_layers=2,
    n_classes=data['n_classes'],
    classical_hidden=32,
    n_classical_layers=2,
    ansatz='strongly_entangling',
    activation='relu',
)

train_cfg = {
    'optimizer': 'adam',
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 150,
    'weight_decay': 0.001,
    'class_weights': weights,
    'patience': 15,
}

print("=" * 60)
print("  HQNN — max 150 epochs, early stopping (patience=15)")
print("=" * 60)
trainer = TorchTrainer(model, train_cfg)

def cb(info):
    ep = info['epoch']
    if ep % 10 == 0 or ep == 1:
        print(f"  Epoch {ep:3d} | loss={info['train_loss']:.4f} "
              f"acc={info['train_acc']:.1%} | val_acc={info['val_acc']:.1%} "
              f"[best={info['best_val_acc']:.1%} @ ep{info['best_epoch']}]")

trainer.train(data['X_train'], data['y_train'],
              data['X_test'],  data['y_test'], callback=cb)

print(f"\n  Restored best weights from epoch {trainer.best_epoch}")

# ── Evaluate HQNN ─────────────────────────────────────────────
metrics = evaluate_model(model, data['X_test'], data['y_test'],
                         data['class_names'], is_torch=True)

print(f"\n  Accuracy    : {metrics['accuracy']:.1%}")
print(f"  F1 Macro    : {metrics['f1_macro']:.1%}")
print(f"  F1 Weighted : {metrics['f1_weighted']:.1%}")
auc = metrics.get('auc_macro')
print(f"  AUC Macro   : {auc:.3f}" if auc else "  AUC Macro   : N/A")
print()
print(classification_report(data['y_test'], metrics['predictions'],
                            target_names=data['class_names'], zero_division=0))

cm = confusion_matrix(data['y_test'], metrics['predictions'])
print(pd.DataFrame(cm,
    index=[f"true_{c}" for c in data['class_names']],
    columns=[f"pred_{c}" for c in data['class_names']]).to_string())

# ── Save HQNN predictions ─────────────────────────────────────
hqnn_out = pd.DataFrame({
    'true_label':      [data['class_names'][y] for y in data['y_test']],
    'predicted_label': [data['class_names'][p] for p in metrics['predictions']],
    'correct':         data['y_test'] == metrics['predictions'],
    **{f'prob_{c}': metrics['probabilities'][:, i] for i, c in enumerate(data['class_names'])},
})
hqnn_path = '/home/iaxiom/Downloads/predictions_hqnn_filtered_es.csv'
hqnn_out.to_csv(hqnn_path, index=False)
print(f"\n  HQNN predictions saved → {hqnn_path}")

# ── Soft-vote ensemble: VQC_v2 + HQNN_es ─────────────────────
print("\n" + "=" * 60)
print("  SOFT-VOTE ENSEMBLE — VQC_v2 + HQNN_es")
print("=" * 60)

vqc = pd.read_csv('/home/iaxiom/Downloads/predictions_vqc_filtered_v2.csv')
y_true     = vqc['true_label'].astype(str).values
avg_proba  = (vqc[PROB_COLS].values + metrics['probabilities']) / 2.0
y_pred_ens = np.array([CLASSES[i] for i in avg_proba.argmax(axis=1)])

y_idx = np.array([CLASSES.index(c) for c in y_true])
y_bin = label_binarize(y_idx, classes=list(range(len(CLASSES))))
auc_ens = roc_auc_score(y_bin, avg_proba, average='macro', multi_class='ovr')

print(f"  Accuracy    : {accuracy_score(y_true, y_pred_ens):.1%}")
print(f"  F1 Macro    : {f1_score(y_true, y_pred_ens, average='macro', zero_division=0):.1%}")
print(f"  F1 Weighted : {f1_score(y_true, y_pred_ens, average='weighted', zero_division=0):.1%}")
print(f"  AUC Macro   : {auc_ens:.3f}")
print()
print(classification_report(y_true, y_pred_ens, target_names=CLASSES, zero_division=0))

# ── Final comparison ──────────────────────────────────────────
print("=" * 60)
print("  COMPARISON")
print("=" * 60)
prev_ens = pd.read_csv('/home/iaxiom/Downloads/predictions_ensemble_filtered.csv')
runs = [
    ('VQC_v2',                    vqc['predicted_label'].astype(str),           vqc[PROB_COLS].values),
    ('HQNN_es (early stopping)',  hqnn_out['predicted_label'].astype(str),      metrics['probabilities']),
    ('Ensemble v1 (VQC+HQNN_v2)', prev_ens['predicted_label'].astype(str),      prev_ens[PROB_COLS].values),
    ('Ensemble v3 (VQC+HQNN_es)', y_pred_ens,                                  avg_proba),
]
rows = []
for name, pred, proba in runs:
    a  = accuracy_score(y_true, pred)
    fm = f1_score(y_true, pred, average='macro',    zero_division=0)
    fw = f1_score(y_true, pred, average='weighted', zero_division=0)
    au = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
    r70 = (pred[y_true == '70'] == '70').mean()
    rows.append({'Model': name, 'Acc': f'{a:.1%}', 'F1 Mac': f'{fm:.1%}',
                 'F1 Wt': f'{fw:.1%}', 'AUC': f'{au:.3f}', 'Cls70': f'{r70:.1%}'})
print(pd.DataFrame(rows).set_index('Model').to_string())

# save ensemble predictions
ens_out = pd.DataFrame({
    'true_label': y_true, 'predicted_label': y_pred_ens,
    'correct': y_true == y_pred_ens,
    **{f'prob_{c}': avg_proba[:, i] for i, c in enumerate(CLASSES)},
})
ens_out.to_csv('/home/iaxiom/Downloads/predictions_ensemble_v3.csv', index=False)
print("\n  Ensemble predictions saved → /home/iaxiom/Downloads/predictions_ensemble_v3.csv")
