import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_processing import DataProcessor
from src.models.hqnn_model import HQNNClassifier
from src.trainer import TorchTrainer, evaluate_model

CSV_PATH     = '/home/iaxiom/Downloads/Radar_data_filtered.csv'
LABEL_COL    = 'Label'
KEEP_CLASSES = [30, 33, 52, 70, 80]

# ── Data prep (same seed/split as previous runs) ──────────────
df = pd.read_csv(CSV_PATH)
df = df[df[LABEL_COL].isin(KEEP_CLASSES)].reset_index(drop=True)
feature_cols = [c for c in df.columns if c != LABEL_COL]

dp = DataProcessor()
df_clean = dp.clean(df, feature_cols, LABEL_COL, {
    'remove_duplicates': True,
    'missing_strategy': 'drop',
    'outlier_method': 'none',
})
data = dp.preprocess(df_clean, feature_cols, LABEL_COL, {
    'normalization': 'standard',
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'use_pca': False,
    'balancing': 'smote',
})

counts  = np.bincount(data['y_train'])
weights = data['n_classes'] / (counts + 1e-9)
weights = (weights / weights.sum() * data['n_classes']).tolist()

print(f"Train: {data['train_size']} (post-SMOTE) | Test: {data['test_size']}")
print(f"Classes: {data['class_names']}\n")

# ── HQNN — 100 epochs ────────────────────────────────────────
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
    'epochs': 100,
    'weight_decay': 0.001,
    'class_weights': weights,
}

print("=" * 60)
print("  HQNN — 100 epochs, wd=0.001, hidden=32, 2 classical layers")
print("=" * 60)
trainer = TorchTrainer(model, train_cfg)

def cb(info):
    ep = info['epoch']
    if ep % 10 == 0 or ep == 1:
        print(f"  Epoch {ep:3d}/100 | loss={info['train_loss']:.4f} "
              f"acc={info['train_acc']:.1%} | val_acc={info['val_acc']:.1%}")

trainer.train(data['X_train'], data['y_train'],
              data['X_test'],  data['y_test'], callback=cb)

# ── Evaluate ──────────────────────────────────────────────────
metrics = evaluate_model(model, data['X_test'], data['y_test'],
                         data['class_names'], is_torch=True)

print(f"\n  Accuracy    : {metrics['accuracy']:.1%}")
print(f"  F1 Macro    : {metrics['f1_macro']:.1%}")
print(f"  F1 Weighted : {metrics['f1_weighted']:.1%}")
auc = metrics.get('auc_macro')
print(f"  AUC Macro   : {auc:.3f}" if auc else "  AUC Macro   : N/A")

print("\n  Per-class report:")
print(classification_report(data['y_test'], metrics['predictions'],
                            target_names=data['class_names'], zero_division=0))

print("  Confusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(data['y_test'], metrics['predictions'])
cm_df = pd.DataFrame(cm,
    index=[f"true_{c}" for c in data['class_names']],
    columns=[f"pred_{c}" for c in data['class_names']])
print(cm_df.to_string())

# ── Save predictions ──────────────────────────────────────────
prob_cols = {f"prob_{c}": metrics['probabilities'][:, i]
             for i, c in enumerate(data['class_names'])}
out = pd.DataFrame({
    'true_label':      [data['class_names'][y] for y in data['y_test']],
    'predicted_label': [data['class_names'][p] for p in metrics['predictions']],
    'correct':         data['y_test'] == metrics['predictions'],
    **prob_cols,
})
out_path = '/home/iaxiom/Downloads/predictions_hqnn_filtered_v3.csv'
out.to_csv(out_path, index=False)
print(f"\n  Predictions saved → {out_path}")
