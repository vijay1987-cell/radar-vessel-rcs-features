import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_processing import DataProcessor
from src.trainer import TorchTrainer, evaluate_model

CSV_PATH     = '/home/iaxiom/Downloads/Radar_data_filtered.csv'
LABEL_COL    = 'Label'
KEEP_CLASSES = [30, 33, 52, 70, 80]

# ── shared data prep ─────────────────────────────────────────
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

print(f"Dataset: {df.shape[0]} rows | Train: {data['train_size']} (post-SMOTE) | Test: {data['test_size']}")
print(f"Classes: {data['class_names']}\n")


def run_model(name, model, epochs, lr, wd):
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)
    train_cfg = {
        'optimizer': 'adam',
        'learning_rate': lr,
        'batch_size': 32,
        'epochs': epochs,
        'weight_decay': wd,
        'class_weights': weights,
    }
    trainer = TorchTrainer(model, train_cfg)

    def cb(info):
        ep = info['epoch']
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs} | loss={info['train_loss']:.4f} "
                  f"acc={info['train_acc']:.1%} | val_acc={info['val_acc']:.1%}")

    trainer.train(data['X_train'], data['y_train'],
                  data['X_test'],  data['y_test'], callback=cb)

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

    # Save predictions CSV
    prob_cols = {f"prob_{c}": metrics['probabilities'][:, i]
                 for i, c in enumerate(data['class_names'])}
    out = pd.DataFrame({
        'true_label':      [data['class_names'][y] for y in data['y_test']],
        'predicted_label': [data['class_names'][p] for p in metrics['predictions']],
        'correct':         data['y_test'] == metrics['predictions'],
        **prob_cols,
    })
    tag = name.split()[0].lower()
    path = f'/home/iaxiom/Downloads/predictions_{tag}_filtered_v2.csv'
    out.to_csv(path, index=False)
    print(f"\n  Predictions saved → {path}\n")
    return metrics


# ── MODEL 1: VQC — 100 epochs, weight_decay=0.001 ───────────
from src.models.vqc_model import VQCClassifier
vqc = VQCClassifier(
    n_features=data['n_features'],
    n_qubits=data['n_features'],
    n_layers=3,
    n_classes=data['n_classes'],
    embedding='angle',
    ansatz='strongly_entangling',
)
vqc_metrics = run_model("VQC  —  100 epochs, wd=0.001, 5 qubits, 3 layers",
                        vqc, epochs=100, lr=0.01, wd=0.001)


# ── MODEL 2: HQNN — 50 epochs, weight_decay=0.001 ───────────
from src.models.hqnn_model import HQNNClassifier
hqnn = HQNNClassifier(
    n_features=data['n_features'],
    n_qubits=data['n_features'],
    n_layers=2,
    n_classes=data['n_classes'],
    classical_hidden=32,
    n_classical_layers=2,
    ansatz='strongly_entangling',
    activation='relu',
)
hqnn_metrics = run_model("HQNN  —  50 epochs, wd=0.001, hidden=32, 2 classical layers",
                         hqnn, epochs=50, lr=0.01, wd=0.001)


# ── SUMMARY ──────────────────────────────────────────────────
print("=" * 60)
print("  SUMMARY COMPARISON")
print("=" * 60)
rows = []
for name, m in [("VQC (filtered v2)", vqc_metrics), ("HQNN (filtered v2)", hqnn_metrics)]:
    rows.append({
        'Model': name,
        'Accuracy': f"{m['accuracy']:.1%}",
        'F1 Macro': f"{m['f1_macro']:.1%}",
        'F1 Weighted': f"{m['f1_weighted']:.1%}",
        'AUC': f"{m['auc_macro']:.3f}" if m.get('auc_macro') else 'N/A',
    })
print(pd.DataFrame(rows).set_index('Model').to_string())
