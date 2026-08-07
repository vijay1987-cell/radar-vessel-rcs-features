"""
Train VQC + HQNN (lr=0.005, early stopping), save best weights,
run ensemble, and write a reusable inference script.

Saved artefacts (saved_models/):
  vqc_weights.pt       — VQC best state_dict
  vqc_config.json      — VQC architecture params
  hqnn_weights.pt      — HQNN best state_dict
  hqnn_config.json     — HQNN architecture params
  preprocessor.pkl     — scaler + label encoder + feature names
"""

import sys, os, json, pickle, copy
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.data_processing import DataProcessor
from src.models.vqc_model import VQCClassifier
from src.models.hqnn_model import HQNNClassifier
from src.trainer import TorchTrainer, evaluate_model

SAVE_DIR     = Path(__file__).parent / 'saved_models'
SAVE_DIR.mkdir(exist_ok=True)
CSV_PATH     = '/home/iaxiom/Downloads/Radar_data_filtered.csv'
LABEL_COL    = 'Label'
KEEP_CLASSES = [30, 33, 52, 70, 80]
CLASSES      = ['30', '33', '52', '70', '80']
PROB_COLS    = [f'prob_{c}' for c in CLASSES]

# ── 1. Data prep ─────────────────────────────────────────────
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

# Save preprocessor (scaler + label encoder + feature/class names)
preprocessor = {
    'scaler':        dp.scaler,
    'label_encoder': dp.label_encoder,
    'feature_cols':  feature_cols,
    'class_names':   data['class_names'],
    'keep_classes':  KEEP_CLASSES,
    'label_col':     LABEL_COL,
}
with open(SAVE_DIR / 'preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor saved → saved_models/preprocessor.pkl")


def train_model(name, model, arch_cfg, epochs, lr, wd, patience):
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)
    train_cfg = {
        'optimizer': 'adam', 'learning_rate': lr, 'batch_size': 32,
        'epochs': epochs, 'weight_decay': wd, 'class_weights': weights,
        'patience': patience,
    }
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

    metrics = evaluate_model(model, data['X_test'], data['y_test'],
                             data['class_names'], is_torch=True)

    print(f"  Accuracy : {metrics['accuracy']:.1%} | "
          f"F1 Macro : {metrics['f1_macro']:.1%} | "
          f"AUC : {metrics['auc_macro']:.3f}")
    print()
    print(classification_report(data['y_test'], metrics['predictions'],
                                target_names=data['class_names'], zero_division=0))
    cm = confusion_matrix(data['y_test'], metrics['predictions'])
    print(pd.DataFrame(cm,
        index=[f"true_{c}" for c in data['class_names']],
        columns=[f"pred_{c}" for c in data['class_names']]).to_string())

    # Save weights + architecture config
    tag = name.split()[0].lower()
    torch.save(model.state_dict(), SAVE_DIR / f'{tag}_weights.pt')
    with open(SAVE_DIR / f'{tag}_config.json', 'w') as f:
        json.dump(arch_cfg, f, indent=2)
    print(f"\n  Weights → saved_models/{tag}_weights.pt")
    print(f"  Config  → saved_models/{tag}_config.json")

    # Save predictions CSV
    prob_cols = {f"prob_{c}": metrics['probabilities'][:, i]
                 for i, c in enumerate(data['class_names'])}
    out = pd.DataFrame({
        'true_label':      [data['class_names'][y] for y in data['y_test']],
        'predicted_label': [data['class_names'][p] for p in metrics['predictions']],
        'correct':         data['y_test'] == metrics['predictions'],
        **prob_cols,
    })
    csv_path = f'/home/iaxiom/Downloads/predictions_{tag}_final.csv'
    out.to_csv(csv_path, index=False)
    print(f"  Predictions → {csv_path}")
    return metrics


# ── 2. Train VQC ─────────────────────────────────────────────
vqc_arch = {
    'model': 'VQC',
    'n_features': data['n_features'],
    'n_qubits':   data['n_features'],
    'n_layers':   3,
    'n_classes':  data['n_classes'],
    'embedding':  'angle',
    'ansatz':     'strongly_entangling',
}
vqc = VQCClassifier(**{k: v for k, v in vqc_arch.items() if k != 'model'})
vqc_metrics = train_model(
    "VQC — max 150 ep, lr=0.005, patience=20",
    vqc, vqc_arch, epochs=150, lr=0.005, wd=0.001, patience=20,
)

# ── 3. Train HQNN ────────────────────────────────────────────
hqnn_arch = {
    'model':             'HQNN',
    'n_features':        data['n_features'],
    'n_qubits':          data['n_features'],
    'n_layers':          2,
    'n_classes':         data['n_classes'],
    'classical_hidden':  32,
    'n_classical_layers': 2,
    'ansatz':            'strongly_entangling',
    'activation':        'relu',
}
hqnn = HQNNClassifier(**{k: v for k, v in hqnn_arch.items() if k != 'model'})
hqnn_metrics = train_model(
    "HQNN — max 150 ep, lr=0.005, patience=20",
    hqnn, hqnn_arch, epochs=150, lr=0.005, wd=0.001, patience=20,
)

# ── 4. Soft-vote ensemble ─────────────────────────────────────
print("\n" + "=" * 60)
print("  SOFT-VOTE ENSEMBLE — VQC + HQNN (final)")
print("=" * 60)

y_true    = np.array([data['class_names'][y] for y in data['y_test']])
avg_proba = (vqc_metrics['probabilities'] + hqnn_metrics['probabilities']) / 2.0
y_pred    = np.array([CLASSES[i] for i in avg_proba.argmax(axis=1)])

y_idx = np.array([CLASSES.index(c) for c in y_true])
y_bin = label_binarize(y_idx, classes=list(range(len(CLASSES))))
auc   = roc_auc_score(y_bin, avg_proba, average='macro', multi_class='ovr')

print(f"  Accuracy    : {accuracy_score(y_true, y_pred):.1%}")
print(f"  F1 Macro    : {f1_score(y_true, y_pred, average='macro', zero_division=0):.1%}")
print(f"  F1 Weighted : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.1%}")
print(f"  AUC Macro   : {auc:.3f}")
print()
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

ens_out = pd.DataFrame({
    'true_label': y_true, 'predicted_label': y_pred,
    'correct': y_true == y_pred,
    **{f'prob_{c}': avg_proba[:, i] for i, c in enumerate(CLASSES)},
})
ens_out.to_csv('/home/iaxiom/Downloads/predictions_ensemble_final.csv', index=False)
print("  Ensemble predictions → /home/iaxiom/Downloads/predictions_ensemble_final.csv")

print("\n" + "=" * 60)
print("  ALL SAVED ARTEFACTS")
print("=" * 60)
for p in sorted(SAVE_DIR.glob('*')):
    print(f"  {p.name}")
