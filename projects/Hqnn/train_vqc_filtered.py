import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_processing import DataProcessor
from src.models.vqc_model import VQCClassifier
from src.trainer import TorchTrainer, evaluate_model

CSV_PATH   = '/home/iaxiom/Downloads/Radar_data_filtered.csv'
LABEL_COL  = 'Label'
KEEP_CLASSES = [30, 33, 52, 70, 80]

# ── 1. Load & filter ──────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df[LABEL_COL].isin(KEEP_CLASSES)].reset_index(drop=True)
feature_cols = [c for c in df.columns if c != LABEL_COL]
print(f"Dataset: {df.shape[0]} rows, {len(feature_cols)} features: {feature_cols}")
print("Class counts:\n", df[LABEL_COL].value_counts().sort_index().to_string())

# ── 2. Preprocess ────────────────────────────────────────────
dp = DataProcessor()
df_clean = dp.clean(df, feature_cols, LABEL_COL, {
    'remove_duplicates': True,
    'missing_strategy': 'drop',
    'outlier_method': 'none',
})

preprocess_cfg = {
    'normalization': 'standard',
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'use_pca': False,
    'balancing': 'smote',
}
data = dp.preprocess(df_clean, feature_cols, LABEL_COL, preprocess_cfg)

# Class-weighted loss (inverse frequency)
counts = np.bincount(data['y_train'])
weights = data['n_classes'] / (counts + 1e-9)
weights = (weights / weights.sum() * data['n_classes']).tolist()
print(f"\nTrain: {data['train_size']}  Test: {data['test_size']}")
print(f"Classes: {data['class_names']}")
print(f"Class weights: {[round(w,3) for w in weights]}")

# ── 3. Build VQC ──────────────────────────────────────────────
n_features = data['n_features']   # 5
n_classes  = data['n_classes']    # 5
n_qubits   = n_features           # 1 qubit per feature

model = VQCClassifier(
    n_features=n_features,
    n_qubits=n_qubits,
    n_layers=3,
    n_classes=n_classes,
    embedding='angle',
    ansatz='strongly_entangling',
)
print(f"\nVQC: {n_qubits} qubits, 3 layers, angle embedding, strongly_entangling ansatz")

# ── 4. Train ──────────────────────────────────────────────────
train_cfg = {
    'optimizer': 'adam',
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 50,
    'weight_decay': 1e-4,
    'class_weights': weights,
}

print("\nTraining VQC (50 epochs)...\n")
trainer = TorchTrainer(model, train_cfg)

def cb(info):
    ep = info['epoch']
    if ep % 10 == 0 or ep == 1:
        print(f"  Epoch {ep:3d}/50 | loss={info['train_loss']:.4f} "
              f"acc={info['train_acc']:.1%} | val_acc={info['val_acc']:.1%}")

trainer.train(data['X_train'], data['y_train'],
              data['X_test'],  data['y_test'], callback=cb)

# ── 5. Evaluate ───────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATION ON TEST SET")
print("="*55)
metrics = evaluate_model(model, data['X_test'], data['y_test'],
                         data['class_names'], is_torch=True)

print(f"Accuracy    : {metrics['accuracy']:.1%}")
print(f"F1 Macro    : {metrics['f1_macro']:.1%}")
print(f"F1 Weighted : {metrics['f1_weighted']:.1%}")
auc = metrics.get('auc_macro')
print(f"AUC Macro   : {auc:.3f}" if auc else "AUC Macro   : N/A")

print("\nPer-class report:")
print(classification_report(data['y_test'], metrics['predictions'],
                             target_names=data['class_names'], zero_division=0))

print("Confusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(data['y_test'], metrics['predictions'])
cm_df = pd.DataFrame(cm,
    index=[f"true_{c}" for c in data['class_names']],
    columns=[f"pred_{c}" for c in data['class_names']])
print(cm_df.to_string())

# ── 6. Save predictions CSV ───────────────────────────────────
prob_cols = {f"prob_{c}": metrics['probabilities'][:, i]
             for i, c in enumerate(data['class_names'])}
out = pd.DataFrame({
    'true_label':      [data['class_names'][y] for y in data['y_test']],
    'predicted_label': [data['class_names'][p] for p in metrics['predictions']],
    'correct':         data['y_test'] == metrics['predictions'],
    **prob_cols,
})
out_path = '/home/iaxiom/Downloads/predictions_VQC_filtered.csv'
out.to_csv(out_path, index=False)
print(f"\nPredictions saved to {out_path}")
