"""
Inference script — run ensemble (VQC + HQNN) on a new CSV.

Usage:
    .venv/bin/python predict.py <path_to_new_data.csv>

The CSV must contain the same feature columns used during training:
    azimuth, TotalAmplitude, ellipse_area, euclid_size, PeakAmplitude

Outputs:
    predictions_<filename>.csv  in the same folder as the input CSV
"""

import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.vqc_model import VQCClassifier
from src.models.hqnn_model import HQNNClassifier

SAVE_DIR = Path(__file__).parent / 'saved_models'

# ── Load preprocessor ─────────────────────────────────────────
with open(SAVE_DIR / 'preprocessor.pkl', 'rb') as f:
    pre = pickle.load(f)

scaler        = pre['scaler']
label_encoder = pre['label_encoder']
feature_cols  = pre['feature_cols']
class_names   = pre['class_names']

# ── Load model configs ────────────────────────────────────────
with open(SAVE_DIR / 'vqc_config.json')  as f: vqc_cfg  = json.load(f)
with open(SAVE_DIR / 'hqnn_config.json') as f: hqnn_cfg = json.load(f)

# ── Rebuild model architectures ───────────────────────────────
vqc = VQCClassifier(
    n_features=vqc_cfg['n_features'],
    n_qubits  =vqc_cfg['n_qubits'],
    n_layers  =vqc_cfg['n_layers'],
    n_classes =vqc_cfg['n_classes'],
    embedding =vqc_cfg['embedding'],
    ansatz    =vqc_cfg['ansatz'],
)
vqc.load_state_dict(torch.load(SAVE_DIR / 'vqc_weights.pt', weights_only=True))
vqc.eval()

hqnn = HQNNClassifier(
    n_features       =hqnn_cfg['n_features'],
    n_qubits         =hqnn_cfg['n_qubits'],
    n_layers         =hqnn_cfg['n_layers'],
    n_classes        =hqnn_cfg['n_classes'],
    classical_hidden =hqnn_cfg['classical_hidden'],
    n_classical_layers=hqnn_cfg['n_classical_layers'],
    ansatz           =hqnn_cfg['ansatz'],
    activation       =hqnn_cfg['activation'],
)
hqnn.load_state_dict(torch.load(SAVE_DIR / 'hqnn_weights.pt', weights_only=True))
hqnn.eval()

print("Models loaded successfully.")
print(f"Features expected : {feature_cols}")
print(f"Classes           : {class_names}\n")


def predict(csv_path: str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Check all required features are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    X_df = df[feature_cols].copy()

    # Impute NaN values using training feature means before scaling
    nan_rows = X_df.isna().any(axis=1)
    if nan_rows.any():
        if scaler is not None and hasattr(scaler, 'mean_'):
            fill_values = {col: scaler.mean_[i] for i, col in enumerate(feature_cols)}
        else:
            fill_values = X_df.median().to_dict()
        X_df = X_df.fillna(fill_values)
        print(f"Warning: {nan_rows.sum()} row(s) had missing values — imputed with training means.")

    X = X_df.values.astype(np.float32)

    # Apply saved scaler
    if scaler is not None:
        X = scaler.transform(X)

    X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        vqc_proba  = torch.softmax(vqc(X_t),  dim=1).numpy()
        hqnn_proba = torch.softmax(hqnn(X_t), dim=1).numpy()

    avg_proba  = (vqc_proba + hqnn_proba) / 2.0
    pred_idx   = avg_proba.argmax(axis=1)
    pred_label = [class_names[i] for i in pred_idx]

    # Keep all non-feature columns (IDs, labels, extras) for traceability
    passthrough_cols = [c for c in df.columns if c not in feature_cols]
    result = df[passthrough_cols].copy() if passthrough_cols else pd.DataFrame(index=df.index)
    result.insert(0, 'predicted_class', pred_label)
    result.insert(1, 'confidence', avg_proba.max(axis=1).round(4))
    for i, c in enumerate(class_names):
        result[f'prob_{c}'] = avg_proba[:, i].round(4)

    out_path = csv_path.parent / f'predictions_{csv_path.stem}.csv'
    result.to_csv(out_path, index=False)
    print(f"Predictions saved → {out_path}")
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python predict.py <path_to_csv>")
        sys.exit(1)
    result = predict(sys.argv[1])
    print(f"\nFirst 10 predictions:")
    cols = ['predicted_class'] + [f'prob_{c}' for c in class_names]
    available = [c for c in cols if c in result.columns]
    print(result[available].head(10).to_string(index=False))
