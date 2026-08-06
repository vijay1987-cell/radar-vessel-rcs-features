"""
Train classical and HQNN models on RCS features for maritime vessel classification.

Usage:
    python scripts/train_rcs_models.py \
        --data /path/to/rcs_train_4class.csv \
        --models-dir ./saved_models \
        --n-per-class 300

The dataset (rcs_train_4class.csv) is proprietary and not included in this
repository. See README for the expected column format.
"""
import sys, argparse, json, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.hqnn_model import HQNNClassifier
from src.models.classical_model import ClassicalModel
from src.trainer import TorchTrainer

FEATS = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'aspect_ratio', 'footprint_m2']
TRAIN_TYPES = [30, 52, 70, 80]
CLASS_NAMES = ['30', '52', '70', '80']
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train RCS vessel classification models")
    p.add_argument('--data',       required=True, help="Path to rcs_train_4class.csv")
    p.add_argument('--models-dir', default='./saved_models', help="Where to save model files")
    p.add_argument('--n-per-class', type=int, default=300, help="Detections per class (training)")
    p.add_argument('--epochs',     type=int, default=30,  help="HQNN training epochs")
    p.add_argument('--no-hqnn',    action='store_true',   help="Skip HQNN training (faster)")
    return p.parse_args()


def ece(proba, y_enc, n_bins=15):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    corr = (pred == y_enc).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        val += (m.sum() / len(y_enc)) * abs(corr[m].mean() - conf[m].mean())
    return val


def load_and_sample(data_path, n_per_class, seed):
    df = pd.read_csv(data_path)
    parts = []
    for t in TRAIN_TYPES:
        sub = df[df['Type'].astype(int) == t]
        if 'range_bin' in sub.columns:
            bins_present = sub['range_bin'].dropna().unique()
            per_bin = max(1, n_per_class // len(bins_present))
            bin_parts = [
                s.sample(min(len(s), per_bin), random_state=seed)
                for rb in bins_present
                for s in [sub[sub['range_bin'] == rb]]
            ]
            chunk = pd.concat(bin_parts).sample(
                min(len(pd.concat(bin_parts)), n_per_class), random_state=seed)
        else:
            chunk = sub.sample(min(len(sub), n_per_class), random_state=seed)
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.data}")
    train_df = load_and_sample(args.data, args.n_per_class, SEED)
    print(f"Training set: {len(train_df)} rows")
    print(train_df['Type'].astype(int).value_counts().sort_index().to_string())

    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[FEATS].fillna(0).values.astype(np.float32))
    y = np.array([CLASS_NAMES.index(str(int(t))) for t in train_df['Type'].values])

    preprocessor = {'scaler': scaler, 'class_names': CLASS_NAMES, 'feature_cols': FEATS}
    with open(models_dir / 'rcs_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train={len(X_train)}, Val={len(X_val)}")

    # ── Classical models ──────────────────────────────────────────────────────
    for model_name, params in [
        ('GBT', dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                     subsample=0.8, random_state=SEED)),
        ('XGBoost', dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, use_label_encoder=False,
                         eval_metric='mlogloss', random_state=SEED, verbosity=0)),
    ]:
        print(f"\nTraining {model_name}...")
        try:
            clf = ClassicalModel(model_name, params)
            clf.fit(X_train, y_train, X_val, y_val)
            proba = clf.predict_proba(X_val)
            pred  = proba.argmax(axis=1)
            acc   = accuracy_score(y_val, pred)
            f1    = f1_score(y_val, pred, average='macro', zero_division=0)
            print(f"  Val  Accuracy={acc:.3f}  F1={f1:.3f}")
            tag = model_name.lower().replace(' ', '_').replace('boost', 'xgb').replace('gradient_xgb', 'gbt')
            with open(models_dir / f'{tag}_rcs_model.pkl', 'wb') as fh:
                pickle.dump(clf.model, fh)
            with open(models_dir / f'{tag}_rcs_config.json', 'w') as fh:
                json.dump({'model': model_name, 'feature_cols': FEATS,
                           'class_names': CLASS_NAMES,
                           'accuracy': round(float(acc), 4),
                           'f1_macro': round(float(f1), 4)}, fh, indent=2)
        except Exception as e:
            print(f"  Skipped {model_name}: {e}")

    # ── HQNN models ───────────────────────────────────────────────────────────
    if not args.no_hqnn:
        for n_qubits in [5, 8]:
            print(f"\nTraining HQNN {n_qubits}q...")
            model = HQNNClassifier(
                n_features=5, n_qubits=n_qubits, n_layers=2,
                n_classes=len(CLASS_NAMES), classical_hidden=32,
                n_classical_layers=2, ansatz='strongly_entangling', activation='relu',
            )
            cfg_train = dict(epochs=args.epochs, batch_size=32, learning_rate=0.01,
                             weight_decay=1e-4, optimizer='adam', patience=10)
            trainer = TorchTrainer(model, cfg_train)
            t0 = time.time()
            trainer.train(X_train, y_train, X_val, y_val, patience=10)
            elapsed = time.time() - t0

            model.eval()
            with torch.no_grad():
                proba = torch.softmax(
                    model(torch.tensor(X_val, dtype=torch.float32)), dim=1).numpy()
            pred = proba.argmax(axis=1)
            acc  = accuracy_score(y_val, pred)
            f1   = f1_score(y_val, pred, average='macro', zero_division=0)
            print(f"  Val  Accuracy={acc:.3f}  F1={f1:.3f}  time={elapsed/60:.1f}min")

            torch.save(model.state_dict(), models_dir / f'hqnn{n_qubits}_rcs_weights.pt')
            with open(models_dir / f'hqnn{n_qubits}_rcs_config.json', 'w') as fh:
                json.dump({'model': 'HQNN', 'n_qubits': n_qubits, 'n_layers': 2,
                           'feature_cols': FEATS, 'class_names': CLASS_NAMES,
                           'accuracy': round(float(acc), 4),
                           'f1_macro': round(float(f1), 4)}, fh, indent=2)

    print("\nDone. Models saved to:", models_dir)


if __name__ == '__main__':
    main()
