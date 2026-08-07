"""
Inference comparison: XGB vs HQNN8-v2
Three scenarios on random_1000_vessels.csv (860 rows, new station):

  S1 — Type 70 (Cargo), all stations  (482 rows)
  S2 — Type 70 (Cargo), Station 13 only  (76 rows)
  S3 — Types 70+80, evaluate only on Type 70 rows  (724 total, ~482 evaluated)

Models compared:
  XGB   : saved_models/xgb_model.pkl          (5-class, V1 train)
  HQNN8 : saved_models/hqnn8_v2_weights.pt    (5-class, 8q, V2 train)

Preprocessing note:
  az_extent_m in the new CSV is in radians; must be converted to metres
  (× range) before applying the saved StandardScaler.
"""
import sys, warnings, json, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/iaxiom/projects/Hqnn/src')
SAVE = '/home/iaxiom/projects/Hqnn/saved_models'
DATA = '/home/iaxiom/Downloads/random_1000_vessels.csv'

FEATS = ['azimuth', 'PeakAmplitude', 'TotalAmplitude',
         'down_range_extent', 'az_extent_m']
LABEL_COL = 'Type'
KNOWN_TYPES = {30, 33, 52, 70, 80}


# ── Load and preprocess data ──────────────────────────────────────────────────
print("=" * 72)
print("LOADING DATA")
print("=" * 72)

df = pd.read_csv(DATA)
print(f"Raw CSV: {len(df)} rows  |  Stations: {sorted(df['STATIONID'].unique())}")
print(f"Type distribution:\n{df['Type'].value_counts().sort_index().to_string()}")

# Unit conversion: az_extent_m (radians) → metres
df['az_extent_m'] = df['az_extent_m'] * df['range']

# Keep only model-known classes and drop NaN
df = df[df[LABEL_COL].isin(KNOWN_TYPES)].dropna(subset=FEATS + [LABEL_COL]).copy()
df[LABEL_COL] = df[LABEL_COL].astype(int)
print(f"\nAfter filtering to known classes: {len(df)} rows")

# Load preprocessor (scaler fitted on training data)
with open(f'{SAVE}/preprocessor.pkl', 'rb') as f:
    pre = pickle.load(f)
scaler = pre['scaler']
le     = pre['label_encoder']          # maps str label → int index
class_names = pre['class_names']       # ['30','33','52','70','80']

# Scale features
X_full = scaler.transform(df[FEATS].values.astype(np.float32))
y_full = np.array([np.where(le.classes_ == str(t))[0][0]
                   for t in df[LABEL_COL]])   # integer encoded labels
sid    = df['STATIONID'].values
type_  = df[LABEL_COL].values

idx_70   = np.where(type_ == 70)[0]
idx_70_s13 = np.where((type_ == 70) & (sid == 13))[0]
idx_70_80  = np.where(np.isin(type_, [70, 80]))[0]
idx_70_in_70_80 = np.where((type_ == 70) & np.isin(type_, [70, 80]))[0]

scenarios = {
    'S1 — Type 70 / all stations': {
        'X': X_full[idx_70],
        'y': y_full[idx_70],
        'eval_mask': np.ones(len(idx_70), dtype=bool),
        'note': f'{len(idx_70)} rows (all Cargo, all stations)',
    },
    'S2 — Type 70 / Station 13':   {
        'X': X_full[idx_70_s13],
        'y': y_full[idx_70_s13],
        'eval_mask': np.ones(len(idx_70_s13), dtype=bool),
        'note': f'{len(idx_70_s13)} rows (Cargo, Sta-13 only)',
    },
    'S3 — Types 70+80 / eval only Type 70': {
        'X': X_full[idx_70_80],
        'y': y_full[idx_70_80],
        'eval_mask': type_[idx_70_80] == 70,   # only evaluate Type-70 rows
        'note': (f'{len(idx_70_80)} rows total '
                 f'({(type_[idx_70_80]==70).sum()} Cargo, '
                 f'{(type_[idx_70_80]==80).sum()} Tanker); '
                 'accuracy measured on Cargo rows only'),
    },
}

# Label indices for reference
idx_70_label = int(np.where(le.classes_ == '70')[0][0])
idx_80_label = int(np.where(le.classes_ == '80')[0][0])
print(f"\nLabel index: Cargo(70)={idx_70_label}  Tanker(80)={idx_80_label}")


# ── Load models ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("LOADING MODELS")
print("=" * 72)

# XGB
with open(f'{SAVE}/xgb_model.pkl', 'rb') as f:
    xgb_clf = pickle.load(f)
print(f"XGB loaded  ({xgb_clf.__class__.__name__})")

# HQNN8 v2
from models.hqnn_model import HQNNClassifier
with open(f'{SAVE}/hqnn8_v2_config.json') as f:
    hcfg = json.load(f)

hqnn = HQNNClassifier(
    n_features       = hcfg['n_features'],
    n_qubits         = hcfg['n_qubits'],
    n_layers         = hcfg['n_layers'],
    n_classes        = hcfg['n_classes'],
    classical_hidden = hcfg['classical_hidden'],
    n_classical_layers = hcfg['n_classical_layers'],
    ansatz           = hcfg['ansatz'],
    activation       = hcfg['activation'],
)
hqnn.load_state_dict(
    torch.load(f'{SAVE}/hqnn8_v2_weights.pt', map_location='cpu', weights_only=True))
hqnn.eval()
print(f"HQNN8-v2 loaded  (8 qubits, {hcfg['n_layers']} layers, "
      f"hidden={hcfg['classical_hidden']}, best_epoch={hcfg.get('best_epoch','-')})")


# ── Inference function ─────────────────────────────────────────────────────────
def run_inference_xgb(X):
    return xgb_clf.predict(X)

def run_inference_hqnn(X):
    with torch.no_grad():
        logits = hqnn(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()

def get_proba_xgb(X):
    return xgb_clf.predict_proba(X)

def get_proba_hqnn(X):
    with torch.no_grad():
        logits = hqnn(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


# ── Run all scenarios ─────────────────────────────────────────────────────────
def evaluate(preds, y_true, eval_mask, class_names, scenario_name, model_name):
    p_eval = preds[eval_mask]
    y_eval = y_true[eval_mask]

    acc = accuracy_score(y_eval, p_eval)
    f1  = f1_score(y_eval, p_eval, average='macro', zero_division=0)
    f1w = f1_score(y_eval, p_eval, average='weighted', zero_division=0)

    # Per-class precision/recall for classes present in eval set
    present_labels = sorted(np.unique(y_eval))
    present_names  = [class_names[i] for i in present_labels]
    rep = classification_report(y_eval, p_eval, labels=present_labels,
                                target_names=present_names, zero_division=0)

    # For S3: also report what fraction of Cargo rows were predicted as Cargo vs Tanker
    cargo_pred_as_cargo  = None
    cargo_pred_as_tanker = None
    if 'S3' in scenario_name:
        mask_cargo = y_eval == idx_70_label
        if mask_cargo.sum() > 0:
            cargo_pred_as_cargo  = (p_eval[mask_cargo] == idx_70_label).mean()
            cargo_pred_as_tanker = (p_eval[mask_cargo] == idx_80_label).mean()

    return dict(acc=acc, f1=f1, f1w=f1w, report=rep,
                n_eval=len(y_eval), n_total=len(preds),
                cargo_tp=cargo_pred_as_cargo,
                cargo_as_tanker=cargo_pred_as_tanker)


results = {}
print()
for sc_name, sc in scenarios.items():
    X, y, mask = sc['X'], sc['y'], sc['eval_mask']
    print(f"\n{'='*72}")
    print(f"SCENARIO: {sc_name}")
    print(f"  {sc['note']}")
    print(f"{'='*72}")

    p_xgb  = run_inference_xgb(X)
    p_hqnn = run_inference_hqnn(X)

    for model_name, preds in [('XGB', p_xgb), ('HQNN8-v2', p_hqnn)]:
        ev = evaluate(preds, y, mask, class_names, sc_name, model_name)
        print(f"\n  [{model_name}]  n_eval={ev['n_eval']}  "
              f"Acc={ev['acc']:.3f}  F1-macro={ev['f1']:.3f}  F1-wtd={ev['f1w']:.3f}")
        if ev['cargo_tp'] is not None:
            print(f"    Cargo predicted as Cargo  : {ev['cargo_tp']*100:.1f}%")
            print(f"    Cargo predicted as Tanker : {ev['cargo_as_tanker']*100:.1f}%")
        # Show where misclassified rows go
        p_eval = preds[mask]
        y_eval = y[mask]
        wrong  = y_eval != p_eval
        if wrong.sum() > 0:
            from collections import Counter
            wrong_preds = Counter(p_eval[wrong])
            wrong_str = ', '.join(
                f"→{class_names[c]}:{n}({n/len(y_eval)*100:.1f}%)"
                for c, n in sorted(wrong_preds.items()))
            print(f"    Misclassified {wrong.sum()} rows: {wrong_str}")
        print(f"\n  Classification report [{model_name}]:")
        for line in ev['report'].splitlines():
            print(f"    {line}")

        results[(sc_name, model_name)] = ev


# ── Summary comparison table ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY — XGB vs HQNN8-v2 across all scenarios")
print("=" * 72)
print(f"{'Scenario':<38}  {'Model':<10}  {'N eval':>6}  "
      f"{'Acc':>6}  {'F1-mac':>7}  {'Cargo-TP%':>10}")
print("-" * 85)

for sc_name, sc in scenarios.items():
    for model_name in ['XGB', 'HQNN8-v2']:
        ev = results[(sc_name, model_name)]
        ctp = f"{ev['cargo_tp']*100:.1f}%" if ev['cargo_tp'] is not None else "   —"
        tag = sc_name.replace('SCENARIO: ', '')
        print(f"  {tag[:36]:<38}  {model_name:<10}  {ev['n_eval']:>6}  "
              f"{ev['acc']:>6.3f}  {ev['f1']:>7.3f}  {ctp:>10}")

print()
print("Note: Cargo-TP% = fraction of true Cargo detections predicted as Cargo")
print("      Applies to S3 only (when Tanker is present as interference)")
