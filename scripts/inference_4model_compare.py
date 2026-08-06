"""
4-model inference comparison on random_1000_vessels.csv

Feature set A — ORIGINAL (ellipse_area / euclid_size):
  Old HQNN  : hqnn_weights.pt      (5q, 2 layers)
  Old VQC   : vqc_weights.pt       (5q, 3 layers)
  Features  : azimuth, TotalAmplitude, ellipse_area, euclid_size, PeakAmplitude
  Scaler    : REFITTED on test data (original scaler not saved — caveat noted)

Feature set B — CURRENT (down_range_extent / az_extent_m):
  XGB       : xgb_model.pkl
  HQNN8-v2  : hqnn8_v2_weights.pt  (8q, 3 layers)
  Features  : azimuth, PeakAmplitude, TotalAmplitude, down_range_extent, az_extent_m
  Scaler    : saved preprocessor.pkl (original training scaler)

Note — az_extent_m in this CSV is raw angular extent (radians). It must be
multiplied by range before applying the current scaler. ellipse_area and
euclid_size are pre-computed in the CSV and used as-is.

Three scenarios:
  S1 — Type 70 (Cargo), all stations           (n ≈ 482)
  S2 — Type 70 (Cargo), Station 13 only        (n ≈ 76)
  S3 — Types 70+80, accuracy measured on Type 70 rows only
"""
import sys, warnings, json, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/iaxiom/projects/Hqnn/src')

SAVE = '/home/iaxiom/projects/Hqnn/saved_models'
DATA = '/home/iaxiom/Downloads/random_1000_vessels.csv'
KNOWN_TYPES = {30, 33, 52, 70, 80}

FEATS_OLD = ['azimuth', 'TotalAmplitude', 'ellipse_area', 'euclid_size', 'PeakAmplitude']
FEATS_NEW = ['azimuth', 'PeakAmplitude', 'TotalAmplitude', 'down_range_extent', 'az_extent_m']


# ── Load raw CSV ───────────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING & PREPROCESSING DATA")
print("=" * 72)

df_raw = pd.read_csv(DATA)
print(f"Raw CSV: {len(df_raw)} rows  |  Stations: {sorted(df_raw['STATIONID'].unique())}")

# Conversion for Feature set B: az_extent_m radians → metres
df_raw['az_extent_m'] = df_raw['az_extent_m'] * df_raw['range']

# Filter to model-known classes
df = df_raw[df_raw['Type'].isin(KNOWN_TYPES)].dropna(
        subset=FEATS_OLD + FEATS_NEW).copy()
df['Type'] = df['Type'].astype(int)
print(f"After filtering to known classes (30,33,52,70,80): {len(df)} rows")
print(f"Type 70: {(df['Type']==70).sum()}  Type 80: {(df['Type']==80).sum()}")

# Class name lists (same for all models)
class_names = ['30', '33', '52', '70', '80']   # consistent with LE order

def encode_labels(types):
    """Map integer type → index in class_names list."""
    return np.array([class_names.index(str(t)) for t in types])

y_all  = encode_labels(df['Type'])
sid    = df['STATIONID'].values
type_  = df['Type'].values
idx_70_label = class_names.index('70')   # 3
idx_80_label = class_names.index('80')   # 4


# ── Feature set A — old features, refit scaler on test data ───────────────────
X_old_raw = df[FEATS_OLD].values.astype(np.float32)
scaler_old = StandardScaler().fit(X_old_raw)          # *** REFIT on test data ***
X_old = scaler_old.transform(X_old_raw)

# ── Feature set B — new features, saved scaler ────────────────────────────────
with open(f'{SAVE}/preprocessor.pkl', 'rb') as f:
    pre = pickle.load(f)
scaler_new = pre['scaler']
X_new = scaler_new.transform(df[FEATS_NEW].values.astype(np.float32))

print(f"\nFeature set A (old): {FEATS_OLD}")
print(f"  Scaler: REFITTED on {len(X_old_raw)}-row test set  ← caveat: not original training scaler")
print(f"Feature set B (new): {FEATS_NEW}")
print(f"  Scaler: saved training StandardScaler")


# ── Scenario index masks ───────────────────────────────────────────────────────
masks = {
    'S1 — Type 70 / all stations': {
        'row_mask':  type_ == 70,
        'eval_mask': np.ones((type_ == 70).sum(), dtype=bool),
        'note': f'{(type_==70).sum()} rows  (Cargo, all stations)',
    },
    'S2 — Type 70 / Station 13': {
        'row_mask':  (type_ == 70) & (sid == 13),
        'eval_mask': np.ones(((type_==70) & (sid==13)).sum(), dtype=bool),
        'note': f'{((type_==70)&(sid==13)).sum()} rows  (Cargo, Sta-13)',
    },
    'S3 — Types 70+80 / eval Type 70 only': {
        'row_mask':  np.isin(type_, [70, 80]),
        'eval_mask': type_[np.isin(type_, [70, 80])] == 70,
        'note': (f'{np.isin(type_,[70,80]).sum()} rows total  '
                 f'({(type_==70).sum()} Cargo / {(type_==80).sum()} Tanker); '
                 'accuracy on Cargo rows only'),
    },
}


# ── Load models ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("LOADING MODELS")
print("=" * 72)

from models.hqnn_model import HQNNClassifier
from models.vqc_model  import VQCClassifier

# Old HQNN (5q)
with open(f'{SAVE}/hqnn_config.json') as f: hcfg_old = json.load(f)
hqnn_old = HQNNClassifier(
    n_features=hcfg_old['n_features'], n_qubits=hcfg_old['n_qubits'],
    n_layers=hcfg_old['n_layers'], n_classes=hcfg_old['n_classes'],
    classical_hidden=hcfg_old['classical_hidden'],
    n_classical_layers=hcfg_old['n_classical_layers'],
    ansatz=hcfg_old['ansatz'], activation=hcfg_old['activation'])
hqnn_old.load_state_dict(
    torch.load(f'{SAVE}/hqnn_weights.pt', map_location='cpu', weights_only=True))
hqnn_old.eval()
print(f"Old HQNN  loaded  ({hcfg_old['n_qubits']}q, {hcfg_old['n_layers']} layers)  — Feature set A")

# Old VQC (5q)
with open(f'{SAVE}/vqc_config.json') as f: vcfg_old = json.load(f)
vqc_old = VQCClassifier(
    n_features=vcfg_old['n_features'], n_qubits=vcfg_old['n_qubits'],
    n_layers=vcfg_old['n_layers'], n_classes=vcfg_old['n_classes'],
    embedding=vcfg_old['embedding'], ansatz=vcfg_old['ansatz'])
vqc_old.load_state_dict(
    torch.load(f'{SAVE}/vqc_weights.pt', map_location='cpu', weights_only=True))
vqc_old.eval()
print(f"Old VQC   loaded  ({vcfg_old['n_qubits']}q, {vcfg_old['n_layers']} layers)  — Feature set A")

# New XGB
with open(f'{SAVE}/xgb_model.pkl', 'rb') as f: xgb = pickle.load(f)
print(f"New XGB   loaded  — Feature set B")

# New HQNN8-v2 (8q)
with open(f'{SAVE}/hqnn8_v2_config.json') as f: hcfg_new = json.load(f)
hqnn_new = HQNNClassifier(
    n_features=hcfg_new['n_features'], n_qubits=hcfg_new['n_qubits'],
    n_layers=hcfg_new['n_layers'], n_classes=hcfg_new['n_classes'],
    classical_hidden=hcfg_new['classical_hidden'],
    n_classical_layers=hcfg_new['n_classical_layers'],
    ansatz=hcfg_new['ansatz'], activation=hcfg_new['activation'])
hqnn_new.load_state_dict(
    torch.load(f'{SAVE}/hqnn8_v2_weights.pt', map_location='cpu', weights_only=True))
hqnn_new.eval()
print(f"New HQNN8 loaded  ({hcfg_new['n_qubits']}q, {hcfg_new['n_layers']} layers)  — Feature set B")


# ── Inference helpers ─────────────────────────────────────────────────────────
def infer_torch(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()

def infer_proba_torch(model, X):
    with torch.no_grad():
        return torch.softmax(
            model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()


# ── Per-scenario evaluation ───────────────────────────────────────────────────
def evaluate(preds, y_true, eval_mask):
    p, y = preds[eval_mask], y_true[eval_mask]
    acc = accuracy_score(y, p)
    f1w = f1_score(y, p, average='weighted', zero_division=0)
    wrong = y != p
    from collections import Counter
    wrong_dist = {class_names[k]: v for k, v in Counter(p[wrong]).items()} if wrong.sum() > 0 else {}
    cargo_tp = (p[y == idx_70_label] == idx_70_label).mean() if (y == idx_70_label).sum() > 0 else None
    cargo_as_tanker = (p[y == idx_70_label] == idx_80_label).mean() if (y == idx_70_label).sum() > 0 else None
    return dict(acc=acc, f1w=f1w, wrong_dist=wrong_dist, n_wrong=wrong.sum(),
                n_eval=len(y), cargo_tp=cargo_tp, cargo_as_tanker=cargo_as_tanker)

MODEL_SPECS = [
    ('Old HQNN  (5q, ellipse/euclid)', 'A', lambda X_a, X_b: infer_torch(hqnn_old, X_a)),
    ('Old VQC   (5q, ellipse/euclid)', 'A', lambda X_a, X_b: infer_torch(vqc_old,  X_a)),
    ('New XGB   (down_range/az_m)',    'B', lambda X_a, X_b: xgb.predict(X_b)),
    ('New HQNN8 (8q, down_range/az_m)','B', lambda X_a, X_b: infer_torch(hqnn_new, X_b)),
]

all_results = {}

for sc_name, sc in masks.items():
    row_mask  = sc['row_mask']
    eval_mask = sc['eval_mask']
    y_sc      = y_all[row_mask]
    X_a_sc    = X_old[row_mask]
    X_b_sc    = X_new[row_mask]

    print(f"\n{'='*72}")
    print(f"SCENARIO: {sc_name}")
    print(f"  {sc['note']}")
    print(f"{'='*72}")

    for mname, fset, infer_fn in MODEL_SPECS:
        preds = infer_fn(X_a_sc, X_b_sc)
        ev    = evaluate(preds, y_sc, eval_mask)
        all_results[(sc_name, mname)] = ev

        wd_str = ', '.join(f"→{k}:{v}({v/ev['n_eval']*100:.1f}%)"
                           for k, v in sorted(ev['wrong_dist'].items()))

        print(f"\n  [{mname}]  n={ev['n_eval']}  "
              f"Acc={ev['acc']:.3f}  F1-wtd={ev['f1w']:.3f}")
        if ev['n_wrong'] > 0:
            print(f"    Misclassified {ev['n_wrong']}: {wd_str}")
        if ev['cargo_tp'] is not None and 'S3' in sc_name:
            print(f"    Cargo-TP={ev['cargo_tp']*100:.1f}%  "
                  f"Cargo→Tanker={ev['cargo_as_tanker']*100:.1f}%")


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY — all 4 models × 3 scenarios")
print("=" * 72)
print(f"{'Scenario':<38}  {'Model':<34}  {'N':>5}  {'Acc':>5}  {'F1w':>5}  {'C-TP%':>6}")
print("-" * 100)

for sc_name in masks:
    for mname, fset, _ in MODEL_SPECS:
        ev  = all_results[(sc_name, mname)]
        ctp = f"{ev['cargo_tp']*100:.1f}" if ev['cargo_tp'] is not None else "  —"
        sc_short = sc_name[:36]
        print(f"  {sc_short:<38}  {mname:<34}  {ev['n_eval']:>5}  "
              f"{ev['acc']:>5.3f}  {ev['f1w']:>5.3f}  {ctp:>6}")
    print()

print("C-TP% = Cargo true-positive rate (only shown for S3)")
print()
print("CAVEAT — Old HQNN / Old VQC: original training scaler not saved.")
print("  Scaler was REFITTED on the 822-row test subset.")
print("  Absolute accuracy for old models is indicative only;")
print("  relative comparisons between old HQNN and old VQC are valid.")
