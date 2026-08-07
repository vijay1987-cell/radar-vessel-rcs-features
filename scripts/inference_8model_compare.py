"""
8-model inference comparison on random_1000_vessels.csv

Feature set A — original raw (ellipse_area / euclid_size):
  Old HQNN5  : hqnn_weights.pt       (5q, 2L, 5-class)
  Old VQC5   : vqc_weights.pt        (5q, 3L, 5-class)
  Scaler     : REFITTED on test data (original lost)

Feature set B — current raw (down_range_extent / az_extent_m):
  XGB        : xgb_model.pkl         (5-class)
  HQNN8-v2   : hqnn8_v2_weights.pt  (8q, 3L, 5-class)
  Scaler     : saved preprocessor.pkl

Feature set C — RCS physically invariant (R4-corrected):
  GBT-RCS    : gbt_rcs_model.pkl     (4-class: 30,52,70,80)
  XGB-RCS    : xgb_rcs_model.pkl     (4-class)
  HQNN5-RCS  : hqnn5_rcs_weights.pt  (5q, 2L, 4-class)
  HQNN8-RCS  : hqnn8_rcs_weights.pt  (8q, 2L, 4-class)
  Scaler     : saved rcs_preprocessor.pkl
  Features   : log_peak_rcs, log_total_rcs, rcs_conc, aspect_ratio, footprint_m2
  Computation from raw CSV:
    log_peak_rcs  = ln(PeakAmplitude) + 4·ln(range)
    log_total_rcs = ln(TotalAmplitude) + 4·ln(range)
    rcs_conc      = PeakAmplitude / TotalAmplitude
    aspect_ratio  = down_range_extent / (az_extent_m × range)
    footprint_m2  = ln(down_range_extent × az_extent_m × range)

Note: RCS models are 4-class (no class 33); A/B models are 5-class.
      For RCS models, class 33 rows are excluded from evaluation.

Three scenarios:
  S1 — Type 70 (Cargo), all stations           (n ≈ 482)
  S2 — Type 70 (Cargo), Station 13 only        (n ≈ 76)
  S3 — Types 70+80, accuracy on Type 70 only
"""
import sys, warnings, json, pickle
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/iaxiom/projects/Hqnn/src')

SAVE = '/home/iaxiom/projects/Hqnn/saved_models'
DATA = '/home/iaxiom/Downloads/random_1000_vessels.csv'

FEATS_A   = ['azimuth', 'TotalAmplitude', 'ellipse_area', 'euclid_size', 'PeakAmplitude']
FEATS_B   = ['azimuth', 'PeakAmplitude', 'TotalAmplitude', 'down_range_extent', 'az_extent_m']
FEATS_RCS = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'aspect_ratio', 'footprint_m2']

# 5-class models know: 30,33,52,70,80
CLASS5 = ['30', '33', '52', '70', '80']
# 4-class RCS models know: 30,52,70,80 (no class 33)
CLASS4 = ['30', '52', '70', '80']

KNOWN5 = {30, 33, 52, 70, 80}
KNOWN4 = {30, 52, 70, 80}

# ── Load and engineer features ─────────────────────────────────────────────────
print("=" * 72)
print("LOADING & ENGINEERING FEATURES")
print("=" * 72)

df = pd.read_csv(DATA)
print(f"Raw: {len(df)} rows  |  Stations: {sorted(df['STATIONID'].unique())}")

R   = df['range'].clip(lower=1.0)
Ap  = df['PeakAmplitude'].clip(lower=1e-9)
At  = df['TotalAmplitude'].clip(lower=1e-9)
er  = df['down_range_extent'].clip(lower=0.1)
ec  = (df['az_extent_m'] * R).clip(lower=0.1)   # angular → metres

# Feature set B conversion: az_extent_m radians → metres (before saving back)
df['az_extent_m'] = ec

# RCS features (compute into new columns)
df['log_peak_rcs']  = np.log(Ap) + 4 * np.log(R)
df['log_total_rcs'] = np.log(At) + 4 * np.log(R)
df['rcs_conc']      = Ap / At
df['aspect_ratio']  = er / ec
df['footprint_m2']  = np.log(er * ec)

# Subset for 5-class models
df5 = df[df['Type'].isin(KNOWN5)].dropna(subset=FEATS_A + FEATS_B + FEATS_RCS).copy()
df5['Type'] = df5['Type'].astype(int)

# Subset for 4-class RCS models (no class 33)
df4 = df[df['Type'].isin(KNOWN4)].dropna(subset=FEATS_RCS).copy()
df4['Type'] = df4['Type'].astype(int)

print(f"5-class subset (A/B): {len(df5)} rows")
print(f"4-class subset (RCS): {len(df4)} rows")

sid5   = df5['STATIONID'].values
type5  = df5['Type'].values
sid4   = df4['STATIONID'].values
type4  = df4['Type'].values

# ── Feature scaling ────────────────────────────────────────────────────────────

# Set A: refit scaler on test data (original lost)
X_a_raw = df5[FEATS_A].values.astype(np.float32)
scaler_a = StandardScaler().fit(X_a_raw)
X_a = scaler_a.transform(X_a_raw)

# Set B: saved scaler
with open(f'{SAVE}/preprocessor.pkl', 'rb') as f:
    pre_b = pickle.load(f)
X_b = pre_b['scaler'].transform(df5[FEATS_B].values.astype(np.float32))

# Set C (RCS): saved scaler
with open(f'{SAVE}/rcs_preprocessor.pkl', 'rb') as f:
    pre_rcs = pickle.load(f)
X_rcs = pre_rcs['scaler'].transform(df4[FEATS_RCS].values.astype(np.float32))

print(f"\nScaling:")
print(f"  Set A — REFITTED on {len(X_a)}-row test data  (original scaler lost)")
print(f"  Set B — saved training StandardScaler")
print(f"  Set C — saved RCS training StandardScaler")


# ── Label encoders ─────────────────────────────────────────────────────────────
def encode(types, cls_list):
    return np.array([cls_list.index(str(t)) for t in types])

y5 = encode(type5, CLASS5)
y4 = encode(type4, CLASS4)

idx70_5 = CLASS5.index('70')   # 3
idx80_5 = CLASS5.index('80')   # 4
idx70_4 = CLASS4.index('70')   # 2
idx80_4 = CLASS4.index('80')   # 3


# ── Load models ────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("LOADING MODELS")
print("=" * 72)

from models.hqnn_model import HQNNClassifier
from models.vqc_model  import VQCClassifier

def load_hqnn(cfg_path, wt_path):
    with open(cfg_path) as f: c = json.load(f)
    m = HQNNClassifier(n_features=c['n_features'], n_qubits=c['n_qubits'],
                       n_layers=c['n_layers'], n_classes=c['n_classes'],
                       classical_hidden=c['classical_hidden'],
                       n_classical_layers=c['n_classical_layers'],
                       ansatz=c['ansatz'], activation=c['activation'])
    m.load_state_dict(torch.load(wt_path, map_location='cpu', weights_only=True))
    m.eval()
    return m, c

def load_vqc(cfg_path, wt_path):
    with open(cfg_path) as f: c = json.load(f)
    m = VQCClassifier(n_features=c['n_features'], n_qubits=c['n_qubits'],
                      n_layers=c['n_layers'], n_classes=c['n_classes'],
                      embedding=c['embedding'], ansatz=c['ansatz'])
    m.load_state_dict(torch.load(wt_path, map_location='cpu', weights_only=True))
    m.eval()
    return m, c

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)

hqnn_old, _ = load_hqnn(f'{SAVE}/hqnn_config.json',     f'{SAVE}/hqnn_weights.pt')
vqc_old,  _ = load_vqc( f'{SAVE}/vqc_config.json',      f'{SAVE}/vqc_weights.pt')
xgb_b       = load_pkl( f'{SAVE}/xgb_model.pkl')
hqnn8_b, _  = load_hqnn(f'{SAVE}/hqnn8_v2_config.json', f'{SAVE}/hqnn8_v2_weights.pt')
gbt_rcs     = load_pkl( f'{SAVE}/gbt_rcs_model.pkl')
xgb_rcs     = load_pkl( f'{SAVE}/xgb_rcs_model.pkl')
hqnn5_rcs, _= load_hqnn(f'{SAVE}/hqnn5_rcs_config.json',f'{SAVE}/hqnn5_rcs_weights.pt')
hqnn8_rcs, _= load_hqnn(f'{SAVE}/hqnn8_rcs_config.json',f'{SAVE}/hqnn8_rcs_weights.pt')

def cfg_note(path):
    with open(path) as f: c = json.load(f)
    return f"{c.get('n_qubits','—')}q  acc_train={c.get('accuracy',0)*100:.1f}%"

print(f"Old HQNN5  (A)  | n_features=5  acc_train=~54% (stored in pkl metrics)")
print(f"Old VQC5   (A)  | n_features=5  acc_train=~51%")
print(f"XGB        (B)  | {cfg_note(f'{SAVE}/xgb_config.json')}")
print(f"HQNN8-v2   (B)  | {cfg_note(f'{SAVE}/hqnn8_v2_config.json')}")
print(f"GBT-RCS    (C)  | {cfg_note(f'{SAVE}/gbt_rcs_config.json')}")
print(f"XGB-RCS    (C)  | {cfg_note(f'{SAVE}/xgb_rcs_config.json')}")
print(f"HQNN5-RCS  (C)  | {cfg_note(f'{SAVE}/hqnn5_rcs_config.json')}")
print(f"HQNN8-RCS  (C)  | {cfg_note(f'{SAVE}/hqnn8_rcs_config.json')}")


# ── Inference ──────────────────────────────────────────────────────────────────
def infer_torch(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()


def evaluate(preds, y_true, row_mask, eval_mask_fn,
             cls_list, idx70, idx80):
    """
    row_mask  : boolean over the dataset (which rows to infer on)
    eval_mask_fn : function(y_in_rows) → boolean mask within those rows
    """
    y_rows = y_true[row_mask]
    p_rows = preds

    e_mask = eval_mask_fn(y_rows)
    y_eval = y_rows[e_mask]
    p_eval = p_rows[e_mask]

    acc = accuracy_score(y_eval, p_eval)
    f1w = f1_score(y_eval, p_eval, average='weighted', zero_division=0)
    wrong = y_eval != p_eval
    wd = {cls_list[k]: v for k, v in Counter(p_eval[wrong]).items()} if wrong.sum() > 0 else {}

    cargo_tp = (p_eval[y_eval == idx70] == idx70).mean() if (y_eval==idx70).sum() > 0 else None
    c_tan    = (p_eval[y_eval == idx70] == idx80).mean() if (y_eval==idx70).sum() > 0 else None
    return dict(acc=acc, f1w=f1w, wd=wd, n_wrong=wrong.sum(),
                n_eval=len(y_eval), n_total=len(p_rows),
                cargo_tp=cargo_tp, cargo_as_tanker=c_tan)

# Scenario masks and eval functions (for 5-class datasets)
SCENARIOS_5 = {
    'S1 — Cargo / all stations': {
        'row_mask': type5 == 70,
        'eval_fn':  lambda y: np.ones(len(y), bool),
        'note':     f'{(type5==70).sum()} rows (Cargo, all stations)',
    },
    'S2 — Cargo / Station 13': {
        'row_mask': (type5 == 70) & (sid5 == 13),
        'eval_fn':  lambda y: np.ones(len(y), bool),
        'note':     f'{((type5==70)&(sid5==13)).sum()} rows (Cargo, Sta-13)',
    },
    'S3 — Cargo+Tanker / eval Cargo': {
        'row_mask': np.isin(type5, [70, 80]),
        'eval_fn':  lambda y: y == idx70_5,
        'note':     (f'{np.isin(type5,[70,80]).sum()} rows total; '
                     f'eval on {(type5==70).sum()} Cargo rows'),
    },
}

# Same scenarios for 4-class RCS datasets (class 33 not in df4)
SCENARIOS_4 = {
    'S1 — Cargo / all stations': {
        'row_mask': type4 == 70,
        'eval_fn':  lambda y: np.ones(len(y), bool),
        'note':     f'{(type4==70).sum()} rows (Cargo, all stations)',
    },
    'S2 — Cargo / Station 13': {
        'row_mask': (type4 == 70) & (sid4 == 13),
        'eval_fn':  lambda y: np.ones(len(y), bool),
        'note':     f'{((type4==70)&(sid4==13)).sum()} rows (Cargo, Sta-13)',
    },
    'S3 — Cargo+Tanker / eval Cargo': {
        'row_mask': np.isin(type4, [70, 80]),
        'eval_fn':  lambda y: y == idx70_4,
        'note':     (f'{np.isin(type4,[70,80]).sum()} rows total; '
                     f'eval on {(type4==70).sum()} Cargo rows'),
    },
}

# Model registry: (label, fset_tag, X_data, y_data, scenarios, cls_list, idx70, idx80, infer_fn)
MODELS = [
    ('Old HQNN5 (A)', 'A', X_a, y5, SCENARIOS_5, CLASS5, idx70_5, idx80_5,
     lambda X: infer_torch(hqnn_old, X)),
    ('Old VQC5  (A)', 'A', X_a, y5, SCENARIOS_5, CLASS5, idx70_5, idx80_5,
     lambda X: infer_torch(vqc_old, X)),
    ('XGB       (B)', 'B', X_b, y5, SCENARIOS_5, CLASS5, idx70_5, idx80_5,
     lambda X: xgb_b.predict(X)),
    ('HQNN8-v2  (B)', 'B', X_b, y5, SCENARIOS_5, CLASS5, idx70_5, idx80_5,
     lambda X: infer_torch(hqnn8_b, X)),
    ('GBT-RCS   (C)', 'C', X_rcs, y4, SCENARIOS_4, CLASS4, idx70_4, idx80_4,
     lambda X: gbt_rcs.predict(X)),
    ('XGB-RCS   (C)', 'C', X_rcs, y4, SCENARIOS_4, CLASS4, idx70_4, idx80_4,
     lambda X: xgb_rcs.predict(X)),
    ('HQNN5-RCS (C)', 'C', X_rcs, y4, SCENARIOS_4, CLASS4, idx70_4, idx80_4,
     lambda X: infer_torch(hqnn5_rcs, X)),
    ('HQNN8-RCS (C)', 'C', X_rcs, y4, SCENARIOS_4, CLASS4, idx70_4, idx80_4,
     lambda X: infer_torch(hqnn8_rcs, X)),
]

all_results = {}

for sc_name in SCENARIOS_5:
    print(f"\n{'='*72}")
    print(f"SCENARIO: {sc_name}")
    print(f"{'='*72}")

    for (mname, fset, X_full, y_full, scenarios, cls_list,
         idx70, idx80, infer_fn) in MODELS:

        sc = scenarios[sc_name]
        row_mask = sc['row_mask']
        X_sc = X_full[row_mask]
        preds = infer_fn(X_sc)
        ev = evaluate(preds, y_full, row_mask, sc['eval_fn'],
                      cls_list, idx70, idx80)
        all_results[(sc_name, mname)] = ev

        wd_str = ', '.join(
            f"→{k}:{v}({v/ev['n_eval']*100:.1f}%)"
            for k, v in sorted(ev['wd'].items())) if ev['wd'] else '—'

        print(f"\n  [{mname}]  n={ev['n_eval']}  "
              f"Acc={ev['acc']:.3f}  F1-wtd={ev['f1w']:.3f}")
        if ev['n_wrong'] > 0:
            print(f"    Errors {ev['n_wrong']}: {wd_str}")
        if ev['cargo_tp'] is not None and 'S3' in sc_name:
            print(f"    Cargo-TP={ev['cargo_tp']*100:.1f}%  "
                  f"→Tanker={ev['cargo_as_tanker']*100:.1f}%")


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY — 8 models × 3 scenarios")
print(f"{'Rank':<5}{'Model':<20}{'Feat':>4}  {'Classes':>7}  "
      f"{'S1-Acc':>7}  {'S2-Acc':>7}  {'S3-Acc':>7}  {'S3-CTP%':>8}")
print("-" * 72)

sc_list = list(SCENARIOS_5.keys())

rows = []
for (mname, fset, *_) in MODELS:
    ev1 = all_results[(sc_list[0], mname)]
    ev2 = all_results[(sc_list[1], mname)]
    ev3 = all_results[(sc_list[2], mname)]
    n_cls = 4 if fset == 'C' else 5
    ctp = f"{ev3['cargo_tp']*100:.1f}" if ev3['cargo_tp'] is not None else "—"
    rows.append((ev1['acc'], mname, fset, n_cls,
                 ev1['acc'], ev2['acc'], ev3['acc'], ctp))

rows.sort(key=lambda r: r[0], reverse=True)

for rank, (s1acc, mname, fset, n_cls, a1, a2, a3, ctp) in enumerate(rows, 1):
    print(f"  {rank:<3} {mname:<20} {fset:>4}  {n_cls:>7}-cls  "
          f"{a1:>7.3f}  {a2:>7.3f}  {a3:>7.3f}  {ctp:>8}")

print()
print("Feat: A=ellipse/euclid (original), B=down_range/az_m (current),")
print("      C=RCS R4-corrected (physically invariant)")
print("S3-Acc and C-TP% = accuracy/recall on Cargo rows when Tanker is also present")
print()
print("CAVEAT: Old HQNN5 / Old VQC5 (Set A) — scaler REFITTED on test data.")
print("        RCS feature distributions shift by ~30% for rcs_conc at this station.")
