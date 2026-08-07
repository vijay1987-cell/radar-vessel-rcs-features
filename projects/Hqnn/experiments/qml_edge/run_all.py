"""
QML Edge Experiments — 4 targeted investigations
=================================================
1. Low-data regime   : 100 samples/class — does HQNN close the gap vs GBT?
2. Higher qubits     : 8q VQC + HQNN vs 5q baseline
3. QKE               : Quantum Kernel Estimation (ZZFeatureMap) vs RBF-SVM
4. Confidence calib  : ECE, mean confidence correct vs incorrect for all models

Results saved to: experiments/qml_edge/results/
"""
import sys, os, json, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE

import pickle
from src.models.vqc_model import VQCClassifier
from src.models.hqnn_model import HQNNClassifier
from src.models.qke_model import QKEClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = '/home/iaxiom/Downloads/radarfeatureL_Study.csv'
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')
SAVED_MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models')
LABEL_COL    = 'Type'
FEATURE_COLS = ['azimuth', 'PeakAmplitude', 'TotalAmplitude',
                'down_range_extent', 'az_extent_m']
KEEP_TYPES   = [30, 33, 52, 70, 80]
RANDOM_STATE = 42

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_and_clean(n_per_class: int):
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[LABEL_COL].isin(KEEP_TYPES)]
    df = df[df['PeakAmplitude'] > 0]
    df['az_extent_m'] = df['range'] * df['cross_range_extent']
    df = df[FEATURE_COLS + [LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(str)
    parts = [grp.sample(min(len(grp), n_per_class), random_state=RANDOM_STATE)
             for _, grp in df.groupby(LABEL_COL)]
    df_s = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    le = LabelEncoder()
    y  = le.fit_transform(df_s[LABEL_COL])
    X  = df_s[FEATURE_COLS].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    class_names = [str(c) for c in le.classes_]
    print(f"  [{n_per_class}/class] train={len(X_tr)} (post-SMOTE) | test={len(X_te)}")
    return X_tr, X_te, y_tr, y_te, class_names


def train_torch(model, X_tr, y_tr, X_te, y_te,
                lr=0.005, epochs=150, patience=20, batch=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    best_val, best_ep, best_w = 0.0, 0, None
    no_improve = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch):
            b = idx[i:i+batch]
            opt.zero_grad()
            loss = criterion(model(X_tr_t[b]), y_tr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
        if val_acc > best_val:
            best_val, best_ep, best_w = val_acc, ep, copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if ep % 20 == 0 or ep == 1:
            with torch.no_grad():
                tr_acc = (model(X_tr_t).argmax(1) == y_tr_t).float().mean().item()
            print(f"    Ep {ep:3d} | val={val_acc:.1%} [best={best_val:.1%} @ep{best_ep}] | tr={tr_acc:.1%}")
        if no_improve >= patience:
            print(f"    Early stop ep{ep} — best={best_val:.1%} @ep{best_ep}")
            break
    model.load_state_dict(best_w)
    model.eval()
    elapsed = time.time() - t0
    print(f"    Training time: {elapsed:.0f}s")
    return model, best_ep, elapsed


def evaluate_torch(model, X_te, y_te, class_names):
    X_t = torch.tensor(X_te, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        proba = torch.softmax(logits, dim=1).numpy()
    preds = proba.argmax(axis=1)
    return _metrics(y_te, preds, proba, class_names)


def evaluate_sklearn(model, X_te, y_te, class_names):
    proba = model.predict_proba(X_te)
    preds = proba.argmax(axis=1)
    return _metrics(y_te, preds, proba, class_names)


def _metrics(y_te, preds, proba, class_names):
    acc  = accuracy_score(y_te, preds)
    f1   = f1_score(y_te, preds, average='macro')
    auc  = roc_auc_score(y_te, proba, multi_class='ovr', average='macro')
    cm   = confusion_matrix(y_te, preds)
    rep  = classification_report(y_te, preds, target_names=class_names, output_dict=True)
    recall_70_80 = (rep[class_names[3]]['recall'], rep[class_names[4]]['recall'])
    return dict(acc=acc, f1=f1, auc=auc, cm=cm.tolist(),
                report=rep, proba=proba, preds=preds,
                recall_70=recall_70_80[0], recall_80=recall_70_80[1])


def ece(proba, y_true, n_bins=15):
    """Expected Calibration Error."""
    conf = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val, rel_x, rel_y = 0.0, [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece_val += (mask.sum() / len(y_true)) * abs(acc_bin - conf_bin)
        rel_x.append(float(conf_bin))
        rel_y.append(float(acc_bin))
    return float(ece_val), rel_x, rel_y


def print_result(name, m):
    print(f"\n  {'='*55}")
    print(f"  {name}")
    print(f"  {'='*55}")
    print(f"  Acc={m['acc']:.1%}  F1={m['f1']:.1%}  AUC={m['auc']:.3f}")
    print(f"  Type70 recall={m['recall_70']:.1%}  Type80 recall={m['recall_80']:.1%}")
    print(f"  ECE={m.get('ece', float('nan')):.4f}  "
          f"conf_correct={m.get('conf_correct', float('nan')):.3f}  "
          f"conf_wrong={m.get('conf_wrong', float('nan')):.3f}")


def add_calibration(m):
    proba  = np.array(m['proba'])
    y_te   = np.array(m['preds'])   # we'll use actual test labels
    return m   # caller passes y_te separately


def save_json(name, data):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    serialisable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in data.items() if k != 'proba'}
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)


def save_torch_v2(name, model, cfg_dict):
    """Save torch model to saved_models/ as v2 so the Streamlit UI can load it."""
    stem = f'{name}_v2'
    torch.save(model.state_dict(),
               os.path.join(SAVED_MODELS, f'{stem}_weights.pt'))
    with open(os.path.join(SAVED_MODELS, f'{stem}_config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=2)
    print(f'  Saved → saved_models/{stem}_weights.pt + {stem}_config.json')


def save_sklearn_v2(name, model, cfg_dict):
    """Save sklearn model to saved_models/ as v2 so the Streamlit UI can load it."""
    stem = f'{name}_v2'
    with open(os.path.join(SAVED_MODELS, f'{stem}_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(SAVED_MODELS, f'{stem}_config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=2)
    print(f'  Saved → saved_models/{stem}_model.pkl + {stem}_config.json')


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — Low-data regime: 100 samples/class
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  EXPERIMENT 1: LOW-DATA REGIME (100 samples/class)')
print('='*60)

X_tr, X_te, y_tr, y_te, CN = load_and_clean(100)

results_all = {}

_COMMON_CFG = dict(
    n_features=5, n_classes=5,
    feature_cols=FEATURE_COLS,
    class_names=CN,
)

# GBT at 100/class
print('\n  [GBT]')
t0=time.time()
gbt_100 = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=RANDOM_STATE)
gbt_100.fit(X_tr, y_tr)
m_gbt100 = evaluate_sklearn(gbt_100, X_te, y_te, CN)
m_gbt100['train_time'] = time.time()-t0
print_result('GBT 100/class', m_gbt100)
results_all['gbt_100'] = m_gbt100
save_sklearn_v2('gbt', gbt_100, {**_COMMON_CFG, 'model': 'GBT',
                                  'note': '100/class, 300est, depth5, lr0.05'})

# RF at 100/class
print('\n  [RF]')
t0=time.time()
rf_100 = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                 class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf_100.fit(X_tr, y_tr)
m_rf100 = evaluate_sklearn(rf_100, X_te, y_te, CN)
m_rf100['train_time'] = time.time()-t0
print_result('RF 100/class', m_rf100)
results_all['rf_100'] = m_rf100
save_sklearn_v2('rf', rf_100, {**_COMMON_CFG, 'model': 'RF',
                                'note': '100/class, 300trees, balanced'})

# VQC 5q at 100/class
print('\n  [VQC 5q, 3L] — 100/class')
vqc5 = VQCClassifier(n_features=5, n_qubits=5, n_layers=3, n_classes=5)
vqc5, ep_vqc5, t_vqc5 = train_torch(vqc5, X_tr, y_tr, X_te, y_te, epochs=150)
m_vqc5 = evaluate_torch(vqc5, X_te, y_te, CN)
m_vqc5.update(best_epoch=ep_vqc5, train_time=t_vqc5)
print_result('VQC 5q/100', m_vqc5)
results_all['vqc5_100'] = m_vqc5
save_torch_v2('vqc5', vqc5, {**_COMMON_CFG, 'model': 'VQC',
                               'n_qubits': 5, 'n_layers': 3,
                               'embedding': 'angle', 'ansatz': 'strongly_entangling',
                               'best_epoch': ep_vqc5, 'note': '100/class v2'})

# HQNN 5q at 100/class
print('\n  [HQNN 5q, 2L, h=32] — 100/class')
hqnn5 = HQNNClassifier(n_features=5, n_qubits=5, n_layers=2, n_classes=5,
                        classical_hidden=32, n_classical_layers=2)
hqnn5, ep_hqnn5, t_hqnn5 = train_torch(hqnn5, X_tr, y_tr, X_te, y_te, epochs=150)
m_hqnn5 = evaluate_torch(hqnn5, X_te, y_te, CN)
m_hqnn5.update(best_epoch=ep_hqnn5, train_time=t_hqnn5)
print_result('HQNN 5q/100', m_hqnn5)
results_all['hqnn5_100'] = m_hqnn5
save_torch_v2('hqnn5', hqnn5, {**_COMMON_CFG, 'model': 'HQNN',
                                 'n_qubits': 5, 'n_layers': 2,
                                 'classical_hidden': 32, 'n_classical_layers': 2,
                                 'ansatz': 'strongly_entangling', 'activation': 'relu',
                                 'best_epoch': ep_hqnn5, 'note': '100/class v2'})


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — Higher qubits: 8q VQC + HQNN (still 100/class for fairness)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  EXPERIMENT 2: HIGHER QUBITS (8 qubits, 4 layers)')
print('='*60)
print('  (reusing 100/class split from Exp 1)')

# VQC 8q
print('\n  [VQC 8q, 4L] — 100/class')
vqc8 = VQCClassifier(n_features=5, n_qubits=8, n_layers=4, n_classes=5)
vqc8, ep_vqc8, t_vqc8 = train_torch(vqc8, X_tr, y_tr, X_te, y_te, epochs=150)
m_vqc8 = evaluate_torch(vqc8, X_te, y_te, CN)
m_vqc8.update(best_epoch=ep_vqc8, train_time=t_vqc8)
print_result('VQC 8q/100', m_vqc8)
results_all['vqc8_100'] = m_vqc8
save_torch_v2('vqc8', vqc8, {**_COMMON_CFG, 'model': 'VQC',
                               'n_qubits': 8, 'n_layers': 4,
                               'embedding': 'angle', 'ansatz': 'strongly_entangling',
                               'best_epoch': ep_vqc8, 'note': '8q 100/class v2'})

# HQNN 8q
print('\n  [HQNN 8q, 3L, h=32] — 100/class')
hqnn8 = HQNNClassifier(n_features=5, n_qubits=8, n_layers=3, n_classes=5,
                        classical_hidden=32, n_classical_layers=2)
hqnn8, ep_hqnn8, t_hqnn8 = train_torch(hqnn8, X_tr, y_tr, X_te, y_te, epochs=150)
m_hqnn8 = evaluate_torch(hqnn8, X_te, y_te, CN)
m_hqnn8.update(best_epoch=ep_hqnn8, train_time=t_hqnn8)
print_result('HQNN 8q/100', m_hqnn8)
results_all['hqnn8_100'] = m_hqnn8
save_torch_v2('hqnn8', hqnn8, {**_COMMON_CFG, 'model': 'HQNN',
                                 'n_qubits': 8, 'n_layers': 3,
                                 'classical_hidden': 32, 'n_classical_layers': 2,
                                 'ansatz': 'strongly_entangling', 'activation': 'relu',
                                 'best_epoch': ep_hqnn8, 'note': '8q 100/class v2'})


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — QKE: Quantum Kernel Estimation
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  EXPERIMENT 3: QUANTUM KERNEL ESTIMATION (QKE)')
print('='*60)
print('  5 qubits, ZZFeatureMap, 2 reps, C=10, max_train=150')
print('  (reusing 100/class split from Exp 1)')

t0 = time.time()
qke = QKEClassifier(n_qubits=5, n_layers=2, embedding='zzfeaturemap',
                    svm_c=10.0, max_train_samples=150)
print('  Computing kernel matrix (this takes a few minutes)...')
qke.fit(X_tr, y_tr, X_te, y_te)
t_qke = time.time() - t0
m_qke = evaluate_sklearn(qke, X_te, y_te, CN)
m_qke['train_time'] = t_qke
print_result(f'QKE ZZFeatureMap (5q, {t_qke:.0f}s)', m_qke)
results_all['qke_zzfm'] = m_qke

# QKE angle embedding for comparison
print('\n  [QKE angle, 5q, 2 reps, C=10]')
t0 = time.time()
qke_angle = QKEClassifier(n_qubits=5, n_layers=2, embedding='angle',
                           svm_c=10.0, max_train_samples=150)
qke_angle.fit(X_tr, y_tr, X_te, y_te)
t_qke_a = time.time() - t0
m_qke_a = evaluate_sklearn(qke_angle, X_te, y_te, CN)
m_qke_a['train_time'] = t_qke_a
print_result(f'QKE Angle (5q, {t_qke_a:.0f}s)', m_qke_a)
results_all['qke_angle'] = m_qke_a

# Classical RBF-SVM baseline for QKE comparison
print('\n  [RBF-SVM C=10 — classical kernel baseline]')
t0 = time.time()
svm_rbf = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced',
              probability=True, random_state=RANDOM_STATE)
svm_rbf.fit(X_tr, y_tr)
t_svm = time.time() - t0
m_svm = evaluate_sklearn(svm_rbf, X_te, y_te, CN)
m_svm['train_time'] = t_svm
print_result(f'RBF-SVM C=10 (classical, {t_svm:.1f}s)', m_svm)
results_all['rbf_svm_100'] = m_svm


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 4 — Confidence Calibration for all models
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  EXPERIMENT 4: CONFIDENCE CALIBRATION')
print('='*60)

calib_summary = []
for name, m in results_all.items():
    proba  = np.array(m['proba'])
    y_true = y_te                     # same test split for all exp1–3

    conf   = proba.max(axis=1)
    preds  = proba.argmax(axis=1)
    correct = (preds == y_true)

    ece_val, rel_x, rel_y = ece(proba, y_true)
    conf_c = float(conf[correct].mean())
    conf_w = float(conf[~correct].mean()) if (~correct).sum() > 0 else float('nan')
    gap    = conf_c - conf_w

    # store back
    results_all[name]['ece']          = ece_val
    results_all[name]['conf_correct'] = conf_c
    results_all[name]['conf_wrong']   = conf_w
    results_all[name]['calib_x']      = rel_x
    results_all[name]['calib_y']      = rel_y

    calib_summary.append({
        'model': name,
        'acc':   f"{m['acc']:.1%}",
        'ece':   f"{ece_val:.4f}",
        'conf_correct': f"{conf_c:.3f}",
        'conf_wrong':   f"{conf_w:.3f}",
        'gap':   f"{gap:.3f}",
    })
    print(f"  {name:20s}  ECE={ece_val:.4f}  "
          f"conf_correct={conf_c:.3f}  conf_wrong={conf_w:.3f}  gap={gap:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('  FINAL SUMMARY')
print('='*60)
header = f"  {'Model':<22} {'Acc':>7} {'F1':>7} {'AUC':>7} {'R70':>6} {'R80':>6} {'ECE':>7}"
print(header)
print('  ' + '-'*65)

order = ['gbt_100', 'rf_100', 'rbf_svm_100',
         'vqc5_100', 'hqnn5_100',
         'vqc8_100', 'hqnn8_100',
         'qke_zzfm', 'qke_angle']

for k in order:
    if k not in results_all:
        continue
    m = results_all[k]
    print(f"  {k:<22} {m['acc']:>6.1%} {m['f1']:>6.1%} {m['auc']:>7.3f} "
          f"{m['recall_70']:>5.1%} {m['recall_80']:>5.1%} "
          f"{m.get('ece', float('nan')):>7.4f}")

# ── Save all results ──────────────────────────────────────────────────────────
for name, m in results_all.items():
    save_json(name, m)

# Save summary CSV
summary_rows = []
for k in order:
    if k not in results_all:
        continue
    m = results_all[k]
    summary_rows.append({
        'model': k,
        'accuracy': round(m['acc'], 4),
        'f1_macro': round(m['f1'], 4),
        'auc': round(m['auc'], 4),
        'recall_70': round(m['recall_70'], 4),
        'recall_80': round(m['recall_80'], 4),
        'ece': round(m.get('ece', float('nan')), 4),
        'conf_correct': round(m.get('conf_correct', float('nan')), 4),
        'conf_wrong': round(m.get('conf_wrong', float('nan')), 4),
        'train_time_s': round(m.get('train_time', 0), 1),
    })
pd.DataFrame(summary_rows).to_csv(
    os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

print(f'\n  All results saved to {RESULTS_DIR}/')
print('  Done.')
