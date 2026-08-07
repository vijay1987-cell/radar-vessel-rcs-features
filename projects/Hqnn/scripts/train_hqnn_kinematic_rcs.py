"""
Train HQNN models (5q and 8q) on combined kinematic+RCS features.

Feature set (9 features):
  RCS: log_peak_rcs, log_total_rcs, rcs_conc, aspect_ratio, footprint_m2
  Kinematic: sog (RSog cap 15m/s), measured_sog_std_900, measured_cog_std_900,
             measured_sog_avg_900

Architecture:
  Classical encoder (9 → n_qubits via dense+Tanh)
  → Quantum circuit (AngleEmbedding + StronglyEntanglingLayers)
  → Classical decoder (n_qubits → 4 classes)

Classes: 30 (Fishing), 52 (Tug), 70 (Cargo), 80 (Tanker)

Training data: 300/class range-stratified sample from main merged CSV

Saves:
  saved_models/hqnn5_kin_weights.pt / hqnn5_kin_config.json
  saved_models/hqnn8_kin_weights.pt / hqnn8_kin_config.json
"""
import sys, os, json, copy, time, warnings

# Cap threads BEFORE importing torch/pennylane.
# Default 20-thread OMP on 32-sample quantum batches causes severe contention.
os.environ['OMP_NUM_THREADS']      = '4'
os.environ['MKL_NUM_THREADS']      = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

torch.set_num_threads(4)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/iaxiom/projects/Hqnn/src')

SAVE = '/home/iaxiom/projects/Hqnn/saved_models'
DATA = '/home/iaxiom/Downloads/radarfeatureL_MERGED_VALID_updated1 1.csv'
KNOWN = {30, 52, 70, 80}
SAMPLES_PER_CLASS = 300


def p(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


# ── Load & feature engineering ────────────────────────────────────────────────
p("=" * 72)
p("LOADING AND PREPARING DATA")
p("=" * 72)

df = pd.read_csv(DATA)
df = df[df['Type'].isin(KNOWN)].copy()
df['Type'] = df['Type'].astype(int)

R  = df['range'].clip(lower=1.0)
Ap = df['PeakAmplitude'].clip(lower=1e-9)
At = df['TotalAmplitude'].clip(lower=1e-9)
er = df['down_range_extent'].clip(lower=0.1)
ec = df['cross_range_extent'].clip(lower=0.1)

df['log_peak_rcs']  = np.log(Ap) + 4 * np.log(R)
df['log_total_rcs'] = np.log(At) + 4 * np.log(R)
df['rcs_conc']      = Ap / At
df['aspect_ratio']  = er / ec
df['footprint_m2']  = np.log(er * ec)
df['sog']           = df['RSog'].clip(upper=15.0)

ALL_FEATS = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'aspect_ratio', 'footprint_m2',
    'sog', 'measured_sog_std_900', 'measured_cog_std_900', 'measured_sog_avg_900',
]
df = df.dropna(subset=ALL_FEATS).copy()
p(f"Usable rows: {len(df)}  |  {df['Type'].value_counts().sort_index().to_dict()}")

# ── Range-stratified sample: 300/class ───────────────────────────────────────
frames = []
for cls in sorted(KNOWN):
    sub = df[df['Type'] == cls].copy()
    sub['_rbin'] = pd.qcut(sub['range'], q=5, labels=False, duplicates='drop')
    sampled = (sub.groupby('_rbin', group_keys=False)
                  .apply(lambda g: g.sample(
                      n=min(len(g), SAMPLES_PER_CLASS // 5), random_state=42)))
    if len(sampled) < SAMPLES_PER_CLASS:
        extra = sub.drop(sampled.index).sample(
            n=SAMPLES_PER_CLASS - len(sampled), random_state=99,
            replace=len(sub) < SAMPLES_PER_CLASS)
        sampled = pd.concat([sampled, extra])
    sampled = sampled.sample(n=min(len(sampled), SAMPLES_PER_CLASS), random_state=42)
    frames.append(sampled)

df_s = pd.concat(frames).reset_index(drop=True)
p(f"\nStratified sample: {len(df_s)} rows  "
  f"{df_s['Type'].value_counts().sort_index().to_dict()}")

# ── Labels & scaling ─────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit([str(c) for c in sorted(KNOWN)])
y = le.transform([str(t) for t in df_s['Type']])
X = df_s[ALL_FEATS].values.astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s  = scaler.transform(X_te)

p(f"Train: {len(X_tr)}  Test: {len(X_te)}")
p(f"Class encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Class weights ─────────────────────────────────────────────────────────────
counts  = np.bincount(y_tr, minlength=4)
weights = 4.0 / (counts + 1e-9)
weights = (weights / weights.sum() * 4).tolist()
w_tensor = torch.tensor(weights, dtype=torch.float32)


# ── Training function ─────────────────────────────────────────────────────────
def train_hqnn(n_qubits, n_layers, label):
    from models.hqnn_model import HQNNClassifier

    model = HQNNClassifier(
        n_features=len(ALL_FEATS),
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=4,
        classical_hidden=64,
        n_classical_layers=2,
        ansatz='strongly_entangling',
        activation='relu',
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-4)

    Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.long)
    Xte_t = torch.tensor(X_te_s, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)

    best_val_acc = 0.0
    best_state   = None
    best_epoch   = 0
    patience_cnt = 0
    PATIENCE     = 20
    MAX_EPOCHS   = 120

    p(f"\n{'='*72}")
    p(f"TRAINING {label}  ({n_qubits}q, {n_layers} layers, "
      f"hidden=64, {len(ALL_FEATS)} features)")
    p(f"{'='*72}")

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(Xtr_t)

        model.eval()
        with torch.no_grad():
            tr_pred  = model(Xtr_t).argmax(1).numpy()
            val_pred = model(Xte_t).argmax(1).numpy()
        tr_acc  = accuracy_score(y_tr, tr_pred)
        val_acc = accuracy_score(y_te, val_pred)

        scheduler.step(val_acc)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            best_epoch   = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            p(f"  Ep {epoch:3d}/{MAX_EPOCHS}  loss={ep_loss:.4f}  "
              f"tr={tr_acc:.3f}  val={val_acc:.3f}  "
              f"[best={best_val_acc:.3f}@ep{best_epoch}]  {elapsed:.0f}s")

        if patience_cnt >= PATIENCE:
            p(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        te_pred = model(Xte_t).argmax(1).numpy()
    te_acc = accuracy_score(y_te, te_pred)
    te_f1  = f1_score(y_te, te_pred, average='macro',    zero_division=0)
    te_f1w = f1_score(y_te, te_pred, average='weighted', zero_division=0)

    p(f"\n  Best epoch {best_epoch}  val_acc={best_val_acc:.4f}")
    p(f"  Test  Acc={te_acc:.4f}  F1-macro={te_f1:.4f}  F1-wtd={te_f1w:.4f}")

    rep = classification_report(y_te, te_pred,
            target_names=[f"Type{c}" for c in sorted(KNOWN)], zero_division=0)
    p(f"\n  Classification report:")
    for line in rep.splitlines():
        p(f"    {line}")

    idx_70 = int(np.where(le.classes_ == '70')[0][0])
    idx_80 = int(np.where(le.classes_ == '80')[0][0])
    cm = y_te == idx_70
    ct = y_te == idx_80
    c_tp  = (te_pred[cm] == idx_70).mean() if cm.sum() > 0 else 0
    c_tan = (te_pred[cm] == idx_80).mean() if cm.sum() > 0 else 0
    t_tp  = (te_pred[ct] == idx_80).mean() if ct.sum() > 0 else 0
    t_car = (te_pred[ct] == idx_70).mean() if ct.sum() > 0 else 0
    p(f"\n  Cargo→Cargo  : {c_tp*100:.1f}%  |  Cargo→Tanker : {c_tan*100:.1f}%")
    p(f"  Tanker→Tanker: {t_tp*100:.1f}%  |  Tanker→Cargo : {t_car*100:.1f}%")

    return model, dict(
        model='HQNN', n_features=len(ALL_FEATS), n_qubits=n_qubits, n_layers=n_layers,
        n_classes=4, classical_hidden=64, n_classical_layers=2,
        ansatz='strongly_entangling', activation='relu',
        feature_cols=ALL_FEATS, class_names=list(le.classes_),
        accuracy=round(te_acc, 4), f1_macro=round(te_f1, 4), f1_weighted=round(te_f1w, 4),
        best_epoch=best_epoch, note=f'HQNN {n_qubits}q kinematic+RCS 300/class',
    )


# ── Quick timing benchmark before full training ───────────────────────────────
p("\nBenchmarking first forward+backward pass times...")
from models.hqnn_model import HQNNClassifier
_x = torch.tensor(X_tr_s[:32], dtype=torch.float32)
_y = torch.tensor(y_tr[:32], dtype=torch.long)
_c = nn.CrossEntropyLoss()

for _q, _l in [(5, 2), (8, 3)]:
    _m = HQNNClassifier(len(ALL_FEATS), _q, _l, 4, 64, 2, 'strongly_entangling', 'relu')
    _opt = torch.optim.Adam(_m.parameters(), lr=0.005)
    t0 = time.time()
    _opt.zero_grad(); _c(_m(_x), _y).backward(); _opt.step()
    elapsed = time.time() - t0
    p(f"  HQNN{_q} ({_l}L): {elapsed:.2f}s/batch  "
      f"→ est. {elapsed*30:.0f}s/epoch × ~50 epochs = {elapsed*30*50/60:.0f} min")

del _m, _opt, _x, _y, _c
p("")


# ── Train both variants ───────────────────────────────────────────────────────
model5, cfg5 = train_hqnn(n_qubits=5, n_layers=2, label='HQNN5-Kin')
model8, cfg8 = train_hqnn(n_qubits=8, n_layers=3, label='HQNN8-Kin')


# ── Save ──────────────────────────────────────────────────────────────────────
p("\n" + "=" * 72)
p("SAVING")
p("=" * 72)

torch.save(model5.state_dict(), f'{SAVE}/hqnn5_kin_weights.pt')
torch.save(model8.state_dict(), f'{SAVE}/hqnn8_kin_weights.pt')
with open(f'{SAVE}/hqnn5_kin_config.json', 'w') as f: json.dump(cfg5, f, indent=2)
with open(f'{SAVE}/hqnn8_kin_config.json', 'w') as f: json.dump(cfg8, f, indent=2)
p("  hqnn5_kin_weights.pt / hqnn5_kin_config.json  saved")
p("  hqnn8_kin_weights.pt / hqnn8_kin_config.json  saved")


# ── Summary ───────────────────────────────────────────────────────────────────
p("\n" + "=" * 72)
p("SUMMARY — Kinematic+RCS HQNN (4-class)")
p("=" * 72)
p(f"{'Model':<25}  {'Acc':>7}  {'F1-mac':>7}  {'Qubits':>7}")
p("-" * 55)
for cfg in [cfg5, cfg8]:
    q = cfg['n_qubits']
    p(f"  {'HQNN'+str(q)+'-Kin':<25}  {cfg['accuracy']:>7.4f}  "
      f"{cfg['f1_macro']:>7.4f}  {q:>7}")
p(f"  {'HQNN5-RCS (baseline)':<25}  {'0.7766':>7}  {'0.7756':>7}  {'5':>7}")
p(f"  {'HQNN8-RCS (baseline)':<25}  {'0.7825':>7}  {'0.7817':>7}  {'8':>7}")
p(f"  {'GBT-Kin (combined)':<25}  {'0.9443':>7}  {'  —':>7}  {'classical':>9}")
p("Done.")
