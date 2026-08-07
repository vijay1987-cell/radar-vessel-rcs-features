"""
Train combined kinematic + RCS models on main merged dataset.

Feature set (9 features):
  RCS (range-invariant):
    log_peak_rcs   = ln(PeakAmplitude) + 4·ln(range)
    log_total_rcs  = ln(TotalAmplitude) + 4·ln(range)
    rcs_conc       = PeakAmplitude / TotalAmplitude
    aspect_ratio   = down_range_extent / cross_range_extent
    footprint_m2   = ln(down_range_extent × cross_range_extent)

  Kinematic (operational role indicators):
    sog            = RSog capped at 15 m/s
    sog_std_900    = speed std over 900s window (15-min variability)
    cog_std_900    = course std over 900s window (manoeuvring index)
    sog_avg_900    = rolling mean speed over 900s

Classes: 30 (Fishing), 52 (Tug), 70 (Cargo), 80 (Tanker)
         [4-class, same as existing RCS models — class 33 excluded]

Models: GBT-Kin (GradientBoosting) and XGB-Kin (XGBoost)
        5-fold stratified CV + held-out test (20%)

Saves:
  saved_models/gbt_kin_model.pkl
  saved_models/xgb_kin_model.pkl
  saved_models/kin_preprocessor.pkl
  results/combined_kinematic_rcs_results.json
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              cohen_kappa_score)
import xgboost as xgb

warnings.filterwarnings('ignore')

SAVE = '/home/iaxiom/projects/Hqnn/saved_models'
DATA = '/home/iaxiom/Downloads/radarfeatureL_MERGED_VALID_updated1 1.csv'
RESULTS_DIR = '/home/iaxiom/projects/Hqnn/results'
KNOWN = {30, 52, 70, 80}

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 72)
print("LOADING DATA")
print("=" * 72)
df = pd.read_csv(DATA)
print(f"Raw: {len(df)} rows")

# Filter to 4 model classes
df = df[df['Type'].isin(KNOWN)].copy()
df['Type'] = df['Type'].astype(int)
print(f"After class filter: {len(df)} rows  "
      f"  {df['Type'].value_counts().sort_index().to_dict()}")

# ── Compute RCS features ───────────────────────────────────────────────────────
R  = df['range'].clip(lower=1.0)
Ap = df['PeakAmplitude'].clip(lower=1e-9)
At = df['TotalAmplitude'].clip(lower=1e-9)
er = df['down_range_extent'].clip(lower=0.1)
ec = df['cross_range_extent'].clip(lower=0.1)   # already in metres

df['log_peak_rcs']  = np.log(Ap) + 4 * np.log(R)
df['log_total_rcs'] = np.log(At) + 4 * np.log(R)
df['rcs_conc']      = Ap / At
df['aspect_ratio']  = er / ec
df['footprint_m2']  = np.log(er * ec)

# Cap RSog at 15 m/s
df['sog']           = df['RSog'].clip(upper=15.0)

# Kinematic rolling features
KIN_FEATS = ['sog', 'measured_sog_std_900', 'measured_cog_std_900', 'measured_sog_avg_900']
RCS_FEATS = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc', 'aspect_ratio', 'footprint_m2']
ALL_FEATS = RCS_FEATS + KIN_FEATS

# Drop rows with NaN in any feature
df = df.dropna(subset=ALL_FEATS).copy()
print(f"After dropping NaN kinematic rows: {len(df)} rows  "
      f"  {df['Type'].value_counts().sort_index().to_dict()}")

# ── Cohen's d analysis: can kinematic push past RCS ceiling? ─────────────────
print("\n" + "=" * 72)
print("COHEN'S d — Feature discrimination by class pair")
print("=" * 72)

classes = sorted(df['Type'].unique())
class_names_4 = [str(c) for c in classes]
pairs = [(classes[i], classes[j]) for i in range(len(classes))
         for j in range(i+1, len(classes))]
pair_labels = {30: 'Fishing', 52: 'Tug', 70: 'Cargo', 80: 'Tanker'}

print(f"\n{'Pair':<20}  " + "  ".join(f"{f[:12]:>12}" for f in ALL_FEATS))
print("-" * (22 + 15 * len(ALL_FEATS)))

for c1, c2 in pairs:
    g1 = df[df['Type'] == c1]
    g2 = df[df['Type'] == c2]
    ds = []
    for feat in ALL_FEATS:
        a, b = g1[feat].values, g2[feat].values
        n1, n2 = len(a), len(b)
        s_pool = np.sqrt(((n1-1)*a.std()**2 + (n2-1)*b.std()**2) / (n1+n2-2))
        d = abs(a.mean() - b.mean()) / (s_pool + 1e-9)
        ds.append(d)
    label = f"{pair_labels[c1]}-{pair_labels[c2]}"
    print(f"  {label:<20}  " + "  ".join(f"{d:>12.3f}" for d in ds))

# ── Labels ────────────────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit([str(c) for c in sorted(KNOWN)])
y = le.transform([str(t) for t in df['Type']])
X = df[ALL_FEATS].values.astype(np.float32)
print(f"\nClass encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Train/test split ──────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s  = scaler.transform(X_te)
print(f"\nTrain: {len(X_tr)}  Test: {len(X_te)}")

# ── 5-fold CV helper ──────────────────────────────────────────────────────────
def cv_eval(model_cls, params, X, y, n_splits=5, label=''):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        sc = StandardScaler().fit(X[tr_idx])
        Xtr = sc.transform(X[tr_idx]); Xval = sc.transform(X[val_idx])
        m = model_cls(**params).fit(Xtr, y[tr_idx])
        p = m.predict(Xval)
        accs.append(accuracy_score(y[val_idx], p))
        f1s.append(f1_score(y[val_idx], p, average='weighted', zero_division=0))
        print(f"    Fold {fold}/5  acc={accs[-1]:.4f}  f1w={f1s[-1]:.4f}")
    print(f"  [{label}]  CV  Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"F1w={np.mean(f1s):.4f}")
    return np.mean(accs), np.mean(f1s)


# ── GBT-Kin ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("TRAINING GBT-Kin")
print("=" * 72)
gbt_params = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                  subsample=0.8, min_samples_leaf=20, random_state=42)
gbt_cv_acc, gbt_cv_f1 = cv_eval(GradientBoostingClassifier, gbt_params,
                                  X_tr, y_tr, label='GBT-Kin')
gbt = GradientBoostingClassifier(**gbt_params).fit(X_tr_s, y_tr)
gbt_te_pred = gbt.predict(X_te_s)
gbt_te_acc  = accuracy_score(y_te, gbt_te_pred)
gbt_te_f1   = f1_score(y_te, gbt_te_pred, average='weighted', zero_division=0)
print(f"\n  [GBT-Kin]  Test  Acc={gbt_te_acc:.4f}  F1w={gbt_te_f1:.4f}")
print(f"\n  Classification report:")
rep = classification_report(y_te, gbt_te_pred,
        target_names=[f"Type{c}" for c in sorted(KNOWN)], zero_division=0)
for line in rep.splitlines(): print(f"    {line}")


# ── XGB-Kin ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("TRAINING XGB-Kin")
print("=" * 72)
xgb_params = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                   use_label_encoder=False, eval_metric='mlogloss',
                   random_state=42, n_jobs=-1)
xgb_cv_acc, xgb_cv_f1 = cv_eval(xgb.XGBClassifier, xgb_params,
                                   X_tr, y_tr, label='XGB-Kin')
xgb_clf = xgb.XGBClassifier(**xgb_params).fit(X_tr_s, y_tr)
xgb_te_pred = xgb_clf.predict(X_te_s)
xgb_te_acc  = accuracy_score(y_te, xgb_te_pred)
xgb_te_f1   = f1_score(y_te, xgb_te_pred, average='weighted', zero_division=0)
print(f"\n  [XGB-Kin]  Test  Acc={xgb_te_acc:.4f}  F1w={xgb_te_f1:.4f}")
print(f"\n  Classification report:")
rep = classification_report(y_te, xgb_te_pred,
        target_names=[f"Type{c}" for c in sorted(KNOWN)], zero_division=0)
for line in rep.splitlines(): print(f"    {line}")


# ── Feature importances ───────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("FEATURE IMPORTANCES")
print("=" * 72)
gbt_imp = gbt.feature_importances_
xgb_imp = xgb_clf.feature_importances_
print(f"\n{'Feature':<25}  {'GBT-Kin':>10}  {'XGB-Kin':>10}")
print("-" * 50)
for i, f in enumerate(ALL_FEATS):
    print(f"  {f:<25}  {gbt_imp[i]:>10.4f}  {xgb_imp[i]:>10.4f}")


# ── Cargo-Tanker error analysis ───────────────────────────────────────────────
print("\n" + "=" * 72)
print("CARGO-TANKER ERROR ANALYSIS")
print("=" * 72)
idx_70 = int(np.where(le.classes_ == '70')[0][0])
idx_80 = int(np.where(le.classes_ == '80')[0][0])

for mname, preds in [('GBT-Kin', gbt_te_pred), ('XGB-Kin', xgb_te_pred)]:
    cargo_mask  = y_te == idx_70
    tanker_mask = y_te == idx_80
    c_tp  = (preds[cargo_mask]  == idx_70).mean()
    c_tan = (preds[cargo_mask]  == idx_80).mean()
    t_tp  = (preds[tanker_mask] == idx_80).mean()
    t_car = (preds[tanker_mask] == idx_70).mean()
    print(f"\n  [{mname}]")
    print(f"    Cargo→Cargo    : {c_tp*100:.1f}%  |  Cargo→Tanker  : {c_tan*100:.1f}%")
    print(f"    Tanker→Tanker  : {t_tp*100:.1f}%  |  Tanker→Cargo  : {t_car*100:.1f}%")


# ── Save models ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SAVING MODELS")
print("=" * 72)

with open(f'{SAVE}/gbt_kin_model.pkl', 'wb') as f: pickle.dump(gbt, f)
with open(f'{SAVE}/xgb_kin_model.pkl', 'wb') as f: pickle.dump(xgb_clf, f)

kin_pre = dict(scaler=scaler, label_encoder=le,
               class_names=list(le.classes_), features=ALL_FEATS)
with open(f'{SAVE}/kin_preprocessor.pkl', 'wb') as f: pickle.dump(kin_pre, f)

gbt_cfg = dict(n_features=len(ALL_FEATS), n_classes=4, features=ALL_FEATS,
               model_type='GradientBoosting', cv_acc=gbt_cv_acc, cv_f1=gbt_cv_f1,
               test_acc=gbt_te_acc, test_f1=gbt_te_f1, **gbt_params)
xgb_cfg = dict(n_features=len(ALL_FEATS), n_classes=4, features=ALL_FEATS,
               model_type='XGBClassifier', cv_acc=xgb_cv_acc, cv_f1=xgb_cv_f1,
               test_acc=xgb_te_acc, test_f1=xgb_te_f1)

with open(f'{SAVE}/gbt_kin_config.json', 'w') as f: json.dump(gbt_cfg, f, indent=2)
with open(f'{SAVE}/xgb_kin_config.json', 'w') as f: json.dump(xgb_cfg, f, indent=2)

print(f"  gbt_kin_model.pkl        saved")
print(f"  xgb_kin_model.pkl        saved")
print(f"  kin_preprocessor.pkl     saved")
print(f"  gbt_kin_config.json      saved")
print(f"  xgb_kin_config.json      saved")

# ── Comparison with RCS-only models ──────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY — Combined vs RCS-only baseline")
print("=" * 72)
print(f"{'Model':<25}  {'CV Acc':>8}  {'Test Acc':>9}  {'Test F1w':>9}  {'Classes'}")
print("-" * 70)
print(f"  {'GBT-RCS (baseline)':<25}  {'—':>8}  {'0.840':>9}  {'—':>9}  4-class")
print(f"  {'XGB-RCS (baseline)':<25}  {'—':>8}  {'0.848':>9}  {'—':>9}  4-class")
print(f"  {'GBT-Kin (combined)':<25}  {gbt_cv_acc:>8.4f}  {gbt_te_acc:>9.4f}  {gbt_te_f1:>9.4f}  4-class")
print(f"  {'XGB-Kin (combined)':<25}  {xgb_cv_acc:>8.4f}  {xgb_te_acc:>9.4f}  {xgb_te_f1:>9.4f}  4-class")

# Save results JSON
import os
os.makedirs(RESULTS_DIR, exist_ok=True)
results = dict(
    gbt_kin_cv_acc=gbt_cv_acc, gbt_kin_cv_f1=gbt_cv_f1,
    gbt_kin_test_acc=gbt_te_acc, gbt_kin_test_f1=gbt_te_f1,
    xgb_kin_cv_acc=xgb_cv_acc, xgb_kin_cv_f1=xgb_cv_f1,
    xgb_kin_test_acc=xgb_te_acc, xgb_kin_test_f1=xgb_te_f1,
    features=ALL_FEATS, n_classes=4,
    class_names=['30','52','70','80'],
    n_train=len(X_tr), n_test=len(X_te),
)
with open(f'{RESULTS_DIR}/combined_kinematic_rcs_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results → results/combined_kinematic_rcs_results.json")
print("Done.")
