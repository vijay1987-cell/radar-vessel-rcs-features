"""
Quantum cross-correlation features for Cargo (Type 70) vs Tanker (Type 80)
per radar station.

Hypothesis: At a given station, Cargo detections produce a different quantum
kernel cross-correlation structure than Tanker detections — because the
CNOT/Ansatz entanglement captures feature-feature correlations (e.g. RCS vs
aspect) that differ by vessel structural type, even when marginal amplitudes
overlap (d=0.56 ceiling).

Two feature maps:
  CNOT-ring : AngleEmbedding + ring-closed CNOT chain (nearest-neighbour
              feature entanglement — cheap, interpretable)
  Ansatz    : AngleEmbedding + StronglyEntanglingLayers (all-to-all
              feature entanglement — richer, fixed seed)

Per (stationID, class):
  1. Sample N_SAMPLE detections → compute N×N quantum kernel matrix K
  2. Extract statistics: mean, std, top eigenvalues, effective rank
  3. Compare Cargo vs Tanker across stations → Cohen's d

Per detection (classification):
  For each detection x at station s, compute:
    k_cargo(x) = mean quantum kernel vs Cargo reference set at s
    k_tanker(x) = mean quantum kernel vs Tanker reference set at s
  Classify: Cargo if k_cargo > k_tanker.
  Evaluate 5-fold CV (split by ObjID).
"""
import warnings, time, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
import pennylane as qml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED      = 42
np.random.seed(SEED)
rng       = np.random.default_rng(SEED)

N_QUBITS       = 5
N_SAMPLE       = 60    # per (station, class) for kernel matrix exploration
N_REF          = 40    # reference set per class for detection-level features
N_LAYERS       = 2
N_FOLDS        = 5
MIN_PER_CLASS  = 20    # minimum detections per class per station

FEATS = ['log_peak_rcs', 'log_total_rcs', 'rcs_conc',
         'aspect_ratio', 'footprint_m2']
CLASSES   = {70: 'Cargo', 80: 'Tanker'}
CLS_NAMES = ['Cargo', 'Tanker']


# ── Load and engineer features ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    '/home/iaxiom/Downloads/radarfeatureL_MERGED_VALID_updated1 1.csv',
    usecols=['ObjID','STATIONID','Type','range',
             'PeakAmplitude','TotalAmplitude',
             'down_range_extent','cross_range_extent'],
    low_memory=False)

df = df[df['Type'].isin([70, 80])].copy()
df['Type']      = df['Type'].astype(int)
df['STATIONID'] = df['STATIONID'].astype(int)

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
df = df.dropna(subset=FEATS).reset_index(drop=True)
df['label'] = (df['Type'] == 80).astype(int)   # 0=Cargo, 1=Tanker

scaler = StandardScaler()
df[FEATS] = scaler.fit_transform(df[FEATS])

cnt = df.groupby(['STATIONID','Type']).size().unstack(fill_value=0)
valid_sids = cnt[(cnt.get(70, pd.Series(0, index=cnt.index)) >= MIN_PER_CLASS) &
                 (cnt.get(80, pd.Series(0, index=cnt.index)) >= MIN_PER_CLASS)].index.tolist()

print(f"Detections: {len(df)}  Cargo: {(df['Type']==70).sum()}  Tanker: {(df['Type']==80).sum()}")
print(f"\nPer-station counts:")
print(cnt.to_string())
print(f"\nValid stations (≥{MIN_PER_CLASS} of each class): {valid_sids}")


# ── Quantum kernels ────────────────────────────────────────────────────────────
try:
    dev = qml.device('lightning.qubit', wires=N_QUBITS)
    print("\nUsing lightning.qubit backend")
except Exception:
    dev = qml.device('default.qubit', wires=N_QUBITS)
    print("\nUsing default.qubit backend")

# Fixed Ansatz weights — same seed across runs
ansatz_w = rng.uniform(-np.pi, np.pi,
    qml.StronglyEntanglingLayers.shape(N_LAYERS, N_QUBITS))

def _fmap_cnot(x):
    """CNOT-ring feature map."""
    qml.AngleEmbedding(x * np.pi, wires=range(N_QUBITS), rotation='Y')
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[N_QUBITS - 1, 0])  # close ring

def _fmap_ansatz(x):
    """Ansatz feature map (strongly entangling)."""
    qml.AngleEmbedding(x * np.pi, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(ansatz_w, wires=range(N_QUBITS))

@qml.qnode(dev)
def _qk_cnot(x1, x2):
    _fmap_cnot(x1)
    qml.adjoint(_fmap_cnot)(x2)
    return qml.probs(wires=range(N_QUBITS))

@qml.qnode(dev)
def _qk_ansatz(x1, x2):
    _fmap_ansatz(x1)
    qml.adjoint(_fmap_ansatz)(x2)
    return qml.probs(wires=range(N_QUBITS))

def qkernel_cnot(x1, x2):   return float(_qk_cnot(x1, x2)[0])
def qkernel_ansatz(x1, x2): return float(_qk_ansatz(x1, x2)[0])


def kernel_matrix(X, kfn):
    """Compute full n×n symmetric kernel matrix."""
    n = len(X)
    K = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            v = kfn(X[i], X[j])
            K[i, j] = K[j, i] = v
    return K

def mean_kernel_vs_ref(x, X_ref, kfn):
    """Mean kernel value between a single point and a reference set."""
    return float(np.mean([kfn(x, xr) for xr in X_ref]))

def extract_kstats(K):
    """Summary statistics from a kernel matrix."""
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off  = K[mask]
    ev   = np.sort(np.linalg.eigvalsh(K))[::-1]
    ev_n = ev / (ev.sum() + 1e-12)
    eff_rank = float(np.exp(-np.sum(ev_n * np.log(ev_n + 1e-12))))
    return dict(
        mean   = float(np.mean(off)),
        std    = float(np.std(off)),
        median = float(np.median(off)),
        p75    = float(np.percentile(off, 75)),
        eig1   = float(ev[0]) / n,
        eig2   = float(ev[1]) / n if n > 1 else 0.,
        eig_r  = float(ev[0] / max(ev[1], 1e-9)),
        eff_rank = eff_rank,
    )


# ── Phase 1: per-station kernel matrix exploration ─────────────────────────────
print("\n" + "="*72)
print("PHASE 1 — Per-station kernel matrix statistics")
print("="*72)

phase1 = {}   # sid → cls_name → kernel_type → stats_dict

for sid in valid_sids:
    phase1[sid] = {}
    print(f"\nStation {sid}:")
    for type_id, cls_name in CLASSES.items():
        sub  = df[(df['STATIONID'] == sid) & (df['Type'] == type_id)]
        n    = min(N_SAMPLE, len(sub))
        idx  = rng.choice(len(sub), n, replace=False)
        X    = sub[FEATS].values[idx]

        phase1[sid][cls_name] = {}
        for kname, kfn in [('CNOT', qkernel_cnot), ('Ansatz', qkernel_ansatz)]:
            t0 = time.time()
            K  = kernel_matrix(X, kfn)
            st = extract_kstats(K)
            phase1[sid][cls_name][kname] = st
            print(f"  {cls_name:7s} [{kname:6s}] n={n:3d}  "
                  f"mean={st['mean']:.4f}  std={st['std']:.4f}  "
                  f"eig1={st['eig1']:.3f}  eff_rank={st['eff_rank']:.2f}  "
                  f"[{time.time()-t0:.1f}s]")


# ── Cohen's d on kernel statistics (pooling across stations) ───────────────────
def cohens_d(a, b):
    a = np.array(a); b = np.array(b)
    sp = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1)) /
                 (len(a)+len(b)-2))
    return float(abs(np.mean(a)-np.mean(b))/sp) if sp > 1e-12 else np.nan

stat_keys = ['mean','std','median','p75','eig1','eig2','eig_r','eff_rank']

print("\n" + "="*72)
print("PHASE 1 — Cohen's d: Cargo vs Tanker kernel statistics")
print("(each station = one data point; pooled across valid stations)")
print("="*72)
print(f"{'Statistic':<20}  {'CNOT d':>8}  {'Ansatz d':>10}  note")
print("-"*58)

best_d, best_stat, best_ktype = 0.0, '', ''
for sk in stat_keys:
    for kname in ['CNOT', 'Ansatz']:
        cargo_vals  = [phase1[sid]['Cargo'][kname][sk]  for sid in valid_sids]
        tanker_vals = [phase1[sid]['Tanker'][kname][sk] for sid in valid_sids]
        d = cohens_d(cargo_vals, tanker_vals)
        if not np.isnan(d) and d > best_d:
            best_d, best_stat, best_ktype = d, sk, kname

for sk in stat_keys:
    d_cnot   = cohens_d(
        [phase1[s]['Cargo']['CNOT'][sk]   for s in valid_sids],
        [phase1[s]['Tanker']['CNOT'][sk]  for s in valid_sids])
    d_ans    = cohens_d(
        [phase1[s]['Cargo']['Ansatz'][sk] for s in valid_sids],
        [phase1[s]['Tanker']['Ansatz'][sk] for s in valid_sids])
    mark = ' <<<' if (max(d_cnot, d_ans) > 0.56) else ''
    print(f"  {sk:<18}  {d_cnot:>8.3f}  {d_ans:>10.3f}{mark}")

print(f"\nAmplitude ceiling (cross_range_extent): d = 0.560")
print(f"Best quantum kernel statistic: {best_stat} [{best_ktype}] d = {best_d:.3f}")


# ── Phase 2: detection-level kernel features + classification ──────────────────
print("\n" + "="*72)
print("PHASE 2 — Detection-level quantum kernel features + 5-fold CV")
print("="*72)
print(f"For each detection: k_cargo = mean kernel vs {N_REF} Cargo refs at same station")
print(f"                    k_tanker = mean kernel vs {N_REF} Tanker refs at same station")

# Work on the pooled valid-station subset only
df_v = df[df['STATIONID'].isin(valid_sids)].copy().reset_index(drop=True)
print(f"\nWorking subset: {len(df_v)} detections")

# 5-fold CV split by ObjID (track-level split to avoid leakage)
skf     = StratifiedGroupKFold(n_splits=N_FOLDS)
groups  = df_v['ObjID'].values
labels  = df_v['label'].values

oof_base_preds  = np.full(len(df_v), -1, dtype=int)  # RCS features only
oof_cnot_preds  = np.full(len(df_v), -1, dtype=int)  # quantum kernel only (CNOT)
oof_ans_preds   = np.full(len(df_v), -1, dtype=int)  # quantum kernel only (Ansatz)

# We'll also collect the kernel feature values for Cohen's d
k_cargo_cnot_all  = np.full(len(df_v), np.nan)
k_tanker_cnot_all = np.full(len(df_v), np.nan)

fold_results = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(df_v, labels, groups)):
    print(f"\n  Fold {fold+1}/{N_FOLDS} (train={len(tr_idx)} det, test={len(te_idx)} det)")
    df_tr = df_v.iloc[tr_idx]
    df_te = df_v.iloc[te_idx]

    # Build per-station reference sets from training fold
    ref_sets = {}   # sid → {cls_name → X_ref}
    for sid in valid_sids:
        ref_sets[sid] = {}
        for type_id, cls_name in CLASSES.items():
            sub = df_tr[(df_tr['STATIONID']==sid) & (df_tr['Type']==type_id)]
            if len(sub) == 0:
                ref_sets[sid][cls_name] = None
                continue
            n_r = min(N_REF, len(sub))
            idx = rng.choice(len(sub), n_r, replace=False)
            ref_sets[sid][cls_name] = sub[FEATS].values[idx]

    # Compute kernel features for each test detection
    t0 = time.time()
    for kname, kfn, oof_preds, k_cargo_arr, k_tanker_arr in [
        ('CNOT',   qkernel_cnot,   oof_cnot_preds,  k_cargo_cnot_all,  k_tanker_cnot_all),
        ('Ansatz', qkernel_ansatz, oof_ans_preds,   None,              None),
    ]:
        for di, row_i in enumerate(te_idx):
            row   = df_te.iloc[di]
            sid   = int(row['STATIONID'])
            x     = row[FEATS].values

            rc = ref_sets[sid].get('Cargo')
            rt = ref_sets[sid].get('Tanker')

            if rc is None or rt is None:
                continue   # station not usable in this fold

            kc = mean_kernel_vs_ref(x, rc, kfn)
            kt = mean_kernel_vs_ref(x, rt, kfn)

            oof_preds[row_i] = 0 if kc >= kt else 1   # 0=Cargo, 1=Tanker

            if k_cargo_arr is not None:
                k_cargo_arr[row_i]  = kc
                k_tanker_arr[row_i] = kt

        elapsed = time.time() - t0
        valid   = oof_preds[te_idx] >= 0
        acc     = accuracy_score(labels[te_idx][valid], oof_preds[te_idx][valid])
        f1      = f1_score(labels[te_idx][valid], oof_preds[te_idx][valid],
                           average='macro', zero_division=0)
        print(f"    [{kname:6s}]  acc={acc:.3f}  f1={f1:.3f}  [{elapsed:.1f}s]")
        t0 = time.time()

    fold_results.append(fold)

# ── Phase 2 results ────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("PHASE 2 — OOF results")
print("="*72)

valid_mask = oof_cnot_preds >= 0
for tag, preds in [('CNOT kernel classify', oof_cnot_preds),
                   ('Ansatz kernel classify', oof_ans_preds)]:
    m = preds >= 0
    acc = accuracy_score(labels[m], preds[m])
    f1  = f1_score(labels[m], preds[m], average='macro', zero_division=0)
    print(f"  {tag:<30}  OOF Acc={acc:.3f}  F1={f1:.3f}")

# Cohen's d on the kernel features themselves
k_diff = k_cargo_cnot_all - k_tanker_cnot_all
valid_kd = ~np.isnan(k_diff)
cargo_diff  = k_diff[valid_kd & (labels == 0)]
tanker_diff = k_diff[valid_kd & (labels == 1)]
d_kd = cohens_d(cargo_diff, tanker_diff)
print(f"\n  Cohen's d on (k_cargo - k_tanker) feature [CNOT]: {d_kd:.3f}")
print(f"  (Cargo: mean_diff={np.mean(cargo_diff):.4f}  "
      f"Tanker: mean_diff={np.mean(tanker_diff):.4f})")
print(f"\n  Amplitude ceiling (cross_range_extent): d = 0.560")


# ── Classification report ──────────────────────────────────────────────────────
m = oof_cnot_preds >= 0
print(f"\nCNOT kernel classifier OOF report:")
print(classification_report(labels[m], oof_cnot_preds[m],
      target_names=CLS_NAMES, zero_division=0))

m = oof_ans_preds >= 0
print(f"Ansatz kernel classifier OOF report:")
print(classification_report(labels[m], oof_ans_preds[m],
      target_names=CLS_NAMES, zero_division=0))


# ── Save results ───────────────────────────────────────────────────────────────
results_out = {
    'valid_stations': valid_sids,
    'n_sample_kernel_matrix': N_SAMPLE,
    'n_ref_detection': N_REF,
    'best_station_kernel_d': best_d,
    'best_station_kernel_stat': best_stat,
    'best_station_kernel_type': best_ktype,
    'cnot_oof_acc': float(accuracy_score(labels[oof_cnot_preds>=0],
                                         oof_cnot_preds[oof_cnot_preds>=0])),
    'ansatz_oof_acc': float(accuracy_score(labels[oof_ans_preds>=0],
                                           oof_ans_preds[oof_ans_preds>=0])),
    'cnot_kernel_feature_cargo_tanker_d': float(d_kd),
    'amplitude_ceiling_d': 0.56,
}
with open('/tmp/quantum_xcorr_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("\nResults saved → /tmp/quantum_xcorr_results.json")


# ── Figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# (a) Per-station mean kernel: Cargo vs Tanker (CNOT)
sids_str = [str(s) for s in valid_sids]
cargo_means_cnot  = [phase1[s]['Cargo']['CNOT']['mean']  for s in valid_sids]
tanker_means_cnot = [phase1[s]['Tanker']['CNOT']['mean'] for s in valid_sids]
x = np.arange(len(valid_sids)); w = 0.35
axes[0,0].bar(x-w/2, cargo_means_cnot,  w, label='Cargo',  color='#e66101', alpha=0.8)
axes[0,0].bar(x+w/2, tanker_means_cnot, w, label='Tanker', color='#5e3c99', alpha=0.8)
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels([f'Sta {s}' for s in valid_sids], fontsize=7)
axes[0,0].set_ylabel('Mean within-class kernel')
axes[0,0].set_title('(a) CNOT: mean kernel per station')
axes[0,0].legend(fontsize=8)

# (b) Per-station mean kernel: Cargo vs Tanker (Ansatz)
cargo_means_ans  = [phase1[s]['Cargo']['Ansatz']['mean']  for s in valid_sids]
tanker_means_ans = [phase1[s]['Tanker']['Ansatz']['mean'] for s in valid_sids]
axes[0,1].bar(x-w/2, cargo_means_ans,  w, label='Cargo',  color='#e66101', alpha=0.8)
axes[0,1].bar(x+w/2, tanker_means_ans, w, label='Tanker', color='#5e3c99', alpha=0.8)
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels([f'Sta {s}' for s in valid_sids], fontsize=7)
axes[0,1].set_ylabel('Mean within-class kernel')
axes[0,1].set_title('(b) Ansatz: mean kernel per station')
axes[0,1].legend(fontsize=8)

# (c) Effective rank per station
cargo_er_cnot  = [phase1[s]['Cargo']['CNOT']['eff_rank']  for s in valid_sids]
tanker_er_cnot = [phase1[s]['Tanker']['CNOT']['eff_rank'] for s in valid_sids]
axes[0,2].plot(sids_str, cargo_er_cnot,  'o-', color='#e66101', label='Cargo')
axes[0,2].plot(sids_str, tanker_er_cnot, 's-', color='#5e3c99', label='Tanker')
axes[0,2].set_xlabel('Station ID')
axes[0,2].set_ylabel('Effective rank (CNOT kernel)')
axes[0,2].set_title('(c) CNOT: effective rank per station')
axes[0,2].legend(fontsize=8)

# (d) Cohen's d per kernel statistic
d_cnot_all   = [cohens_d([phase1[s]['Cargo']['CNOT'][sk]   for s in valid_sids],
                          [phase1[s]['Tanker']['CNOT'][sk]  for s in valid_sids])
                for sk in stat_keys]
d_ansatz_all = [cohens_d([phase1[s]['Cargo']['Ansatz'][sk] for s in valid_sids],
                          [phase1[s]['Tanker']['Ansatz'][sk] for s in valid_sids])
                for sk in stat_keys]
x2 = np.arange(len(stat_keys))
axes[1,0].bar(x2-w/2, d_cnot_all,   w, label='CNOT',   color='#2166ac', alpha=0.8)
axes[1,0].bar(x2+w/2, d_ansatz_all, w, label='Ansatz', color='#d6604d', alpha=0.8)
axes[1,0].axhline(0.56, color='black', ls='--', lw=1.2, label='Amplitude ceiling')
axes[1,0].set_xticks(x2); axes[1,0].set_xticklabels(stat_keys, rotation=30, ha='right', fontsize=7)
axes[1,0].set_ylabel("Cohen's d (Cargo vs Tanker)")
axes[1,0].set_title("(d) Cohen's d per kernel statistic")
axes[1,0].legend(fontsize=7)

# (e) Distribution of k_cargo - k_tanker by class [CNOT]
valid_kd2 = ~np.isnan(k_diff)
for lbl, col, name in [(0,'#e66101','Cargo'), (1,'#5e3c99','Tanker')]:
    vals = k_diff[valid_kd2 & (labels == lbl)]
    axes[1,1].hist(vals, bins=30, alpha=0.6, density=True, color=col, label=name)
axes[1,1].axvline(0, color='black', ls='--', lw=1.0)
axes[1,1].set_xlabel('k_cargo − k_tanker  (CNOT kernel)')
axes[1,1].set_title(f'(e) Kernel differential  (d={d_kd:.3f})')
axes[1,1].legend(fontsize=8)

# (f) Per-station accuracy: CNOT vs Ansatz classifiers
cnot_per_sid, ans_per_sid = [], []
for sid in valid_sids:
    sid_mask = df_v['STATIONID'].values == sid
    m_cnot = sid_mask & (oof_cnot_preds >= 0)
    m_ans  = sid_mask & (oof_ans_preds  >= 0)
    cnot_per_sid.append(accuracy_score(labels[m_cnot], oof_cnot_preds[m_cnot])
                        if m_cnot.sum() > 0 else np.nan)
    ans_per_sid.append(accuracy_score(labels[m_ans], oof_ans_preds[m_ans])
                       if m_ans.sum() > 0 else np.nan)

axes[1,2].plot(sids_str, cnot_per_sid, 'o-', color='#2166ac', label='CNOT')
axes[1,2].plot(sids_str, ans_per_sid,  's-', color='#d6604d', label='Ansatz')
axes[1,2].axhline(0.5, color='black', ls='--', lw=1.0, label='Chance')
axes[1,2].set_ylim(0, 1); axes[1,2].set_xlabel('Station ID')
axes[1,2].set_ylabel('OOF Accuracy'); axes[1,2].set_title('(f) Per-station accuracy')
axes[1,2].legend(fontsize=8)

fig.tight_layout(pad=0.8)
fig_path = ('/home/iaxiom/projects/Hqnn/paper/submission_ready/'
            'figures/fig_quantum_xcorr.pdf')
fig.savefig(fig_path, bbox_inches='tight', dpi=150)
print(f"Figure saved → {fig_path}")
