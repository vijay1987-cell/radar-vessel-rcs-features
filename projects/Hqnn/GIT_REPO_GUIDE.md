# Git Repository Setup Guide
## radar-vessel-rcs-features — AA Enterprises

> **Purpose of this file:** When the user asks to create the GitHub repository,
> follow this guide exactly. It captures the full repo structure, which files
> to include, what to exclude, and the exact README and requirements content.

---

## 1. Repository Identity

| Field | Value |
|---|---|
| **Repo name** | `radar-vessel-rcs-features` |
| **GitHub organisation / user** | `aa-enterprises` (create if not exists) |
| **Visibility** | Public (methodology only — no data) |
| **License** | MIT |
| **Description** | Range-invariant RCS feature engineering and hybrid quantum-classical classification for maritime vessel type recognition from terrestrial radar |
| **Topics / tags** | `radar`, `vessel-classification`, `rcs`, `feature-engineering`, `quantum-machine-learning`, `hqnn`, `pennylane`, `maritime-surveillance` |

---

## 2. Final Repo URL

Once created, replace this placeholder in all 5 papers in `submission_ready/`:
```
https://github.com/aa-enterprises/radar-vessel-rcs-features
```
Search for `github.com/aa-enterprises` in all `.tex` files and update.
Also update the Data Availability section in each paper's `.tex` file.

---

## 3. Directory Structure to Create

```
radar-vessel-rcs-features/
├── README.md                        ← full project README (content in §6 below)
├── LICENSE                          ← MIT licence text
├── requirements.txt                 ← Python dependencies (§7)
├── setup.py                         ← optional package install
│
├── features/
│   └── rcs_features.py             ← COPY from: create new (§4a)
│
├── src/
│   ├── __init__.py                 ← COPY from: /home/iaxiom/projects/Hqnn/src/__init__.py
│   ├── data_processing.py          ← COPY from: /home/iaxiom/projects/Hqnn/src/data_processing.py
│   ├── trainer.py                  ← COPY from: /home/iaxiom/projects/Hqnn/src/trainer.py
│   ├── visualization.py            ← COPY from: /home/iaxiom/projects/Hqnn/src/visualization.py
│   └── models/
│       ├── __init__.py             ← COPY from: /home/iaxiom/projects/Hqnn/src/models/__init__.py
│       ├── classical_model.py      ← COPY from: /home/iaxiom/projects/Hqnn/src/models/classical_model.py
│       ├── ensemble_model.py       ← COPY from: /home/iaxiom/projects/Hqnn/src/models/ensemble_model.py
│       ├── hqnn_model.py           ← COPY from: /home/iaxiom/projects/Hqnn/src/models/hqnn_model.py
│       ├── vqc_model.py            ← COPY from: /home/iaxiom/projects/Hqnn/src/models/vqc_model.py
│       └── qke_model.py            ← COPY from: /home/iaxiom/projects/Hqnn/src/models/qke_model.py
│
├── scripts/
│   ├── train_rcs_models.py         ← COPY from: /tmp/train_hqnn_rcs.py  (strip data path — §4b)
│   ├── size_group_analysis.py      ← COPY from: /tmp/size_group_analysis.py
│   └── generate_figures.py         ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/generate_figures.py
│
├── synthetic_validation/
│   └── verify_r4_correction.py     ← CREATE new (§4c)
│
├── configs/
│   ├── gbt_rcs_config.json         ← COPY from: /home/iaxiom/projects/Hqnn/saved_models/gbt_rcs_config.json
│   ├── hqnn5_rcs_config.json       ← COPY from: /home/iaxiom/projects/Hqnn/saved_models/hqnn5_rcs_config.json
│   └── hqnn8_rcs_config.json       ← COPY from: /home/iaxiom/projects/Hqnn/saved_models/hqnn8_rcs_config.json
│
├── paper/                           ← LaTeX source for all 5 papers
│   ├── paper1_features.tex         ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/paper1_features.tex
│   ├── paper2_classical_ml.tex     ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/paper2_classical_ml.tex
│   ├── paper3_qml_edge.tex         ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/paper3_qml_edge.tex
│   ├── paper4_rq4_cargo_tanker.tex ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/paper4_rq4_cargo_tanker.tex
│   ├── paper5_rcs_invariant.tex    ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/paper5_rcs_invariant.tex
│   └── figures/                    ← COPY from: /home/iaxiom/projects/Hqnn/paper/submission_ready/figures/
│
└── .gitignore                       ← content in §5
```

---

## 4. Files to Create (Not Copied)

### 4a — `features/rcs_features.py`
Core feature engineering module. The heart of the methodology.
Create this as a clean, documented standalone function:

```python
"""
Range-invariant RCS feature engineering for maritime vessel classification.
Derived from the monostatic radar range equation (Skolnik 2001).

Input columns required:  PeakAmplitude, TotalAmplitude, range,
                         down_range_extent, cross_range_extent
Output columns:          log_peak_rcs, log_total_rcs, rcs_conc,
                         aspect_ratio, footprint_m2
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    'log_peak_rcs', 'log_total_rcs', 'rcs_conc',
    'aspect_ratio', 'footprint_m2'
]

def compute_rcs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives five range-invariant RCS features from raw radar observables.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: PeakAmplitude, TotalAmplitude, range (metres),
        down_range_extent (metres), cross_range_extent (metres).

    Returns
    -------
    pd.DataFrame with five new columns appended (FEATURE_COLS).
    """
    out = df.copy()
    R = df['range'].clip(lower=1.0)  # avoid log(0)
    Ap = df['PeakAmplitude'].clip(lower=1e-9)
    At = df['TotalAmplitude'].clip(lower=1e-9)
    er = df['down_range_extent'].clip(lower=0.1)
    ec = df['cross_range_extent'].clip(lower=0.1)

    out['log_peak_rcs']  = np.log(Ap) + 4 * np.log(R)   # Feature 1
    out['log_total_rcs'] = np.log(At) + 4 * np.log(R)   # Feature 2
    out['rcs_conc']      = Ap / At                        # Feature 3 (R^4 cancels)
    out['aspect_ratio']  = er / ec                        # Feature 4
    out['footprint_m2']  = np.log(er * ec)               # Feature 5

    return out
```

### 4b — `scripts/train_rcs_models.py`
Copy `/tmp/train_hqnn_rcs.py` but:
- Replace the hardcoded `/home/iaxiom/Downloads/rcs_train_4class.csv` path
  with a `--data` CLI argument using `argparse`
- Replace the hardcoded `MODELS_DIR` with a `--models-dir` argument
- Add a `if __name__ == '__main__':` guard
- Remove any absolute paths to the machine

### 4c — `synthetic_validation/verify_r4_correction.py`
Standalone proof that R⁴ correction works. Already written as prose in
Paper 5 §6.3 — turn it into runnable Python:

```python
"""
Synthetic verification that log(Ap) + 4*log(R) eliminates range dependence.
Reproduces the result stated in Paper 5, Section VI-C.
No proprietary data required.
"""
import numpy as np
from scipy.stats import pearsonr

RNG = np.random.default_rng(42)
N = 2000

range_m   = RNG.uniform(5e3, 50e3, N)
true_rcs  = RNG.lognormal(mean=2.0, sigma=1.0, size=N)   # range-independent
raw_amp   = true_rcs / range_m**4                          # simulate R^-4 loss

r_raw, p_raw = pearsonr(np.log(raw_amp), np.log(range_m))
log_rcs       = np.log(raw_amp) + 4 * np.log(range_m)
r_corr, p_corr = pearsonr(log_rcs, np.log(range_m))

print(f"Before correction: r = {r_raw:.4f}  (p = {p_raw:.2e})")
print(f"After  correction: r = {r_corr:.4f}  (p = {p_corr:.2f})")
assert abs(r_corr) < 0.05, "Correction failed — check implementation"
print("PASS: range dependence eliminated.")
```

---

## 5. `.gitignore` Content

```
# Data — never commit proprietary radar data
*.csv
*.parquet
*.h5
*.hdf5
data/

# Trained weights — large binary files
*.pt
*.pkl
*.joblib

# LaTeX build artifacts
*.aux *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.env

# OS
.DS_Store
Thumbs.db
```

**IMPORTANT:** Model weight files (`.pt`, `.pkl`) are excluded.
The `configs/` JSON files (no secrets, no data) ARE committed.
Users can retrain from scratch using their own compatible data.

---

## 6. README.md Content (write this as the repo README)

The README should have these sections in order:

1. **Title + one-line description**
2. **Paper series** — link to each of the 5 papers once published (placeholder DOIs for now)
3. **Methodology overview** — the 5 RCS features with the equations (copy from Paper 5 abstract)
4. **Quick start** — install + run synthetic validation (3 commands)
5. **Data format** — the CSV column spec (PeakAmplitude, TotalAmplitude, range, down_range_extent, cross_range_extent); state dataset is proprietary and not included
6. **Repository structure** — directory tree
7. **Citation** — BibTeX entry for Paper 5 (placeholder until DOI assigned)
8. **Affiliation** — AA Enterprises, Bangalore, India
9. **Licence** — MIT

---

## 7. `requirements.txt` Content

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
torch>=2.0
pennylane>=0.35
pennylane-lightning>=0.35
xgboost>=2.0
```

---

## 8. Git Commands to Run (in order)

```bash
# 1. Create and enter repo directory
mkdir -p /home/iaxiom/projects/radar-vessel-rcs-features
cd /home/iaxiom/projects/radar-vessel-rcs-features
git init
git branch -M main

# 2. Create all directories
mkdir -p features src/models scripts synthetic_validation configs paper/figures

# 3. Copy source files (see §3 for full list)

# 4. Create new files (see §4 for content)

# 5. Stage and commit
git add .
git commit -m "Initial release: RCS feature engineering + HQNN vessel classification

Range-invariant radar cross section features for maritime vessel type
classification from terrestrial radar. Includes classical and hybrid
quantum-classical (HQNN) model training code and all 5 paper LaTeX sources.
Dataset not included (proprietary). Synthetic validation included.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

# 6. Add remote and push
git remote add origin https://github.com/aa-enterprises/radar-vessel-rcs-features.git
git push -u origin main
```

---

## 9. After the Repo is Created

Update these locations with the real URL:

| File | Line to find | Replace with |
|---|---|---|
| `paper/submission_ready/paper5_rcs_invariant.tex` | `github.com/aa-enterprises/...` (appears twice) | real URL |
| `paper/submission_ready/paper1_features.tex` | Data Availability section | real URL |
| `paper/submission_ready/paper2_classical_ml.tex` | Data Availability section | real URL |
| `paper/submission_ready/paper3_qml_edge.tex` | Data Availability section | real URL |
| `paper/submission_ready/paper4_rq4_cargo_tanker.tex` | Data Availability section | real URL |

After updating, recompile all 5 PDFs:
```bash
cd /home/iaxiom/projects/Hqnn/paper/submission_ready
for p in paper1_features paper2_classical_ml paper3_qml_edge paper4_rq4_cargo_tanker paper5_rcs_invariant; do
  pdflatex -interaction=nonstopmode "$p.tex"
done
```

---

## 10. Key Facts to Remember

- **Dataset name:** `radarfeatureL_Study` — 100,430 observations, X-band coastal radar
- **4 trained vessel types:** Fishing (30), Tug (52), Cargo (70), Tanker (80)
- **3 OOD types:** Dredging (33), Passenger (60), Other (90)
- **5 RCS features:** log_peak_rcs, log_total_rcs, rcs_conc, aspect_ratio, footprint_m2
- **Best accuracy model:** GBT_RCS — 86.5% [84.3, 88.8%]
- **Best calibration model:** HQNN-8q_RCS — ECE 0.0236
- **Saved model configs:** `/home/iaxiom/projects/Hqnn/saved_models/` (JSON files safe to commit)
- **Saved model weights:** `.pt` and `.pkl` files — do NOT commit to public repo
- **Training data CSVs:** `/home/iaxiom/Downloads/rcs_train_4class.csv` — do NOT commit
- **Figure generation:** `/home/iaxiom/projects/Hqnn/paper/submission_ready/generate_figures.py`
- **Paper source:** `/home/iaxiom/projects/Hqnn/paper/submission_ready/`
- **Company:** AA Enterprises, Bangalore, India
- **Authors:** Vijay Kumar V, Mutturaj C, Mahesh V
