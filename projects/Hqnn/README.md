# Radar Vessel Type Classifier — QML + Classical ML

A 6-step interactive Streamlit application for training and running inference on
radar-based vessel type classification models. Supports Variational Quantum
Classifiers (VQC), Hybrid Quantum-Classical Neural Networks (HQNN), Gradient
Boosting Trees (GBT), and Random Forest (RF).

---

## What It Does

The app guides you through six steps:

| Step | Description |
|------|-------------|
| 1 — Data | Load a radar CSV and explore class distributions |
| 2 — Preprocessing | Feature engineering, SMOTE balancing, train/test split |
| 3 — Model Config | Choose model type and hyperparameters |
| 4 — Train | Train the selected model with live epoch logging |
| 5 — Results | Accuracy, F1, AUC, per-class recall, confusion matrix |
| 6 — Inference | Run predictions on new data using any saved model, or benchmark all models at once |

---

## Project Structure

```
Hqnn/
├── app.py                  # Main Streamlit application
├── requirements.txt        # All Python dependencies
├── run.sh                  # One-command launcher
├── src/
│   ├── data_processing.py
│   ├── trainer.py
│   ├── visualization.py
│   └── models/
│       ├── vqc_model.py    # Variational Quantum Classifier
│       ├── hqnn_model.py   # Hybrid Quantum-Classical NN
│       ├── classical_model.py
│       ├── ensemble_model.py
│       └── qke_model.py    # Quantum Kernel Estimation
└── saved_models/           # Trained weights and preprocessor (20 MB total)
    ├── preprocessor.pkl    # ← CRITICAL: scaler + label encoder
    ├── *_weights.pt        # PyTorch weights (VQC, HQNN)
    ├── *_model.pkl         # Sklearn models (GBT, RF)
    └── *_config.json       # Model architecture configs
```

---

## Requirements

- **Python 3.12** (exact version recommended — PyTorch and PennyLane are sensitive to version changes)
- 4 GB RAM minimum (8 GB recommended for quantum circuit simulation)
- CPU is sufficient for inference; GPU optional for training

---

## Setup on a New Machine

### Step 1 — Copy the project

Transfer the project folder to the new machine. The minimum required files are:

```
app.py
requirements.txt
run.sh
src/
saved_models/
```

The `experiments/` and `paper/` folders are optional (research artefacts only).

Using `scp` from the source machine:
```bash
scp -r /path/to/Hqnn user@new-machine:/home/user/projects/Hqnn
```

Or zip and copy:
```bash
zip -r Hqnn_project.zip app.py requirements.txt run.sh src/ saved_models/
```

---

### Step 2 — Install Python 3.12

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

**Verify:**
```bash
python3.12 --version
# Expected: Python 3.12.x
```

---

### Step 3 — Create the virtual environment

```bash
cd Hqnn
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

> Installation takes 5–15 minutes depending on internet speed. PyTorch alone is ~2 GB.

---

### Step 4 — Launch the app

```bash
chmod +x run.sh
./run.sh
```

Then open your browser at:
```
http://localhost:8501
```

**If running on a remote server** (e.g. a DGX or cloud VM), bind to all interfaces so you can access it from your local browser:
```bash
.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```
Then open `http://<server-ip>:8501` from your local machine.

---

## Preparing Your Inference Data

The app expects a CSV file with these five columns for inference:

| Column | Unit | Description |
|--------|------|-------------|
| `azimuth` | radians | Bearing from radar to target |
| `PeakAmplitude` | raw | Peak radar return amplitude |
| `TotalAmplitude` | raw | Summed amplitude across all range/azimuth cells |
| `down_range_extent` | metres | Physical range extent of the detection |
| `az_extent_m` | metres | Physical cross-range extent of the detection |

### If your CSV is missing `az_extent_m`

Check what unit `cross_range_extent` is in:

**If `cross_range_extent` is in radians** (small values, ~0.01–0.05):
```python
df['az_extent_m'] = df['range'] * df['cross_range_extent']
```

**If `cross_range_extent` is already in metres** (values in the hundreds):
```python
df['az_extent_m'] = df['cross_range_extent']
```

### If `azimuth` is in degrees

Convert to radians before inference:
```python
import numpy as np
df['azimuth'] = np.deg2rad(df['azimuth'])
```

### Remove zero-value rows

Any row where a feature is zero will produce unreliable predictions. Remove them:
```python
feature_cols = ['azimuth', 'PeakAmplitude', 'TotalAmplitude', 'down_range_extent', 'az_extent_m']
df = df[(df[feature_cols] != 0).all(axis=1)].reset_index(drop=True)
```

---

## Saved Models

The `saved_models/` folder contains all pre-trained models ready for inference:

| Model | File | Accuracy | Notes |
|-------|------|----------|-------|
| HQNN 5q (V1) | `hqnn_v1paper_weights.pt` | 82.3% | 300/class training set |
| VQC 5q (V1) | `vqc_v1paper_weights.pt` | 72.4% | 300/class training set |
| GBT (V1) | `gbt_v1paper_model.pkl` | 90.0% | Best overall accuracy |
| HQNN 5q (V2) | `hqnn5_v2_weights.pt` | 78.0% | 100/class low-data regime |
| HQNN 8q (V2) | `hqnn8_v2_weights.pt` | 76.0% | Best calibration (ECE 0.1019) |
| VQC 5q (V2) | `vqc5_v2_weights.pt` | 57.0% | 100/class low-data regime |
| VQC 8q (V2) | `vqc8_v2_weights.pt` | 65.0% | 100/class, wider circuit |
| GBT (V2) | `gbt_v2_model.pkl` | 80.0% | 100/class low-data regime |
| RF (V2) | `rf_v2_model.pkl` | 84.0% | 100/class low-data regime |

The `preprocessor.pkl` file is shared by all models and contains the fitted
StandardScaler and LabelEncoder. **Do not delete or replace it.**

Vessel classes:
- **30** — Fishing
- **33** — Dredging
- **52** — Tug
- **70** — Cargo
- **80** — Tanker

---

## GPU Support (Optional)

The app runs on CPU by default. If the machine has an NVIDIA GPU and you want
faster quantum circuit simulation during training:

```bash
.venv/bin/pip install pennylane-lightning-gpu
```

> Note: For 5–8 qubit circuits, CPU is often faster than GPU due to kernel
> launch overhead. GPU benefits appear at larger qubit counts (12+).

---

## Troubleshooting

**App won't start — `ModuleNotFoundError`**
```bash
# Make sure you're using the venv, not system Python
.venv/bin/python3 -c "import streamlit, pennylane, torch"
```

**`preprocessor.pkl` not found**
The `saved_models/` folder must be at the same level as `app.py`. Check:
```bash
ls saved_models/preprocessor.pkl
```

**Port 8501 already in use**
```bash
.venv/bin/streamlit run app.py --server.port 8502
```

**Slow inference on quantum models**
VQC and HQNN use `lightning.qubit` (CPU statevector simulation). Inference
on a few hundred rows takes 1–5 minutes. GBT and RF are instantaneous.

---

## Quick Verification

Run this after setup to confirm everything is working:

```bash
.venv/bin/python3 - << 'EOF'
import pennylane, torch, streamlit, sklearn, pandas
print("pennylane :", pennylane.__version__)
print("torch     :", torch.__version__)
print("streamlit :", streamlit.__version__)
print("sklearn   :", sklearn.__version__)
print("pandas    :", pandas.__version__)
import pickle
with open('saved_models/preprocessor.pkl','rb') as f:
    pre = pickle.load(f)
print("preprocessor features:", pre['feature_cols'])
print("\nAll checks passed — ready to launch.")
EOF
```

Expected output:
```
pennylane : 0.45.1
torch     : 2.13.0
streamlit : 1.60.0
sklearn   : 1.9.0
pandas    : 3.0.5
preprocessor features: ['azimuth', 'PeakAmplitude', 'TotalAmplitude', 'down_range_extent', 'az_extent_m']

All checks passed — ready to launch.
```
