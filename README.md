# radar-vessel-rcs-features

Range-invariant RCS feature engineering and hybrid quantum-classical classification for maritime vessel type recognition from terrestrial coastal radar.

---

## Paper Series

This repository accompanies a five-paper series on radar-based maritime vessel classification:

| # | Title | Status |
|---|-------|--------|
| 1 | Range-Invariant RCS Features for Maritime Vessel Classification | Under review |
| 2 | Classical ML Benchmarks for Radar Vessel Classification | Under review |
| 3 | Hybrid Quantum-Classical Networks for Edge Radar Inference | Under review |
| 4 | The Cargo–Tanker Discrimination Ceiling: A Physics Constraint | Under review |
| 5 | RCS-Invariant Vessel Signatures Across Range | Under review |

---

## Methodology Overview

Five range-invariant features are derived from raw radar amplitude returns using the R⁴ correction from the monostatic radar range equation (Skolnik 2001):

| Feature | Formula | Physical basis |
|---------|---------|----------------|
| `log_peak_rcs` | ln(A_peak) + 4·ln(R) | Dominant scatterer strength |
| `log_total_rcs` | ln(A_total) + 4·ln(R) | Total scattering power |
| `rcs_conc` | A_peak / A_total | Point vs. distributed scatterer |
| `aspect_ratio` | extent_range / extent_az | Vessel orientation proxy |
| `footprint_m2` | ln(extent_range × extent_az) | Coarse vessel size |

These features are provably range-independent: the R⁴ term cancels the propagation loss, making the features equivalent whether measured at 5 km or 50 km.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run synthetic R⁴ correction verification (no data needed)
python synthetic_validation/verify_r4_correction.py

# Train models on your own compatible data
python scripts/train_rcs_models.py --data /path/to/your/data.csv --models-dir ./saved_models
```

---

## Data Format

The training CSV must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `PeakAmplitude` | float | Peak cell amplitude (linear scale) |
| `TotalAmplitude` | float | Sum of all cell amplitudes |
| `range` | float | Slant range in metres |
| `down_range_extent` | float | Detection span in range (metres) |
| `cross_range_extent` | float | Detection span in azimuth (metres) |
| `Type` | int | Vessel class label: 30=Fishing, 52=Tug, 70=Cargo, 80=Tanker |
| `ObjID` | str/int | Track identifier (for track-level analysis) |

**The proprietary radar dataset (`radarfeatureL_Study`) used in the paper series is not included in this repository and cannot be released.**  
The `synthetic_validation/` script runs entirely on synthetic data and requires no proprietary files.

---

## Repository Structure

```
radar-vessel-rcs-features/
├── features/
│   └── rcs_features.py             # Core R⁴ feature engineering module
├── src/
│   ├── data_processing.py
│   ├── trainer.py
│   ├── visualization.py
│   └── models/
│       ├── classical_model.py      # RF, GBT, XGBoost, SVM, MLP wrapper
│       ├── ensemble_model.py
│       ├── hqnn_model.py           # Hybrid Quantum Neural Network (PennyLane)
│       ├── vqc_model.py
│       └── qke_model.py
├── scripts/
│   ├── train_rcs_models.py         # End-to-end training script (argparse)
│   ├── size_group_analysis.py
│   └── generate_figures.py
├── synthetic_validation/
│   └── verify_r4_correction.py     # Standalone R⁴ proof (no data required)
├── configs/
│   ├── gbt_rcs_config.json         # Best classical model config
│   ├── hqnn5_rcs_config.json       # HQNN 5-qubit config
│   └── hqnn8_rcs_config.json       # HQNN 8-qubit config
├── paper/
│   ├── paper1_features.tex
│   ├── paper2_classical_ml.tex
│   ├── paper3_qml_edge.tex
│   ├── paper4_rq4_cargo_tanker.tex
│   ├── paper5_rcs_invariant.tex
│   └── figures/
├── requirements.txt
└── .gitignore
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{kumar2026rcs,
  title   = {Range-Invariant Radar Cross-Section Features for Maritime
             Vessel Type Classification from Terrestrial Coastal Radar},
  author  = {Kumar V, Vijay and C, Mutturaj and V, Mahesh},
  journal = {Under review},
  year    = {2026}
}
```

---

## Affiliation

**AA Enterprises**, Bangalore, India

---

## Licence

MIT — see [LICENSE](LICENSE)
