import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time


# ── Step navigation helpers ─────────────────────────────────────

STEPS = [
    "📊 Data",
    "🔧 Preprocessing",
    "🧠 Model Config",
    "🚀 Train",
    "📈 Results",
    "🔮 Inference",
]


def _tab_footer(next_index: int, next_label: str, is_ready: bool,
                ready_msg: str, not_ready_msg: str, key: str):
    st.markdown("---")
    col_msg, col_btn = st.columns([4, 1])
    with col_msg:
        if is_ready:
            st.success(f"✅  {ready_msg}")
        else:
            st.warning(f"⚠️  {not_ready_msg}")
    with col_btn:
        if is_ready:
            if st.button(f"Next →", type="primary", key=key, use_container_width=True):
                st.session_state.current_step = next_index
                st.rerun()

import pickle
import json
import pathlib

from src.data_processing import DataProcessor
from src.trainer import TorchTrainer, evaluate_model
from src.visualization import (
    plot_class_distribution, plot_training_curves, plot_confusion_matrix,
    plot_roc_curves, plot_model_comparison, plot_feature_importance,
)

st.set_page_config(
    page_title="Radar Target QML Classifier",
    layout="wide",
    page_icon="📡",
)

# ---------- model persistence ----------

_MODELS_DIR = pathlib.Path(__file__).parent / 'saved_models'
_MODELS_DIR.mkdir(exist_ok=True)


def _save_model(name: str, model_data: dict):
    """Save model metadata + the actual model object so inference survives restarts."""
    try:
        # 1. Metadata pkl (metrics, history, model_type) — same as before
        saveable = {k: v for k, v in model_data.items() if k != 'model'}
        with open(_MODELS_DIR / f'{name}.pkl', 'wb') as f:
            pickle.dump(saveable, f)

        model_obj = model_data.get('model')
        if model_obj is None:
            return

        is_torch = model_data.get('is_torch', False)
        feature_cols = (
            st.session_state.get('feature_cols') or
            st.session_state.get('processed_data', {}).get('feature_names') or []
        )
        class_names = (
            (st.session_state.get('processed_data') or {}).get('class_names') or []
        )
        metrics = model_data.get('metrics', {})
        cfg_type = model_data.get('model_type', 'Unknown')

        # Derive short model key (e.g. "HQNN", "GBT", "XGB")
        short_type = cfg_type.replace('Classical — ', '').replace('HQNN — ', 'HQNN').replace('VQC — ', 'VQC')
        if 'Gradient Boosting' in short_type:
            short_type = 'GBT'
        elif 'Random Forest' in short_type:
            short_type = 'RF'
        elif 'XGBoost' in short_type:
            short_type = 'XGB'

        # 2. Config json
        cfg = {
            'model': short_type,
            'model_type': cfg_type,
            'n_features': len(feature_cols),
            'n_classes': len(class_names),
            'feature_cols': feature_cols,
            'class_names': class_names,
            'accuracy': float(metrics.get('accuracy', 0)),
            'f1_macro': float(metrics.get('f1_macro', 0)),
            'f1_weighted': float(metrics.get('f1_weighted', 0)),
        }
        if is_torch:
            # PyTorch: save state dict
            m = model_obj.model if hasattr(model_obj, 'model') else model_obj
            if hasattr(m, 'state_dict'):
                torch.save(m.state_dict(), _MODELS_DIR / f'{name}_weights.pt')
                cfg['n_qubits']          = getattr(m, 'n_qubits', 0)
                cfg['n_layers']          = getattr(m, 'n_layers', 0)
                cfg['classical_hidden']  = getattr(m, 'classical_hidden', 16)
                cfg['n_classical_layers']= getattr(m, 'n_classical_layers', 1)
                cfg['ansatz']            = getattr(m, 'ansatz', 'strongly_entangling')
                cfg['activation']        = getattr(m, 'activation', 'relu')
                cfg['embedding']         = getattr(m, 'embedding', 'angle')
        else:
            # sklearn / XGBoost: save model object
            inner = model_obj.model if hasattr(model_obj, 'model') else model_obj
            with open(_MODELS_DIR / f'{name}_model.pkl', 'wb') as f:
                pickle.dump(inner, f)

        with open(_MODELS_DIR / f'{name}_config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

    except Exception as e:
        st.warning(f"Could not save model '{name}' to disk: {e}")


def _load_all_models() -> dict:
    models = {}

    # 1. App-native metadata pkls (have 'metrics' key)
    for p in sorted(_MODELS_DIR.glob('*.pkl')):
        try:
            with open(p, 'rb') as f:
                data = pickle.load(f)
            if 'metrics' not in data:
                continue
            data['model'] = None
            models[p.stem] = data
        except Exception:
            pass

    # 2. Pre-trained models saved as *_config.json + *_model.pkl / *_weights.pt
    for cfg_path in sorted(_MODELS_DIR.glob('*_config.json')):
        name = cfg_path.stem.replace('_config', '')
        if name in models:
            continue  # already loaded from app-native pkl
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            acc = float(cfg.get('accuracy', 0))
            f1m = float(cfg.get('f1_macro', 0))
            f1w = float(cfg.get('f1_weighted', f1m))
            if acc == 0 and f1m == 0:
                continue  # skip placeholder configs with no metrics
            short = cfg.get('model', 'Unknown')
            display_type = cfg.get('model_type') or f"Classical — {short}" if short in ('GBT','RF','XGB','SVM','MLP') else short
            models[name] = {
                'model': None,
                'metrics': {
                    'accuracy': acc,
                    'f1_macro': f1m,
                    'f1_weighted': f1w,
                    'auc_macro': None,
                    'confusion_matrix': None,
                    'probabilities': None,
                    'classification_report': {},
                },
                'history': {},
                'model_type': display_type,
                'is_torch': short in ('VQC', 'HQNN'),
                'from_config': True,
            }
        except Exception:
            pass

    return models


def _delete_model(name: str):
    p = _MODELS_DIR / f'{name}.pkl'
    if p.exists():
        p.unlink()


# ---------- inference helpers ----------

_PREPROCESSOR_REGISTRY: dict | None = None   # cache: feature_cols_tuple → preprocessor dict

def _load_preprocessor_for(name: str) -> dict:
    """Return the right preprocessor for a saved model.

    Models whose config.feature_cols differ from the main preprocessor.pkl
    (e.g. RCS-feature models) have their own <prefix>_preprocessor.pkl saved
    alongside in saved_models/.  This function finds the matching one.
    """
    global _PREPROCESSOR_REGISTRY
    # Build registry lazily
    if _PREPROCESSOR_REGISTRY is None:
        _PREPROCESSOR_REGISTRY = {}
        for pkl in _MODELS_DIR.glob('*preprocessor.pkl'):
            with open(pkl, 'rb') as f:
                pre = pickle.load(f)
            key = tuple(pre.get('feature_cols', []))
            _PREPROCESSOR_REGISTRY[key] = pre

    # Read model's declared feature_cols from its config
    cfg_path = _MODELS_DIR / f'{name}_config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        feat_key = tuple(cfg.get('feature_cols') or [])
        if feat_key in _PREPROCESSOR_REGISTRY:
            return _PREPROCESSOR_REGISTRY[feat_key]

    # Fallback to main preprocessor
    main = _MODELS_DIR / 'preprocessor.pkl'
    with open(main, 'rb') as f:
        return pickle.load(f)


def _get_saved_weight_models() -> list:
    """Return names of all persistently saved models (torch weights or sklearn pkl)."""
    names = []
    for wp in _MODELS_DIR.glob('*_weights.pt'):
        name = wp.stem.replace('_weights', '')
        if (_MODELS_DIR / f'{name}_config.json').exists():
            names.append(name)
    # Also include sklearn models saved as <name>_model.pkl (e.g. gbt_model.pkl)
    for mp in _MODELS_DIR.glob('*_model.pkl'):
        name = mp.stem.replace('_model', '')
        if (_MODELS_DIR / f'{name}_config.json').exists():
            names.append(name)
    return sorted(set(names))


def _load_weights_model(name: str):
    """Reconstruct a model from saved weights/pkl + config. Returns (model, cfg)."""
    with open(_MODELS_DIR / f'{name}_config.json') as f:
        cfg = json.load(f)
    if cfg['model'] == 'VQC':
        from src.models.vqc_model import VQCClassifier
        model = VQCClassifier(
            n_features=cfg['n_features'], n_qubits=cfg['n_qubits'],
            n_layers=cfg['n_layers'],     n_classes=cfg['n_classes'],
            embedding=cfg['embedding'],   ansatz=cfg['ansatz'],
        )
        model.load_state_dict(
            torch.load(_MODELS_DIR / f'{name}_weights.pt', weights_only=True)
        )
        model.eval()
    elif cfg['model'] == 'HQNN':
        from src.models.hqnn_model import HQNNClassifier
        model = HQNNClassifier(
            n_features=cfg['n_features'],         n_qubits=cfg['n_qubits'],
            n_layers=cfg['n_layers'],             n_classes=cfg['n_classes'],
            classical_hidden=cfg['classical_hidden'],
            n_classical_layers=cfg['n_classical_layers'],
            ansatz=cfg['ansatz'],                 activation=cfg['activation'],
        )
        model.load_state_dict(
            torch.load(_MODELS_DIR / f'{name}_weights.pt', weights_only=True)
        )
        model.eval()
    elif cfg['model'] in ('GBT', 'RF', 'SVM', 'MLP', 'XGB'):
        with open(_MODELS_DIR / f'{name}_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unknown model type in config: {cfg['model']}")
    return model, cfg


def _run_torch_inference(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


def _run_sklearn_inference(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X).astype(np.float32)


def _run_any_inference(model, cfg: dict, X: np.ndarray) -> np.ndarray:
    if cfg['model'] in ('VQC', 'HQNN'):
        return _run_torch_inference(model, X)
    return _run_sklearn_inference(model, X)   # GBT, RF, QKE, etc.


# ── Inference ETL ─────────────────────────────────────────────────────────────

# Minimum raw columns required for any model
_BASE_RAW_COLS = ['range', 'azimuth', 'PeakAmplitude', 'TotalAmplitude', 'down_range_extent']

# Kinematic rolling-stat columns (require track history in source data)
_KIN_STAT_COLS = ['RSog', 'measured_sog_std_900', 'measured_cog_std_900', 'measured_sog_avg_900']

# Human-readable descriptions for UI
_RAW_COL_NOTES = {
    'range':                 'radar slant range (m)',
    'azimuth':               'azimuth angle (deg)',
    'PeakAmplitude':         'peak detection amplitude',
    'TotalAmplitude':        'total integrated amplitude',
    'down_range_extent':     'range gate width (m)',
    'cross_range_extent':    'cross-range extent (m)  ← preferred',
    'az_extent_m':           'azimuth extent (radians) ← converted via × range',
    'RSog':                  'radar speed over ground (m/s)',
    'measured_sog_std_900':  'speed std dev over 900 s window',
    'measured_cog_std_900':  'course std dev over 900 s window',
    'measured_sog_avg_900':  'mean speed over 900 s window',
}


def _etl_inference(raw_df: 'pd.DataFrame', cfg: dict, pre: dict) -> tuple:
    """Unified ETL for inference — works with any raw radar CSV.

    Steps:
      1. Normalise cross-range column (metres or radians → metres)
      2. Validate base raw columns are present
      3. Compute all derivable features in one pass
      4. Drop physically invalid rows (amplitude/range ≤ 0)
      5. Select model's feature_cols; raise clear error if kinematic
         stats needed but not available
      6. Impute NaN with training-scaler means
      7. Scale with model's preprocessor scaler

    Returns: (X_scaled np.ndarray, df_clean pd.DataFrame, report dict)
    """
    df = raw_df.copy()
    report = {'n_input': len(df), 'warnings': [], 'conversions': []}

    # 1. Cross-range normalisation
    if 'cross_range_extent' not in df.columns:
        if 'az_extent_m' in df.columns:
            if 'range' not in df.columns:
                raise ValueError("CSV has 'az_extent_m' but no 'range' — cannot convert radians to metres.")
            df['cross_range_extent'] = df['az_extent_m'] * df['range']
            report['conversions'].append(
                "az_extent_m (radians) × range → cross_range_extent (metres)"
            )
        else:
            raise ValueError(
                "CSV must contain 'cross_range_extent' (metres) "
                "or 'az_extent_m' (radians) to compute extent features."
            )

    # 2. Validate base columns
    missing_base = [c for c in _BASE_RAW_COLS if c not in df.columns]
    if missing_base:
        raise ValueError(f"CSV is missing required raw columns: {missing_base}")

    # 3. Compute all derivable features
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
    df['az_extent_m']   = ec          # metres, overwrite radians if present

    if 'RSog' in df.columns:
        df['sog'] = df['RSog'].clip(upper=15.0)

    # 4. Drop physically invalid rows
    invalid = (
        (df['range']              <= 0) |
        (df['PeakAmplitude']      <= 0) |
        (df['TotalAmplitude']     <= 0) |
        (df['down_range_extent']  <= 0) |
        (df['cross_range_extent'] <= 0)
    )
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        df = df[~invalid].reset_index(drop=True)
        report['warnings'].append(
            f"{n_invalid} rows dropped: physically invalid values "
            f"(≤0 in range / amplitude / extent)"
        )

    # 5. Select model feature_cols — raise informative error if kinematic missing
    feature_cols = pre.get('feature_cols') or cfg.get('feature_cols', [])
    kin_needed   = [c for c in feature_cols if c in _KIN_STAT_COLS or c == 'sog']
    kin_missing  = [c for c in kin_needed if c not in df.columns]
    if kin_missing:
        raise ValueError(
            f"Model **{cfg.get('note', cfg.get('model',''))}** needs kinematic features "
            f"that are not in this CSV: `{kin_missing}`.\n\n"
            "Kinematic features require track history (rolling 900 s statistics). "
            "Upload the main tracked radar CSV (`radarfeatureL_MERGED_VALID_updated1.csv`) "
            "rather than a snapshot file."
        )

    other_missing = [c for c in feature_cols if c not in df.columns]
    if other_missing:
        raise ValueError(f"Cannot compute required model features: {other_missing}")

    X_df = df[feature_cols].copy()

    # 6. Impute NaN with training scaler means
    nan_mask = X_df.isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        scaler = pre.get('scaler')
        if scaler is not None and hasattr(scaler, 'mean_'):
            fill = dict(zip(feature_cols, scaler.mean_))
        else:
            fill = X_df.median().to_dict()
        X_df = X_df.fillna(fill)
        report['warnings'].append(
            f"{n_nan} rows had NaN feature values — imputed with training means"
        )

    report['n_output']    = len(X_df)
    report['n_dropped']   = report['n_input'] - report['n_output']
    report['feature_cols'] = feature_cols

    # 7. Scale
    scaler = pre.get('scaler')
    X_scaled = (scaler.transform(X_df.values.astype(np.float32))
                if scaler is not None else X_df.values.astype(np.float32))

    return X_scaled, df.iloc[:len(X_df)].copy(), report


# ---------- session state ----------

def _init():
    defaults = {
        'raw_df': None,
        'feature_cols': [],
        'label_col': None,
        'clean_df': None,
        'processed_data': None,
        'preprocessing_config': {},
        'model_config': {},
        'trained_models': _load_all_models(),   # restore from disk on every cold start
        'dp': DataProcessor(),
        'current_step': 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ---------- sidebar status ----------

def _sidebar():
    st.sidebar.title("📡 Radar QML")
    st.sidebar.markdown("### Steps")

    _has_weights = bool(list(_MODELS_DIR.glob('*_weights.pt')))
    _has_session_models = any(
        m.get('model') is not None
        for m in st.session_state.trained_models.values()
    )
    step_done = [
        st.session_state.raw_df is not None and bool(st.session_state.feature_cols),
        st.session_state.processed_data is not None,
        bool(st.session_state.model_config.get('model_type')),
        len(st.session_state.trained_models) > 0,
        len(st.session_state.trained_models) > 0,
        _has_weights or _has_session_models,
    ]

    cur = st.session_state.current_step
    for i, label in enumerate(STEPS):
        if i == cur:
            icon = "▶️"
        elif step_done[i]:
            icon = "✅"
        else:
            icon = "⬜"
        # Allow clicking any completed step or the current one
        disabled = not (step_done[i] or i == cur or i == 0)
        if st.sidebar.button(f"{icon}  {label}", key=f"_sb_step_{i}",
                             use_container_width=True,
                             disabled=disabled):
            st.session_state.current_step = i
            st.rerun()

    if st.session_state.trained_models:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Trained models**")
        for key, m in list(st.session_state.trained_models.items()):
            acc = m['metrics']['accuracy']
            col_lbl, col_del = st.sidebar.columns([3, 1])
            col_lbl.markdown(f"`{key}`  \n**{acc:.1%}**")
            if col_del.button("✕", key=f"_del_{key}", help=f"Remove {key}"):
                del st.session_state.trained_models[key]
                _delete_model(key)
                st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑 Clear All", help="Reset everything and delete saved models"):
        for p in _MODELS_DIR.glob('*.pkl'):
            p.unlink()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        _init()
        st.rerun()

_sidebar()


# ---------- main content ----------

st.title("📡 Radar Target Classification — QML Framework")
_cur = st.session_state.current_step
st.markdown(f"### Step {_cur + 1} of {len(STEPS)}: {STEPS[_cur]}")
st.markdown("---")


# ══════════════════════════════════════════════════════════════
# STEP 0 — Data
# ══════════════════════════════════════════════════════════════
if _cur == 0:
    st.header("Upload & Inspect Data")
    uploaded = st.file_uploader("Upload radar CSV file", type=["csv", "txt"])

    if uploaded:
        try:
            dp: DataProcessor = st.session_state.dp
            df = dp.load_csv(uploaded)
            st.session_state.raw_df = df
            st.session_state.clean_df = None
            st.session_state.processed_data = None
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

    df = st.session_state.raw_df
    if df is not None:
        st.success(f"Loaded — {df.shape[0]} rows × {df.shape[1]} columns")

        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            all_cols = df.columns.tolist()

            label_col = st.selectbox(
                "Label column",
                options=all_cols,
                index=len(all_cols) - 1,
                key='label_col_select',
            )
            st.session_state.label_col = label_col

            # Auto-detect ID-like columns: name matches common patterns OR all values unique
            _id_patterns = ('id', 'idx', 'index', 'no', 'num', 'number', 'row', 'serial', 'key', 'code')
            def _looks_like_id(col):
                if col == label_col:
                    return False
                name_match = any(col.lower() == p or col.lower().startswith(p + '_') or col.lower().endswith('_' + p)
                                 for p in _id_patterns)
                # Only flag as ID by uniqueness if the column is integer-typed (floats are legitimately all-unique)
                is_int = pd.api.types.is_integer_dtype(df[col])
                all_unique_int = is_int and (df[col].nunique() == len(df))
                return name_match or all_unique_int

            detected_id_cols = [c for c in numeric_cols if _looks_like_id(c)]

            default_features = [c for c in numeric_cols if c != label_col and c not in detected_id_cols]
            feature_cols = st.multiselect(
                "Feature columns",
                options=[c for c in all_cols if c != label_col],
                default=default_features,
                key='feature_cols_select',
            )
            st.session_state.feature_cols = feature_cols

            if detected_id_cols:
                st.warning(
                    f"Auto-excluded from features (look like IDs): **{', '.join(detected_id_cols)}**  \n"
                    "Re-add them above if they are genuine features."
                )

        with col2:
            if feature_cols and label_col:
                stats = dp.get_statistics(df, feature_cols, label_col)
                st.metric("Rows", stats['shape'][0])
                st.metric("Features selected", len(feature_cols))
                st.metric("Missing cells", int(stats['missing'].sum()))
                st.metric("Duplicate rows", int(stats['duplicates']))

        if feature_cols and label_col:
            with st.expander("Data preview", expanded=False):
                st.dataframe(df[feature_cols + [label_col]].head(50))

            with st.expander("Feature statistics", expanded=False):
                st.dataframe(df[feature_cols].describe().T)

            st.subheader("Class distribution")
            counts = df[label_col].value_counts().sort_index()
            st.plotly_chart(plot_class_distribution(counts), use_container_width=True)

            # ── Class filter ───────────────────────────────────────────
            st.subheader("Select classes to keep")
            all_classes = sorted([str(c) for c in df[label_col].dropna().unique().tolist()])
            prev_included = st.session_state.get('included_classes', all_classes)
            # Keep only classes still present in the current file
            prev_included = [c for c in prev_included if c in all_classes]

            included_classes = st.multiselect(
                "Deselect any class to drop it from all splits",
                options=all_classes,
                default=prev_included,
                key='included_classes_select',
            )
            if included_classes != st.session_state.get('included_classes'):
                st.session_state.included_classes = included_classes
                st.session_state.processed_data = None   # force re-preprocessing

            if not included_classes:
                st.error("Select at least one class.")
            else:
                n_kept  = int(df[label_col].astype(str).isin(included_classes).sum())
                n_total = len(df)
                dropped_classes = [c for c in all_classes if c not in included_classes]

                # Per-class row counts with keep/drop badge
                summary_rows = []
                for cls in all_classes:
                    n = int((df[label_col].astype(str) == cls).sum())
                    status = "✅ keep" if cls in included_classes else "🗑 drop"
                    summary_rows.append({'Class': cls, 'Rows': n, 'Status': status})
                st.dataframe(pd.DataFrame(summary_rows).set_index('Class'), use_container_width=True)

                if dropped_classes:
                    n_dropped = n_total - n_kept
                    st.warning(
                        f"Dropping **{len(dropped_classes)} class(es)**: {dropped_classes}  \n"
                        f"Removes **{n_dropped} rows** ({n_dropped/n_total:.1%} of dataset). "
                        f"**{n_kept} rows** remain."
                    )
                else:
                    st.success(f"All {len(all_classes)} classes included ({n_total} rows).")

            st.subheader("Missing values")
            missing = df[feature_cols + [label_col]].isnull().sum()
            if missing.sum() == 0:
                st.info("No missing values.")
            else:
                st.dataframe(missing[missing > 0].rename("Missing count"))

            # ── Tab 1 footer ───────────────────────────────────────────
            _inc = st.session_state.get('included_classes', [])
            _tab1_ready = bool(feature_cols) and bool(label_col) and bool(_inc)
            _tab1_msg = (
                f"{len(feature_cols)} feature(s) selected · "
                f"Label: **{label_col}** · "
                f"{len(_inc)} class(es) included. "
                f"Ready to configure preprocessing."
            )
            _tab_footer(
                next_index=1,
                next_label="2 · Preprocessing",
                is_ready=_tab1_ready,
                ready_msg=_tab1_msg,
                not_ready_msg="Select at least one feature column, a label column, and at least one class to continue.",
                key="tab1_next",
            )


# ══════════════════════════════════════════════════════════════
# STEP 1 — Preprocessing
# ══════════════════════════════════════════════════════════════
if _cur == 1:
    st.header("Data Cleaning & Preprocessing")
    if st.session_state.raw_df is None or not st.session_state.feature_cols:
        st.warning("Load a dataset and select columns in Tab 1 first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cleaning")
            remove_dup = st.checkbox("Remove duplicate rows", value=True)
            missing_strategy = st.selectbox(
                "Missing value strategy",
                ["drop", "mean", "median", "mode"],
                help="How to handle NaN values in feature columns",
            )
            outlier_method = st.selectbox(
                "Outlier removal",
                ["none", "iqr", "zscore"],
                help="IQR: removes rows where any feature is beyond 1.5×IQR. Z-score: removes rows beyond threshold.",
            )
            zscore_thr = 3.0
            if outlier_method == 'zscore':
                zscore_thr = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1)

        with col2:
            st.subheader("Normalization & Split")
            normalization = st.selectbox(
                "Normalization",
                ["standard", "minmax", "robust", "none"],
                help="Fit on train, apply to test (no data leakage).",
            )
            test_size = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)
            stratify = st.checkbox("Stratified split", value=True)
            random_state = st.number_input("Random seed", value=42, step=1)

            st.subheader("Dimensionality Reduction (PCA)")
            use_pca = st.checkbox(
                "Apply PCA before training",
                help="Fit PCA on the training split; required when n_features > n_qubits for quantum models.",
            )
            pca_components = 4
            if use_pca:
                n_feat = len(st.session_state.feature_cols)
                pca_components = st.slider("Components to keep", 2, min(n_feat, 20), min(4, n_feat))

            st.subheader("Class Balancing (train only)")
            balancing = st.selectbox(
                "Resampling method",
                ["none", "smote", "undersample"],
                help=(
                    "**none** — use data as-is  \n"
                    "**smote** — synthesise new minority-class samples (oversampling)  \n"
                    "**undersample** — randomly remove majority-class samples"
                ),
            )
            use_class_weights = st.checkbox(
                "Class-weighted loss (for neural/quantum models)",
                value=False,
                help=(
                    "Passes inverse-frequency weights to CrossEntropyLoss so the model is penalised "
                    "more for misclassifying rare classes. Can be combined with resampling or used alone."
                ),
            )

        # ── PCA Explorer ──────────────────────────────────────────────
        if use_pca:
            st.markdown("---")
            st.subheader("PCA Explorer")
            st.caption(
                "Run PCA on the cleaned, normalised data to inspect variance and feature loadings "
                "before committing to a component count."
            )

            _pca_feats = st.session_state.feature_cols
            _pca_lbl   = st.session_state.label_col
            with st.expander("Columns entering PCA (verify before running)", expanded=True):
                st.markdown(f"**Label column** (excluded): `{_pca_lbl}`")
                st.markdown(f"**Feature columns** ({len(_pca_feats)}):  \n" +
                            "  \n".join(f"- `{c}`" for c in _pca_feats))
                _non_feat = [c for c in st.session_state.raw_df.columns
                             if c not in _pca_feats and c != _pca_lbl]
                if _non_feat:
                    st.markdown(f"**Excluded** (not in feature list): `{', '.join(_non_feat)}`")

            col_pca_btn, col_pca_show = st.columns([1, 3])
            with col_pca_btn:
                run_pca_preview = st.button("🔍 Analyse PCA", help="Fits PCA on all components so you can pick N.")
                show_n = st.slider("Components to display", 2, min(len(st.session_state.feature_cols), 12), 4,
                                   key='pca_show_n')

            if run_pca_preview:
                try:
                    dp_tmp: DataProcessor = st.session_state.dp
                    feat = st.session_state.feature_cols
                    lbl  = st.session_state.label_col
                    raw  = st.session_state.raw_df

                    with st.spinner("Cleaning and fitting PCA…"):
                        clean_cfg_tmp = {
                            'remove_duplicates': remove_dup,
                            'missing_strategy': missing_strategy,
                            'outlier_method': outlier_method,
                            'zscore_threshold': zscore_thr,
                        }
                        clean_tmp = dp_tmp.clean(raw, feat, lbl, clean_cfg_tmp)
                        analysis  = dp_tmp.pca_analysis(clean_tmp, feat, normalization if normalization != 'none' else 'standard')
                    st.session_state['pca_analysis'] = analysis
                except Exception as e:
                    st.error(f"PCA analysis failed: {e}")

            analysis = st.session_state.get('pca_analysis')
            if analysis:
                from src.visualization import plot_pca_scree, plot_pca_loadings, plot_pca_variance_bar
                n_total = analysis['n_components_total']

                # Scree plot + variance retained
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.plotly_chart(plot_pca_scree(analysis), use_container_width=True)
                with sc2:
                    st.plotly_chart(plot_pca_variance_bar(analysis, pca_components), use_container_width=True)

                # Loadings heatmap for top N components
                st.plotly_chart(plot_pca_loadings(analysis, show_n), use_container_width=True)

                # Numeric table of top-4 loadings
                with st.expander("Top 4 PC loadings (table)", expanded=False):
                    import pandas as _pd
                    n4 = min(4, n_total)
                    feat_names = analysis['feature_names']
                    pc_labels  = [f'PC{i+1}' for i in range(n4)]
                    load_df = _pd.DataFrame(
                        analysis['components'][:n4].T,
                        index=feat_names,
                        columns=pc_labels,
                    ).round(4)
                    # highlight strongest contributor per PC
                    try:
                        styled = load_df.style.background_gradient(cmap='RdBu', axis=0, vmin=-1, vmax=1)
                        st.dataframe(styled, use_container_width=True)
                    except ImportError:
                        st.dataframe(load_df, use_container_width=True)

                cum_pct = float(analysis['cumulative_variance'][pca_components - 1]) * 100
                st.info(
                    f"Keeping **{pca_components} components** retains **{cum_pct:.1f}%** of total variance. "
                    f"Adjust the slider above if needed, then click **Apply Preprocessing**."
                )

        st.markdown("---")
        if st.button("✅ Apply Preprocessing", type="primary"):
            try:
                dp: DataProcessor = st.session_state.dp
                feature_cols = st.session_state.feature_cols
                label_col = st.session_state.label_col
                df = st.session_state.raw_df

                clean_cfg = {
                    'remove_duplicates': remove_dup,
                    'missing_strategy': missing_strategy,
                    'outlier_method': outlier_method,
                    'zscore_threshold': zscore_thr,
                }
                pre_cfg = {
                    'normalization': normalization if normalization != 'none' else None,
                    'test_size': test_size,
                    'stratify': stratify,
                    'random_state': int(random_state),
                    'use_pca': use_pca,
                    'pca_components': pca_components if use_pca else None,
                    'balancing': balancing,
                }

                with st.spinner("Cleaning data..."):
                    clean_df = dp.clean(df, feature_cols, label_col, clean_cfg)

                # Drop excluded classes
                included = st.session_state.get('included_classes')
                if included:
                    before = len(clean_df)
                    clean_df = clean_df[clean_df[label_col].astype(str).isin(included)].reset_index(drop=True)
                    dropped_rows = before - len(clean_df)
                    if dropped_rows:
                        st.info(f"Dropped {dropped_rows} rows belonging to excluded classes.")

                st.session_state.clean_df = clean_df
                st.session_state.preprocessing_config = {**clean_cfg, **pre_cfg}

                with st.spinner("Normalizing and splitting..."):
                    data = dp.preprocess(clean_df, feature_cols, label_col, pre_cfg)

                # Compute inverse-frequency class weights from the train split
                import numpy as _np
                counts_arr = _np.bincount(data['y_train'], minlength=data['n_classes']).astype(float)
                counts_arr = _np.where(counts_arr == 0, 1, counts_arr)   # avoid /0
                weights    = 1.0 / counts_arr
                weights    = (weights / weights.sum() * data['n_classes']).tolist()
                data['class_weights']     = weights
                data['use_class_weights'] = use_class_weights

                st.session_state.processed_data = data

                st.success(
                    f"Done — Train: {data['train_size']} | Test: {data['test_size']} | "
                    f"Features: {data['n_features']} | Classes: {data['n_classes']}"
                )
                if data['pca_explained_variance']:
                    st.info(f"PCA variance explained: {data['pca_explained_variance']:.1%}")

                counts_series = pd.Series(data['y_train']).value_counts().sort_index()
                counts_series.index = [data['class_names'][i] for i in counts_series.index]
                st.plotly_chart(
                    plot_class_distribution(counts_series, "Train Class Distribution (after resampling)"),
                    use_container_width=True,
                )

                # Imbalance ratio summary
                max_c, min_c = int(counts_series.max()), int(counts_series.min())
                ratio = max_c / max(min_c, 1)
                if ratio > 3:
                    st.warning(
                        f"Imbalance ratio {ratio:.1f}× (largest {max_c} vs smallest {min_c} samples).  \n"
                        f"Consider **SMOTE**, **undersampling**, or enabling **class-weighted loss**."
                    )
                else:
                    st.success(f"Classes are reasonably balanced (ratio {ratio:.1f}×).")

                if use_class_weights:
                    weight_table = pd.DataFrame({
                        'Class': data['class_names'],
                        'Train samples': [int(counts_arr[i]) for i in range(data['n_classes'])],
                        'Loss weight': [round(w, 4) for w in weights],
                    }).set_index('Class')
                    with st.expander("Class loss weights", expanded=True):
                        st.dataframe(weight_table, use_container_width=True)
                        st.caption("Rare classes get a higher weight → larger gradient penalty for misclassification.")
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                raise

        # ── Tab 2 footer ───────────────────────────────────────────
        _pd = st.session_state.processed_data
        _tab2_ready = _pd is not None
        _tab2_msg = (
            f"Preprocessing complete · "
            f"Train: **{_pd['train_size']}** · Test: **{_pd['test_size']}** · "
            f"{_pd['n_features']} feature(s) · {_pd['n_classes']} class(es). "
            f"Ready to configure a model."
        ) if _tab2_ready else ""
        _tab_footer(
            next_index=2,
            next_label="3 · Model Config",
            is_ready=_tab2_ready,
            ready_msg=_tab2_msg,
            not_ready_msg="Click **Apply Preprocessing** above to complete this step.",
            key="tab2_next",
        )


# ══════════════════════════════════════════════════════════════
# STEP 2 — Model Configuration
# ══════════════════════════════════════════════════════════════
if _cur == 2:
    st.header("Model Configuration")
    if st.session_state.processed_data is None:
        st.warning("Complete preprocessing in Tab 2 first.")
    else:
        data = st.session_state.processed_data
        n_features = data['n_features']
        n_classes = data['n_classes']
        class_names = data['class_names']

        st.info(f"Input: **{n_features}** features → **{n_classes}** classes: {', '.join(class_names)}")

        model_type = st.selectbox(
            "Model type",
            ["VQC — Variational Quantum Classifier",
             "HQNN — Hybrid Quantum Neural Network",
             "QKE — Quantum Kernel Estimation",
             "Classical — Random Forest",
             "Classical — SVM",
             "Classical — MLP",
             "Classical — Gradient Boosting",
             "Classical — XGBoost",
             "Ensemble — Combine Trained Models"],
        )

        st.session_state.model_config = {'model_type': model_type,
                                          'n_features': n_features,
                                          'n_classes': n_classes}

        if model_type.startswith("VQC") or model_type.startswith("HQNN"):
            st.subheader("Quantum Circuit")
            c1, c2, c3 = st.columns(3)
            with c1:
                n_qubits = st.slider("Qubits", 2, 10, min(6, n_features),
                                     help="Recommended ≤ 8 for reasonable simulation speed.")
            with c2:
                n_layers = st.slider("Ansatz layers", 1, 6, 2)
            with c3:
                ansatz = st.selectbox("Ansatz", ["strongly_entangling", "basic_entangler", "hardware_efficient"])

            if model_type.startswith("VQC"):
                embedding = st.selectbox("Embedding", ["angle", "zzfeaturemap"])
            else:
                embedding = "angle"

            if model_type.startswith("HQNN"):
                st.subheader("Classical Encoder / Decoder")
                cc1, cc2 = st.columns(2)
                with cc1:
                    classical_hidden = st.slider("Hidden units", 8, 128, 16)
                    n_classical_layers = st.slider("Classical layers (enc/dec)", 1, 4, 1)
                with cc2:
                    activation = st.selectbox("Activation", ["relu", "tanh", "leaky_relu", "sigmoid"])
                st.session_state.model_config.update({
                    'classical_hidden': classical_hidden,
                    'n_classical_layers': n_classical_layers,
                    'activation': activation,
                })

            st.subheader("Training")
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                optimizer = st.selectbox("Optimizer", ["adam", "adamw", "sgd", "rmsprop"])
            with tc2:
                lr = st.number_input("Learning rate", value=0.01, format="%.4f", step=0.001)
            with tc3:
                batch_size = st.slider("Batch size", 8, 256, 32)
            with tc4:
                epochs = st.slider("Epochs", 5, 200, 30)
            weight_decay = st.number_input("Weight decay", value=0.0, format="%.5f", step=0.0001)

            st.session_state.model_config.update({
                'n_qubits': n_qubits, 'n_layers': n_layers,
                'ansatz': ansatz, 'embedding': embedding,
                'optimizer': optimizer, 'learning_rate': lr,
                'batch_size': batch_size, 'epochs': epochs,
                'weight_decay': weight_decay,
            })

        elif model_type.startswith("QKE"):
            st.subheader("Quantum Kernel")
            qc1, qc2 = st.columns(2)
            with qc1:
                n_qubits = st.slider("Qubits", 2, 8, min(4, n_features))
                n_layers = st.slider("Feature map repetitions", 1, 4, 1)
                qke_embedding = st.selectbox("Feature map", ["zzfeaturemap", "angle"])
            with qc2:
                svm_c = st.number_input("SVM C", value=1.0, format="%.3f", step=0.1)
                max_samples = st.slider("Max train samples (for speed)", 50, 500, 200,
                                        help="QKE scales O(N²). Large datasets will be subsampled.")
            st.warning(
                f"QKE requires O(N²) circuit evaluations. "
                f"With {min(max_samples, data['train_size'])} samples × {n_qubits} qubits this may take several minutes."
            )
            st.session_state.model_config.update({
                'n_qubits': n_qubits, 'n_layers': n_layers,
                'embedding': qke_embedding, 'svm_c': svm_c,
                'max_train_samples': max_samples,
            })

        elif model_type == "Classical — Random Forest":
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                n_estimators = st.slider("n_estimators", 10, 500, 100)
            with cc2:
                max_depth = st.slider("max_depth (0=None)", 0, 50, 0)
            with cc3:
                min_samples_split = st.slider("min_samples_split", 2, 20, 2)
            st.session_state.model_config.update({
                'n_estimators': n_estimators,
                'max_depth': max_depth if max_depth > 0 else None,
                'min_samples_split': min_samples_split,
                'random_state': 42,
            })

        elif model_type == "Classical — SVM":
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            with sc2:
                svm_c = st.number_input("C", value=1.0, format="%.3f")
            with sc3:
                svm_gamma = st.selectbox("Gamma", ["scale", "auto"])
            st.session_state.model_config.update({
                'kernel': svm_kernel, 'C': svm_c, 'gamma': svm_gamma,
                'probability': True,
            })

        elif model_type == "Classical — MLP":
            mc1, mc2 = st.columns(2)
            with mc1:
                hidden_str = st.text_input("Hidden layers (e.g. 64,32)", "64,32")
                try:
                    hidden = tuple(int(x.strip()) for x in hidden_str.split(','))
                except Exception:
                    hidden = (64, 32)
                mlp_activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])
            with mc2:
                mlp_lr = st.number_input("Learning rate", value=0.001, format="%.5f")
                mlp_epochs = st.slider("Max iterations", 50, 1000, 200)
            st.session_state.model_config.update({
                'hidden_layer_sizes': hidden,
                'activation': mlp_activation,
                'learning_rate_init': mlp_lr,
                'max_iter': mlp_epochs,
                'random_state': 42,
            })

        elif model_type == "Classical — Gradient Boosting":
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                gb_estimators = st.slider("n_estimators", 10, 500, 100)
                gb_lr = st.number_input("Learning rate", value=0.1, format="%.3f")
            with gc2:
                gb_depth = st.slider("max_depth", 1, 10, 3)
            with gc3:
                gb_subsample = st.slider("subsample", 0.5, 1.0, 1.0, 0.05)
            st.session_state.model_config.update({
                'n_estimators': gb_estimators,
                'learning_rate': gb_lr,
                'max_depth': gb_depth,
                'subsample': gb_subsample,
                'random_state': 42,
            })

        elif model_type == "Classical — XGBoost":
            xc1, xc2, xc3 = st.columns(3)
            with xc1:
                xgb_estimators = st.slider("n_estimators", 50, 1000, 300)
                xgb_lr = st.number_input("Learning rate", value=0.05, format="%.3f", key="xgb_lr")
            with xc2:
                xgb_depth = st.slider("max_depth", 2, 8, 4)
                xgb_subsample = st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key="xgb_sub")
            with xc3:
                xgb_colsample = st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
                xgb_min_child = st.slider("min_child_weight", 1, 10, 2)
            st.session_state.model_config.update({
                'n_estimators': xgb_estimators,
                'learning_rate': xgb_lr,
                'max_depth': xgb_depth,
                'subsample': xgb_subsample,
                'colsample_bytree': xgb_colsample,
                'min_child_weight': xgb_min_child,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'verbosity': 0,
            })

        if model_type.startswith("Ensemble"):
            st.subheader("Ensemble Configuration")
            available_models = list(st.session_state.trained_models.keys())

            if len(available_models) < 2:
                st.error(
                    f"Only **{len(available_models)}** trained model(s) found. "
                    "Train at least 2 models (e.g. VQC and HQNN) in Step 4, then come back here."
                )
                ensemble_models, voting = [], 'soft'
            else:
                st.info(f"**{len(available_models)} trained models available:** {', '.join(f'`{m}`' for m in available_models)}")
                ec1, ec2 = st.columns(2)
                with ec1:
                    ensemble_models = st.multiselect(
                        "Models to combine (select ≥ 2)",
                        options=available_models,
                        default=available_models,
                        key='ensemble_model_select',
                    )
                with ec2:
                    voting = st.radio(
                        "Voting method",
                        ["soft", "hard"],
                        help=(
                            "**Soft** — average class probabilities (recommended).  \n"
                            "**Hard** — majority vote on predicted class."
                        ),
                    )
                if len(ensemble_models) < 2:
                    st.warning("Select at least 2 models to enable the ensemble.")
                else:
                    st.success(
                        f"Ready to combine: **{' + '.join(ensemble_models)}**  \n"
                        f"Voting: **{voting}**. Runs inference immediately — no retraining needed."
                    )

            st.session_state.model_config.update({
                'ensemble_models': ensemble_models,
                'voting': voting,
            })

        model_name = st.text_input(
            "Model run name",
            value=model_type.split('—')[0].strip().replace(' ', '_'),
            help="Used as the key in the results comparison.",
        )
        st.session_state.model_config['run_name'] = model_name

        # ── Tab 3 footer ───────────────────────────────────────────
        _mc = st.session_state.model_config
        _is_ensemble = _mc.get('model_type', '').startswith('Ensemble')
        _ensemble_ok  = len(_mc.get('ensemble_models', [])) >= 2
        _tab3_ready   = (
            bool(_mc.get('model_type')) and
            bool(_mc.get('run_name')) and
            (not _is_ensemble or _ensemble_ok)
        )
        _tab3_msg = (
            f"Model configured · **{_mc.get('model_type', '')}** · "
            f"Run name: **{_mc.get('run_name', '')}**. "
            f"Ready to train."
        )
        _tab_footer(
            next_index=3,
            next_label="4 · Train",
            is_ready=_tab3_ready,
            ready_msg=_tab3_msg,
            not_ready_msg="Complete model configuration above.",
            key="tab3_next",
        )


# ══════════════════════════════════════════════════════════════
# STEP 3 — Training
# ══════════════════════════════════════════════════════════════
if _cur == 3:
    st.header("Train Model")
    if st.session_state.processed_data is None:
        st.warning("Complete preprocessing in Tab 2 first.")
    elif not st.session_state.model_config:
        st.warning("Configure a model in Tab 3 first.")
    else:
        cfg = st.session_state.model_config
        data = st.session_state.processed_data
        st.info(
            f"**Model:** {cfg['model_type']}  |  "
            f"**Run name:** {cfg.get('run_name', 'model')}  |  "
            f"**Features:** {data['n_features']}  |  "
            f"**Classes:** {data['n_classes']}"
        )

        if st.button("🚀 Start Training", type="primary"):
            run_name = cfg.get('run_name', 'model')
            model_type = cfg['model_type']
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            class_names = data['class_names']

            progress_bar = st.progress(0)
            status_text = st.empty()
            metric_cols = st.columns(4)
            chart_placeholder = st.empty()
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            is_torch_model = False
            trained_model = None

            try:
                if model_type.startswith("VQC"):
                    from src.models.vqc_model import VQCClassifier
                    is_torch_model = True
                    status_text.text("Building VQC model...")
                    model = VQCClassifier(
                        n_features=data['n_features'],
                        n_qubits=cfg['n_qubits'],
                        n_layers=cfg['n_layers'],
                        n_classes=data['n_classes'],
                        embedding=cfg['embedding'],
                        ansatz=cfg['ansatz'],
                    )
                    train_cfg = {k: cfg[k] for k in ['optimizer', 'learning_rate', 'batch_size', 'epochs', 'weight_decay']}
                    if data.get('use_class_weights'):
                        train_cfg['class_weights'] = data['class_weights']

                elif model_type.startswith("HQNN"):
                    from src.models.hqnn_model import HQNNClassifier
                    is_torch_model = True
                    status_text.text("Building HQNN model...")
                    model = HQNNClassifier(
                        n_features=data['n_features'],
                        n_qubits=cfg['n_qubits'],
                        n_layers=cfg['n_layers'],
                        n_classes=data['n_classes'],
                        classical_hidden=cfg['classical_hidden'],
                        n_classical_layers=cfg['n_classical_layers'],
                        ansatz=cfg['ansatz'],
                        activation=cfg['activation'],
                    )
                    train_cfg = {k: cfg[k] for k in ['optimizer', 'learning_rate', 'batch_size', 'epochs', 'weight_decay']}
                    if data.get('use_class_weights'):
                        train_cfg['class_weights'] = data['class_weights']

                elif model_type.startswith("QKE"):
                    from src.models.qke_model import QKEClassifier
                    status_text.text("Computing quantum kernel matrix (this takes time)...")
                    model = QKEClassifier(
                        n_qubits=cfg['n_qubits'],
                        n_layers=cfg['n_layers'],
                        embedding=cfg['embedding'],
                        svm_c=cfg['svm_c'],
                        max_train_samples=cfg['max_train_samples'],
                    )
                    model.fit(X_train, y_train, X_test, y_test)
                    history = model.history
                    trained_model = model

                elif model_type.startswith("Ensemble"):
                    from src.models.ensemble_model import EnsembleModel
                    selected_names = cfg.get('ensemble_models', [])
                    voting = cfg.get('voting', 'soft')
                    if len(selected_names) < 2:
                        available = list(st.session_state.trained_models.keys())
                        st.error(
                            f"Ensemble needs ≥ 2 models selected — config currently has: "
                            f"`{selected_names}`.  \n"
                            f"Available saved models: **{', '.join(available) if available else 'none'}**.  \n"
                            "Go back to **Step 3**, select ≥ 2 models in the multiselect, then return here."
                        )
                        if st.button("← Back to Step 3"):
                            st.session_state.current_step = 2
                            st.rerun()
                        st.stop()
                    status_text.text(f"Building {voting}-voting ensemble from: {', '.join(selected_names)}…")
                    models_meta = []
                    for mname in selected_names:
                        m = st.session_state.trained_models[mname]
                        models_meta.append({
                            'model':         m.get('model'),
                            'is_torch':      m['is_torch'],
                            'name':          mname,
                            'probabilities': m['metrics'].get('probabilities'),
                        })
                    trained_model = EnsembleModel(models_meta, voting=voting)
                    history = {}
                    progress_bar.progress(1.0)

                else:
                    from src.models.classical_model import ClassicalModel
                    display_name = model_type.replace("Classical — ", "")
                    params = {k: v for k, v in cfg.items()
                              if k not in ('model_type', 'n_features', 'n_classes', 'run_name')}
                    model = ClassicalModel(display_name, params)
                    status_text.text(f"Training {display_name}...")
                    model.fit(X_train, y_train, X_test, y_test)
                    history = model.history
                    trained_model = model

                if is_torch_model:
                    trainer = TorchTrainer(model, train_cfg)
                    epochs_total = train_cfg['epochs']

                    def _callback(info):
                        ep = info['epoch']
                        progress_bar.progress(ep / epochs_total)
                        status_text.text(
                            f"Epoch {ep}/{epochs_total} — "
                            f"Loss: {info['train_loss']:.4f}  Acc: {info['train_acc']:.1%}  "
                            f"Val Acc: {info['val_acc']:.1%}"
                        )
                        history['train_loss'].append(info['train_loss'])
                        history['train_acc'].append(info['train_acc'])
                        history['val_loss'].append(info['val_loss'])
                        history['val_acc'].append(info['val_acc'])
                        if ep % max(1, epochs_total // 20) == 0:
                            fig = plot_training_curves(history, run_name)
                            chart_placeholder.plotly_chart(fig, use_container_width=True)

                    trainer.train(X_train, y_train, X_test, y_test, callback=_callback)
                    trained_model = model
                    progress_bar.progress(1.0)

                status_text.text("Evaluating...")
                metrics = evaluate_model(
                    trained_model,
                    X_test, y_test,
                    class_names,
                    is_torch=is_torch_model,
                )

                model_entry = {
                    'model': trained_model,
                    'metrics': metrics,
                    'history': history,
                    'model_type': model_type,
                    'is_torch': is_torch_model,
                }
                st.session_state.trained_models[run_name] = model_entry
                _save_model(run_name, model_entry)

                progress_bar.progress(1.0)
                status_text.success(
                    f"Training complete — Accuracy: {metrics['accuracy']:.1%}  "
                    f"F1 Macro: {metrics['f1_macro']:.1%}"
                )

                c1, c2, c3, c4 = metric_cols
                c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                c2.metric("F1 Macro", f"{metrics['f1_macro']:.1%}")
                c3.metric("F1 Weighted", f"{metrics['f1_weighted']:.1%}")
                auc_val = metrics.get('auc_macro')
                c4.metric("AUC Macro", f"{auc_val:.3f}" if auc_val else "N/A")

            except Exception as e:
                st.error(f"Training failed: {e}")
                raise

        # ── Tab 4 footer ───────────────────────────────────────────
        _tm = st.session_state.trained_models
        _tab4_ready = len(_tm) > 0
        _best = max(_tm.items(), key=lambda kv: kv[1]['metrics']['accuracy'])[0] if _tab4_ready else None
        _tab4_msg = (
            f"{len(_tm)} model(s) trained · "
            f"Best so far: **{_best}** "
            f"({_tm[_best]['metrics']['accuracy']:.1%} accuracy). "
            f"View full results and comparisons."
        ) if _tab4_ready else ""
        _tab_footer(
            next_index=4,
            next_label="5 · Results & Compare",
            is_ready=_tab4_ready,
            ready_msg=_tab4_msg,
            not_ready_msg="Train at least one model above to unlock results.",
            key="tab4_next",
        )


# ══════════════════════════════════════════════════════════════
# STEP 4 — Results & Compare
# ══════════════════════════════════════════════════════════════
if _cur == 4:
    st.header("Results & Model Comparison")
    models = st.session_state.trained_models

    if not models:
        st.info("Train at least one model in Tab 4.")
    else:
        # Comparison bar chart
        if len(models) > 1:
            st.subheader("Comparison")
            st.plotly_chart(plot_model_comparison(models), use_container_width=True)

        # Summary table
        st.subheader("Metrics Summary")
        rows = []
        for key, m in models.items():
            met = m['metrics']
            rows.append({
                'Run': key,
                'Type': m['model_type'],
                'Accuracy': f"{met['accuracy']:.1%}",
                'F1 Macro': f"{met['f1_macro']:.1%}",
                'F1 Weighted': f"{met['f1_weighted']:.1%}",
                'AUC Macro': f"{met['auc_macro']:.3f}" if met.get('auc_macro') else 'N/A',
            })
        st.dataframe(pd.DataFrame(rows).set_index('Run'), use_container_width=True)

        # Per-model deep dive
        st.subheader("Per-model Analysis")
        selected = st.selectbox("Select model to inspect", list(models.keys()))
        m = models[selected]
        met = m['metrics']
        _pd = st.session_state.processed_data
        class_names = _pd['class_names'] if _pd else [str(c) for c in range(4)]

        col_a, col_b = st.columns(2)

        with col_a:
            if met.get('confusion_matrix') is not None:
                st.plotly_chart(
                    plot_confusion_matrix(met['confusion_matrix'], class_names, selected),
                    use_container_width=True,
                )
            else:
                st.info("Confusion matrix not available for pre-trained models loaded from disk.  \n"
                        "Upload data and run inference to evaluate on new data (Step 6).")

        with col_b:
            if st.session_state.processed_data is not None:
                fig_roc = plot_roc_curves(
                    st.session_state.processed_data['y_test'],
                    met.get('probabilities'),
                    class_names, selected,
                )
                if fig_roc:
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC curves not available (model does not output probabilities).")
            else:
                st.info("Load data in Steps 1–2 to view ROC curves.")

        # Training curves
        if m['history'] and any(len(v) > 0 for v in m['history'].values()):
            fig_curves = plot_training_curves(m['history'], selected)
            if fig_curves:
                st.plotly_chart(fig_curves, use_container_width=True)

        # Classification report
        st.subheader("Classification Report")
        report = met['classification_report']
        report_rows = []
        for cls in class_names:
            if cls in report:
                r = report[cls]
                report_rows.append({
                    'Class': cls,
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1': f"{r['f1-score']:.3f}",
                    'Support': int(r['support']),
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows).set_index('Class'), use_container_width=True)

        # Feature importance
        model_obj = m['model']
        if hasattr(model_obj, 'model'):
            model_obj = model_obj.model
        importances = None
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, 'get_feature_importance'):
            importances = model_obj.get_feature_importance()

        if importances is not None:
            feat_names = st.session_state.processed_data.get('feature_names',
                         [f'F{i}' for i in range(len(importances))])
            st.subheader("Feature Importance")
            st.plotly_chart(
                plot_feature_importance(importances, feat_names, selected),
                use_container_width=True,
            )

        # Export predictions
        st.subheader("Export")
        preds = met.get('predictions')
        if preds is not None:
            y_test = st.session_state.processed_data['y_test']
            export_df = pd.DataFrame({
                'true_label': [class_names[i] for i in y_test],
                'predicted_label': [class_names[i] for i in preds],
                'correct': y_test == preds,
            })
            if met.get('probabilities') is not None:
                for j, cls in enumerate(class_names):
                    export_df[f'prob_{cls}'] = met['probabilities'][:, j]
            csv_bytes = export_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Predictions CSV",
                data=csv_bytes,
                file_name=f"predictions_{selected}.csv",
                mime='text/csv',
            )


# ══════════════════════════════════════════════════════════════
# STEP 5 — Inference on New Data  +  Benchmark All Models
# ══════════════════════════════════════════════════════════════

def _ece_calc(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    conf  = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    corr  = (preds == y_true).astype(float)
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(corr[mask].mean() - conf[mask].mean())
    conf_c = float(conf[corr == 1].mean()) if corr.sum() > 0 else float('nan')
    conf_w = float(conf[corr == 0].mean()) if (1 - corr).sum() > 0 else float('nan')
    return round(ece, 4), round(conf_c, 3), round(conf_w, 3)


if _cur == 5:
    st.header("Inference & Benchmark")

    inf_mode = st.radio(
        "Mode",
        ["🔮 Single Model Inference", "🏆 Benchmark — compare all saved models"],
        horizontal=True,
        key="inf_mode_radio",
    )
    st.markdown("---")

    # ── Discover available models (shared across both modes) ──
    saved_weight_names = _get_saved_weight_models()
    session_model_names = [
        k for k, v in st.session_state.trained_models.items()
        if v.get('model') is not None
    ]

    if not saved_weight_names and not session_model_names:
        st.warning(
            "No models available.  \n"
            "• Train models in **Step 4**, or  \n"
            "• Save weights via the training scripts."
        )
        st.stop()

    # ══════════════════════════════════════════════════════════
    #  SINGLE MODEL MODE
    # ══════════════════════════════════════════════════════════
    if inf_mode == "🔮 Single Model Inference":
        # ── Model source + selection ────────────────────────────
        st.subheader("1 · Select Model")

        sources = []
        if saved_weight_names:
            sources.append("Saved weights (persistent across restarts)")
        if session_model_names:
            sources.append("Current session models")
    
        source = st.radio("Model source", sources, horizontal=True)
    
        if source == "Saved weights (persistent across restarts)":
            options = saved_weight_names.copy()
            if len(saved_weight_names) >= 2:
                options.append("Ensemble (soft vote all saved models)")
            model_choice = st.selectbox("Model", options,
                                        help="Models saved to saved_models/ with full weights")
    
            # Show what's available
            with st.expander("Saved model details", expanded=False):
                for n in saved_weight_names:
                    cfg_path = _MODELS_DIR / f'{n}_config.json'
                    if cfg_path.exists():
                        with open(cfg_path) as f:
                            cfg = json.load(f)
                        mdl = cfg['model']
                        note = cfg.get('note', '')
                        if mdl in ('VQC', 'HQNN'):
                            st.markdown(f"**{n.upper()}** — {mdl}, "
                                        f"{cfg['n_qubits']}q / {cfg['n_layers']} layers"
                                        + (f" · *{note}*" if note else ""))
                        else:
                            st.markdown(f"**{n.upper()}** — {mdl}, "
                                        f"{cfg['n_features']} feat / {cfg['n_classes']} classes"
                                        + (f" · *{note}*" if note else ""))
    
            # Check preprocessor — select the one that matches the chosen model's features
            main_pre_path = _MODELS_DIR / 'preprocessor.pkl'
            if not main_pre_path.exists():
                st.error(
                    "`saved_models/preprocessor.pkl` not found.  \n"
                    "Run `train_and_save.py` which saves the preprocessor alongside weights."
                )
                st.stop()
            # Use model-specific preprocessor if the model has different feature_cols
            _chosen = model_choice if "Ensemble" not in model_choice else saved_weight_names[0]
            pre = _load_preprocessor_for(_chosen)
            expected_features = pre['feature_cols']
            class_names_inf   = pre['class_names']
            scaler_inf        = pre['scaler']
            use_saved_weights = True
    
        else:
            options = session_model_names.copy()
            if len(session_model_names) >= 2:
                options.append("Ensemble (soft vote all session models)")
            model_choice = st.selectbox("Model", options,
                                        help="Models trained in this session (lost on restart)")
    
            pd_session = st.session_state.processed_data
            if pd_session is None:
                st.error("No preprocessing data found. Complete Step 2 first.")
                st.stop()
            expected_features = st.session_state.feature_cols
            class_names_inf   = pd_session['class_names']
            scaler_inf        = st.session_state.dp.scaler
            use_saved_weights = False
    
        # Show what raw columns the model needs (not derived feature names)
        if use_saved_weights:
            _kin_needed = any(c in (pre.get('feature_cols') or [])
                              for c in _KIN_STAT_COLS + ['sog'])
            _req_cols = _BASE_RAW_COLS + ['cross_range_extent or az_extent_m']
            if _kin_needed:
                _req_cols += _KIN_STAT_COLS
            st.info(
                f"**Required raw columns:** `{'`, `'.join(_req_cols)}`  \n"
                f"Derived features are computed automatically.  \n"
                f"**Output classes:** {', '.join(class_names_inf)}"
            )
        else:
            st.info(f"**Expected features:** {', '.join(expected_features)}  \n"
                    f"**Classes:** {', '.join(class_names_inf)}")

        # ── Upload new CSV ────────────────────────────────────────
        st.subheader("2 · Upload New Data")
        inf_file = st.file_uploader(
            "Upload any raw radar CSV — feature engineering runs automatically",
            type=["csv", "txt"],
            key="inference_upload",
        )

        if inf_file:
            try:
                dp_inf = DataProcessor()
                inf_df = dp_inf.load_csv(inf_file)
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
                st.stop()

            st.success(f"Loaded — {inf_df.shape[0]} rows × {inf_df.shape[1]} columns")
            with st.expander("Raw data preview", expanded=False):
                st.dataframe(inf_df.head(20))

            # ── Run inference ─────────────────────────────────────
            st.subheader("3 · Run Inference")
            if st.button("Run Inference", type="primary"):
                with st.spinner("Running ETL + model inference..."):
                    try:
                        is_ensemble_choice = "Ensemble" in model_choice

                        if use_saved_weights:
                            if is_ensemble_choice:
                                # All saved models share the same ETL per their preprocessor
                                probas = []
                                for n in saved_weight_names:
                                    _pre_n = _load_preprocessor_for(n)
                                    with open(_MODELS_DIR / f'{n}_config.json') as _f:
                                        _cfg_n = json.load(_f)
                                    X_inf, _, _rep = _etl_inference(inf_df, _cfg_n, _pre_n)
                                    m, cfg_m = _load_weights_model(n)
                                    probas.append(_run_any_inference(m, cfg_m, X_inf))
                                avg_proba = np.mean(probas, axis=0)
                                etl_report = _rep
                            else:
                                with open(_MODELS_DIR / f'{model_choice}_config.json') as _f:
                                    cfg_chosen = json.load(_f)
                                X_inf, df_clean, etl_report = _etl_inference(
                                    inf_df, cfg_chosen, pre)
                                m, cfg_m = _load_weights_model(model_choice)
                                avg_proba = _run_any_inference(m, cfg_m, X_inf)

                            # Show ETL report
                            if etl_report.get('conversions'):
                                for msg in etl_report['conversions']:
                                    st.info(f"ETL: {msg}")
                            if etl_report.get('warnings'):
                                for msg in etl_report['warnings']:
                                    st.warning(f"ETL: {msg}")
                            if etl_report.get('n_dropped', 0) > 0:
                                st.caption(
                                    f"Rows: {etl_report['n_input']} input → "
                                    f"{etl_report['n_output']} after cleaning "
                                    f"({etl_report['n_dropped']} dropped)"
                                )

                        else:
                            # Session models: use legacy path (scaler from session)
                            missing_cols = [c for c in expected_features
                                            if c not in inf_df.columns]
                            if missing_cols:
                                st.error(f"CSV missing columns: {missing_cols}")
                                st.stop()
                            X_df = inf_df[expected_features].copy()
                            nan_rows = X_df.isna().any(axis=1)
                            if nan_rows.sum() > 0:
                                X_df = X_df.fillna(X_df.median())
                                st.warning(f"⚠️ {nan_rows.sum()} rows imputed with column medians")
                            X_inf = X_df.values.astype(np.float32)
                            if scaler_inf is not None:
                                X_inf = scaler_inf.transform(X_inf)

                            if is_ensemble_choice:
                                probas = []
                                for n in session_model_names:
                                    entry = st.session_state.trained_models[n]
                                    sm = entry['model']
                                    if entry['is_torch']:
                                        probas.append(_run_torch_inference(sm, X_inf))
                                    else:
                                        probas.append(_run_sklearn_inference(sm, X_inf))
                                avg_proba = np.mean(probas, axis=0)
                            else:
                                entry = st.session_state.trained_models[model_choice]
                                sm = entry['model']
                                if entry['is_torch']:
                                    avg_proba = _run_torch_inference(sm, X_inf)
                                else:
                                    avg_proba = _run_sklearn_inference(sm, X_inf)
    
                        avg_proba   = np.atleast_2d(avg_proba)
                        pred_idx    = avg_proba.argmax(axis=1).ravel()
                        pred_labels = [class_names_inf[int(i)] for i in pred_idx]
                        confidence  = avg_proba.max(axis=1).ravel()
    
                        # Carry through useful identifier columns from the raw CSV
                        _id_candidates = ['ObjID', 'STATIONID', 'Type', 'range',
                                          'azimuth', 'datetime', 'timeformat',
                                          'RLatitude', 'RLongitude']
                        passthrough_cols = [c for c in _id_candidates if c in inf_df.columns]
                        prob_cols_out = [f'prob_{c}' for c in class_names_inf]

                        # Build result dataframe — passthrough cols first, then predictions
                        result_df = inf_df[passthrough_cols].reset_index(drop=True) if passthrough_cols else pd.DataFrame()
                        result_df.insert(0, 'predicted_class', pred_labels)
                        result_df.insert(1, 'confidence', confidence.round(4))
                        for i, cls in enumerate(class_names_inf):
                            result_df[f'prob_{cls}'] = avg_proba[:, i].round(4)
    
                        st.success(f"Inference complete — {len(avg_proba)} rows classified.")
    
                        if passthrough_cols:
                            st.info(f"ID / extra columns carried through: **{', '.join(passthrough_cols)}**")
    
                        # Summary metrics
                        from collections import Counter
                        counts_pred = Counter(pred_labels)
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Rows classified", len(inf_df))
                        mc2.metric("Avg confidence", f"{confidence.mean():.1%}")
                        mc3.metric("Low confidence (<50%)", int((confidence < 0.5).sum()))
    
                        # Prediction distribution
                        st.subheader("Prediction distribution")
                        dist_df = pd.DataFrame([
                            {'Class': c, 'Count': counts_pred.get(c, 0),
                             'Pct': f"{counts_pred.get(c, 0)/len(inf_df):.1%}"}
                            for c in class_names_inf
                        ]).set_index('Class')
                        st.dataframe(dist_df, use_container_width=True)
    
                        # Results table — show passthrough + prediction cols
                        st.subheader("Predictions")
                        display_cols = passthrough_cols + ['predicted_class', 'confidence'] + prob_cols_out
                        display_df = result_df[display_cols]
                        max_style_cells = 262144
                        if display_df.size <= max_style_cells:
                            st.dataframe(
                                display_df.style.background_gradient(subset=['confidence'], cmap='Greens'),
                                use_container_width=True,
                            )
                        else:
                            st.dataframe(display_df, use_container_width=True)
    
                        # Low-confidence rows
                        low_conf = result_df[confidence < 0.5]
                        if len(low_conf) > 0:
                            with st.expander(
                                f"⚠️ {len(low_conf)} low-confidence predictions (model uncertain)",
                                expanded=False,
                            ):
                                st.dataframe(low_conf[display_cols], use_container_width=True)
    
                        # ── Per-class prediction breakdown ────────────
                        st.subheader("Per-Class Prediction Breakdown")
                        try:
                            from sklearn.metrics import (f1_score as _f1_cls,
                                                         accuracy_score as _acc_cls)
                            pred_arr = np.array(pred_labels)
                            conf_arr = confidence.ravel()
                            pc_rows = []
                            for cls in class_names_inf:
                                mask = pred_arr == cls
                                n    = int(mask.sum())
                                avg_conf = float(conf_arr[mask].mean()) if n > 0 else 0.0
                                low_n    = int((conf_arr[mask] < 0.5).sum()) if n > 0 else 0
                                pc_rows.append({
                                    'Class': cls,
                                    'Predicted Count': n,
                                    'Predicted %': f"{n / len(pred_arr):.1%}",
                                    'Avg Confidence': f"{avg_conf:.1%}",
                                    'Low Conf (<50%)': low_n,
                                })

                            # If ground truth label column is present, compute accuracy & F1
                            gt_col = next(
                                (c for c in passthrough_cols
                                 if c.lower() in ('type', 'label', 'class', 'vessel_type', 'true_label')),
                                None,
                            )
                            if gt_col is not None:
                                def _norm_lbl(v):
                                    try:
                                        return str(int(float(str(v))))
                                    except (ValueError, OverflowError):
                                        return str(v)
                                true_arr = np.array([_norm_lbl(v) for v in result_df[gt_col].values])

                                overall_acc = _acc_cls(true_arr, pred_arr)
                                overall_f1  = _f1_cls(true_arr, pred_arr,
                                                      average='macro', zero_division=0)
                                st.info(f"Ground truth column **{gt_col}** detected — comparing predictions against labels.")
                                oa1, oa2, oa3 = st.columns(3)
                                oa1.metric("Overall Accuracy", f"{overall_acc:.1%}")
                                oa2.metric("Macro F1", f"{overall_f1:.1%}")
                                oa3.metric("Correct predictions",
                                           f"{int(overall_acc * len(true_arr))} / {len(true_arr)}")

                                for row in pc_rows:
                                    cls = row['Class']
                                    true_mask = true_arr == cls
                                    n_true = int(true_mask.sum())
                                    if n_true > 0:
                                        tp = int(((true_arr == cls) & (pred_arr == cls)).sum())
                                        row['Recall'] = f"{tp / n_true:.1%}"
                                    else:
                                        row['Recall'] = 'N/A'
                                    f1_val = _f1_cls(
                                        true_arr == cls, pred_arr == cls,
                                        average='binary', zero_division=0,
                                    )
                                    row['F1 Score'] = f"{f1_val:.1%}"

                            st.dataframe(
                                pd.DataFrame(pc_rows).set_index('Class'),
                                use_container_width=True,
                            )
                        except Exception as _pc_err:
                            st.warning(f"Per-class breakdown skipped: {_pc_err}")
    
                        # Download — all columns including passthrough
                        csv_bytes = result_df.to_csv(index=False).encode()
                        st.download_button(
                            "⬇ Download Predictions CSV",
                            data=csv_bytes,
                            file_name=f"inference_{model_choice.replace(' ', '_')}.csv",
                            mime='text/csv',
                        )
    
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        raise
    
    # ══════════════════════════════════════════════════════════
    #  BENCHMARK MODE — run ALL saved models and compare
    # ══════════════════════════════════════════════════════════
    else:
        if not saved_weight_names:
            st.error("No saved models found. Train and save models first.")
            st.stop()

        pre_path = _MODELS_DIR / 'preprocessor.pkl'
        if not pre_path.exists():
            st.error("`saved_models/preprocessor.pkl` not found.")
            st.stop()
        with open(pre_path, 'rb') as f:
            pre = pickle.load(f)
        expected_features = pre['feature_cols']
        class_names_bm    = pre['class_names']
        scaler_bm         = pre['scaler']

        # Filter benchmark to models whose feature_cols match this preprocessor
        compatible_bm, incompatible_bm = [], []
        for n in saved_weight_names:
            cfg_p = _MODELS_DIR / f'{n}_config.json'
            if cfg_p.exists():
                with open(cfg_p) as f:
                    _c = json.load(f)
                fc = _c.get('feature_cols') or []
                if fc and list(fc) != list(expected_features):
                    incompatible_bm.append((n, fc))
                    continue
            compatible_bm.append(n)
        saved_weight_names = compatible_bm   # restrict benchmark to compatible models

        if incompatible_bm:
            st.warning(
                "⚠️ The following models use **different features** and are excluded from "
                "benchmark (run them individually in Single Model mode):  \n"
                + "  \n".join(f"**{n}** — expects: `{', '.join(fc)}`"
                              for n, fc in incompatible_bm)
            )

        st.info(
            f"**{len(saved_weight_names)} compatible models:** "
            + ", ".join(f"`{n}`" for n in saved_weight_names)
        )
        st.info(f"**Features:** {', '.join(expected_features)}  \n"
                f"**Classes:** {', '.join(class_names_bm)}")

        # ── Upload CSV ────────────────────────────────────────
        st.subheader("1 · Upload Data")
        bm_file = st.file_uploader(
            "Upload CSV with radar features",
            type=["csv", "txt"],
            key="benchmark_upload",
        )
        label_col_bm = st.text_input(
            "Label column (optional — leave blank for unlabelled data)",
            value="Type",
            key="bm_label_col",
        ).strip()

        if bm_file:
            try:
                dp_bm  = DataProcessor()
                bm_df  = dp_bm.load_csv(bm_file)
            except Exception as e:
                st.error(f"Failed to load CSV: {e}")
                st.stop()

            missing = [c for c in expected_features if c not in bm_df.columns]
            if missing:
                st.error(f"Missing columns: **{missing}**")
                st.stop()

            has_labels = bool(label_col_bm) and label_col_bm in bm_df.columns
            if label_col_bm and not has_labels:
                st.warning(f"Column `{label_col_bm}` not found — running without labels.")

            st.success(f"Loaded — {bm_df.shape[0]} rows. "
                       + ("Labels found." if has_labels else "No label column."))

            # Scale features
            X_bm = bm_df[expected_features].fillna(0).values.astype(np.float32)
            X_bm = scaler_bm.transform(X_bm)

            # Encode true labels if available
            y_bm = None
            if has_labels:
                from sklearn.preprocessing import LabelEncoder as _LE
                le_bm = _LE()
                le_bm.classes_ = np.array(class_names_bm)
                # Normalise float-formatted integers: '30.0' → '30'
                def _norm_label(v):
                    s = str(v)
                    try:
                        return str(int(float(s)))
                    except (ValueError, OverflowError):
                        return s
                raw_labels = np.array([_norm_label(v) for v in bm_df[label_col_bm].values])
                valid_mask = np.isin(raw_labels, class_names_bm)
                if valid_mask.sum() < len(raw_labels):
                    st.warning(f"{(~valid_mask).sum()} rows have unknown labels — excluded from metrics.")
                y_bm = np.array([class_names_bm.index(l) if l in class_names_bm else -1
                                 for l in raw_labels])

            # ── Run all models ────────────────────────────────
            st.subheader("2 · Run Benchmark")
            if st.button("🏆 Run All Models", type="primary"):
                all_probas  = {}
                all_preds   = {}
                load_errors = {}

                prog = st.progress(0.0)
                status = st.empty()
                for i, mname in enumerate(saved_weight_names):
                    status.text(f"Running {mname} ({i+1}/{len(saved_weight_names)})…")
                    try:
                        m_obj, cfg_m = _load_weights_model(mname)
                        proba = _run_any_inference(m_obj, cfg_m, X_bm)
                        proba = np.atleast_2d(proba)  # guarantee 2D
                        all_probas[mname] = proba
                        all_preds[mname]  = proba.argmax(axis=1).ravel()  # guarantee 1D
                    except Exception as e:
                        load_errors[mname] = str(e)
                    prog.progress((i + 1) / len(saved_weight_names))

                status.success(f"Done — {len(all_probas)} models ran successfully.")
                if load_errors:
                    st.warning("Failed to load: " + ", ".join(
                        f"`{k}`: {v}" for k, v in load_errors.items()))

                if not all_probas:
                    st.error("No models produced results.")
                    st.stop()

                # ── Summary metrics table ─────────────────────
                st.subheader("Overall Performance")
                from sklearn.metrics import (accuracy_score as _acc,
                                             f1_score as _f1,
                                             roc_auc_score as _auc)
                summary_rows = []
                for mname, proba in all_probas.items():
                    preds = all_preds[mname]
                    conf  = proba.max(axis=1).ravel()
                    row = {
                        'Model': mname,
                        'Avg Conf': f"{conf.mean():.1%}",
                        'Low Conf (<50%)': int((conf < 0.5).sum()),
                    }
                    if has_labels and y_bm is not None:
                        valid = y_bm.ravel() >= 0
                        if valid.sum() > 0:
                            yv, pv, prbv = y_bm.ravel()[valid], preds.ravel()[valid], proba[valid]
                            row['Accuracy']  = f"{_acc(yv, pv):.1%}"
                            row['F1 Macro']  = f"{_f1(yv, pv, average='macro', zero_division=0):.1%}"
                            try:
                                row['AUC'] = f"{_auc(yv, prbv, multi_class='ovr', average='macro'):.3f}"
                            except Exception:
                                row['AUC'] = 'N/A'
                            ece_v, cc, cw = _ece_calc(prbv, yv)
                            row['ECE']           = f"{ece_v:.4f}"
                            row['Conf Correct']  = f"{cc:.3f}"
                            row['Conf Wrong']    = f"{cw:.3f}"
                    summary_rows.append(row)

                st.dataframe(pd.DataFrame(summary_rows).set_index('Model'),
                             use_container_width=True)

                # ── Per-class recall heatmap ──────────────────
                if has_labels and y_bm is not None:
                    st.subheader("Per-Class Recall")
                    from sklearn.metrics import recall_score as _recall
                    recall_rows = []
                    valid = y_bm.ravel() >= 0
                    for mname in all_probas:
                        pv   = all_preds[mname].ravel()[valid]
                        yv   = y_bm.ravel()[valid]
                        recs = _recall(yv, pv, average=None,
                                       labels=list(range(len(class_names_bm))),
                                       zero_division=0)
                        recall_rows.append({'Model': mname,
                                            **{cn: f"{r:.1%}" for cn, r in
                                               zip(class_names_bm, recs)}})
                    st.dataframe(pd.DataFrame(recall_rows).set_index('Model'),
                                 use_container_width=True)

                # ── Prediction agreement ──────────────────────
                st.subheader("Prediction Agreement Across Models")
                agree_df = pd.DataFrame(
                    {n: [class_names_bm[int(p)] for p in all_preds[n].ravel()]
                     for n in all_probas},
                    index=bm_df.index,
                )
                modal_pred = agree_df.mode(axis=1)[0]
                agreement  = (agree_df == modal_pred.values[:, None]).mean(axis=1)
                agree_df.insert(0, 'consensus', modal_pred)
                agree_df.insert(1, 'agreement_%', (agreement * 100).round(1))
                low_agree = agree_df['agreement_%'] < 60
                st.metric("Rows with full consensus (100% agreement)",
                          int((agreement == 1.0).sum()))
                st.metric("Rows with low agreement (<60%)", int(low_agree.sum()))

                with st.expander("Full agreement table", expanded=False):
                    st.dataframe(
                        agree_df.style.background_gradient(
                            subset=['agreement_%'], cmap='RdYlGn'),
                        use_container_width=True,
                    )

                # ── Per-class prediction breakdown (benchmark) ───
                st.subheader("Per-Class Prediction Breakdown")
                try:
                    from sklearn.metrics import f1_score as _f1_bm
                    bm_pred_rows = []
                    # Pre-compute label filter once (not inside the class loop)
                    bm_valid = (y_bm >= 0).ravel() if (has_labels and y_bm is not None) else None
                    bm_true_arr = (
                        np.array([class_names_bm[int(y)] for y in y_bm.ravel()[bm_valid]])
                        if bm_valid is not None else None
                    )
                    for cls in class_names_bm:
                        row = {'Class': cls}
                        for mname, proba in all_probas.items():
                            preds_1d  = all_preds[mname].ravel()
                            preds_m   = np.array([class_names_bm[int(p)] for p in preds_1d])
                            mask      = preds_m == cls
                            n         = int(mask.sum())
                            conf_1d   = proba.max(axis=1).ravel()
                            avg_conf  = float(conf_1d[mask].mean()) if n > 0 else 0.0
                            row[f'{mname} Count'] = n
                            row[f'{mname} Conf']  = f"{avg_conf:.1%}"
                        if bm_valid is not None and bm_true_arr is not None:
                            for mname in all_probas:
                                preds_1d_v = all_preds[mname].ravel()[bm_valid]
                                pred_m_valid = np.array([class_names_bm[int(p)]
                                                          for p in preds_1d_v])
                                tp = int(((bm_true_arr == cls) & (pred_m_valid == cls)).sum())
                                n_true = int((bm_true_arr == cls).sum())
                                row[f'{mname} Recall'] = (
                                    f"{tp / n_true:.1%}" if n_true > 0 else 'N/A'
                                )
                                f1_val = _f1_bm(
                                    bm_true_arr == cls, pred_m_valid == cls,
                                    average='binary', zero_division=0,
                                )
                                row[f'{mname} F1'] = f"{f1_val:.1%}"
                        bm_pred_rows.append(row)
                except Exception as _bm_pc_err:
                    st.warning(f"Per-class breakdown skipped: {_bm_pc_err}")
                    bm_pred_rows = []

                with st.expander("Per-class breakdown — all models", expanded=True):
                    st.dataframe(
                        pd.DataFrame(bm_pred_rows).set_index('Class'),
                        use_container_width=True,
                    )

                # ── Download wide CSV ─────────────────────────
                st.subheader("Download Results")
                wide_df = bm_df[expected_features].copy()
                if has_labels:
                    wide_df['true_label'] = bm_df[label_col_bm]
                wide_df['consensus']    = modal_pred.values
                wide_df['agreement_%'] = (agreement * 100).round(1)
                for mname, proba in all_probas.items():
                    wide_df[f'pred_{mname}'] = [class_names_bm[int(p)] for p in all_preds[mname].ravel()]
                    wide_df[f'conf_{mname}'] = proba.max(axis=1).ravel().round(4)
                csv_bytes = wide_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download Benchmark CSV (all model predictions)",
                    data=csv_bytes,
                    file_name="benchmark_all_models.csv",
                    mime='text/csv',
                )
