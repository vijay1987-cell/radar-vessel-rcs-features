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
    R  = df['range'].clip(lower=1.0)
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
