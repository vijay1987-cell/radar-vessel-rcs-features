"""
Synthetic verification that log(Ap) + 4*log(R) eliminates range dependence.
Reproduces the result stated in Paper 5, Section VI-C.
No proprietary data required — runs on synthetic data only.
"""
import numpy as np
from scipy.stats import pearsonr

RNG = np.random.default_rng(42)
N = 2000

range_m  = RNG.uniform(5e3, 50e3, N)
true_rcs = RNG.lognormal(mean=2.0, sigma=1.0, size=N)   # range-independent truth
raw_amp  = true_rcs / range_m**4                          # simulate R^-4 propagation loss

r_raw,  p_raw  = pearsonr(np.log(raw_amp), np.log(range_m))
log_rcs         = np.log(raw_amp) + 4 * np.log(range_m)
r_corr, p_corr = pearsonr(log_rcs, np.log(range_m))

print(f"Before correction: r = {r_raw:.4f}  (p = {p_raw:.2e})")
print(f"After  correction: r = {r_corr:.4f}  (p = {p_corr:.2f})")
assert abs(r_corr) < 0.05, "Correction failed — check implementation"
print("PASS: range dependence eliminated.")
