#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trial-level mixed models with RT covariate (per feature)

Model (per ROI×band feature):
    log1p(power) ~ role_code + cond_code + role_code:cond_code + zRT
                   + (1 + zRT | subject)

Where:
- role_code: Leader=+0.5, Follower=-0.5
- cond_code: Matching (MA)=+0.5, Mirroring (MR)=-0.5
- zRT: RT z-scored within subject (across all their trials)
- subject groups random intercept; try random slope for zRT, fallback to intercept-only.

Inputs
------
- --eeg-dir: directory of *_parcellated_power.pkl files
- --matched-csv: matched RT–EEG index (participant_id, cell, trial_index, rt)
- --output-dir: new directory for results; existing directories are refused

Outputs
-------
- mixedlm_rt_cov_results.csv         # per feature × effect; Wald & LRT p-values
- mixedlm_rt_cov_results_fdr.csv     # p-values with FDR (BH) per effect
- brief console summary
"""

import argparse
import os
import re
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from glob import glob

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
import warnings
warnings.simplefilter("ignore", ConvergenceWarning)

# ============================ CONFIG ==========================================
parser = argparse.ArgumentParser(description="Fit trial-level RT-adjusted tonic EEG models.")
parser.add_argument("--eeg-dir", required=True, type=Path,
                    help="Directory containing *_parcellated_power.pkl files")
parser.add_argument("--matched-csv", required=True, type=Path,
                    help="Matched RT–EEG index CSV")
parser.add_argument("--output-dir", required=True, type=Path,
                    help="New output directory; must not already exist")
args = parser.parse_args()

EEG_DIR = args.eeg_dir
MATCHED = args.matched_csv
OUT_DIR = args.output_dir
if not EEG_DIR.is_dir():
    parser.error(f"EEG directory does not exist: {EEG_DIR}")
if not MATCHED.is_file():
    parser.error(f"Matched CSV does not exist: {MATCHED}")
if OUT_DIR.exists():
    parser.error(f"Output path already exists; choose a new directory: {OUT_DIR}")
OUT_DIR.mkdir(parents=True)

# Max iterations for optimizer
MAXITER = 200

# ============================ HELPERS =========================================
def norm_role(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("l"): return "Leader"
    if s.startswith("f"): return "Follower"
    return x

def norm_condition(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("match") or s == "ma": return "Matching"
    if s.startswith("mirror") or s == "mr": return "Mirroring"
    return x

def cell_code(role: str, cond: str) -> str:
    r = "L" if role == "Leader" else "F"
    c = "MA" if cond == "Matching" else "MR"
    return f"{r}{c}"

def norm_pid(pid: str) -> str:
    s = str(pid).strip()
    m = re.search(r'(\d+)_P([12])', s, flags=re.IGNORECASE)
    if m: return f"{int(m.group(1)):02d}_P{m.group(2)}"
    m = re.search(r'dyad\s*(\d+)\s*_?P([12])', s, flags=re.IGNORECASE)
    if m: return f"{int(m.group(1)):02d}_P{m.group(2)}"
    return s

def feature_names_band_major(bands, rois):
    return [f"{roi} | {band}" for band in bands for roi in rois]

def flatten_band_power(bp: np.ndarray) -> np.ndarray:
    # band-major flatten consistent with previous steps
    return bp.reshape(-1, order="C")

def effect_codes(cell: str):
    # LMR, LMA, FMR, FMA
    cell = str(cell).upper().strip()
    role_code = +0.5 if cell.startswith("L") else -0.5
    cond_code = +0.5 if cell.endswith("MA") else -0.5
    return role_code, cond_code

def zscore_within_subject(df, col, by="participant_id"):
    g = df.groupby(by)[col]
    mu = g.transform("mean")
    sd = g.transform("std")
    sd = sd.replace(0, np.nan)
    z = (df[col] - mu) / sd
    z = z.fillna(0.0)
    return z

# ============================ LOAD MATCHED =====================================
print(f"[INFO] Loading matched: {MATCHED}")
matched = pd.read_csv(MATCHED)  # participant_id, cell, trial_index, rt
if matched.empty:
    raise RuntimeError("matched_rt_eeg_index.csv is empty.")
matched["participant_id"] = matched["participant_id"].astype(str)
matched["cell"] = matched["cell"].astype(str).str.upper().str.strip()
matched["rt"] = pd.to_numeric(matched["rt"], errors="coerce")
matched = matched.dropna(subset=["rt"]).reset_index(drop=True)
matched["rt_z"] = zscore_within_subject(matched, "rt", by="participant_id")

# ======================= DISCOVER LABELS =======================================
pkl_files = sorted(glob(os.path.join(EEG_DIR, "*_parcellated_power.pkl")))
if not pkl_files:
    raise FileNotFoundError("No *_parcellated_power.pkl found in EEG_DIR.")
band_labels = roi_labels = None
for p in pkl_files:
    with open(p, "rb") as f:
        trials = pickle.load(f)
    if isinstance(trials, list) and trials:
        bp = trials[0].get("band_power", None)
        if isinstance(bp, np.ndarray) and bp.ndim == 2:
            band_labels = list(trials[0]["band_labels"])
            roi_labels  = list(trials[0]["roi_labels"])
            break
if band_labels is None or roi_labels is None:
    raise RuntimeError("Could not read band/ROI labels.")

B, R = len(band_labels), len(roi_labels)
N_FEATURES = B * R
FEATURES = feature_names_band_major(band_labels, roi_labels)
print(f"[INFO] Features: {N_FEATURES} = {B} bands × {R} rois")

# ======================= BUILD TRIAL MATRIX (Y) ================================
# We build arrays across all matched trials:
Y_rows = []          # list of (408,) float arrays (log transform will be applied later)
meta_rows = []       # dicts with participant_id, cell, trial_index, rt_z, role_code, cond_code

CELL_ORDER = ["LMR","LMA","FMR","FMA"]

for p in pkl_files:
    with open(p, "rb") as f:
        trials = pickle.load(f)
    if not isinstance(trials, list) or len(trials) == 0:
        continue

    pid_raw = trials[0].get("participant_id", os.path.basename(p).replace("_parcellated_power.pkl",""))
    pid = norm_pid(pid_raw)

    # build df of EEG trials for this participant, with cell + trial_index
    rows = []
    for tr in trials:
        bp = tr.get("band_power", None)
        if not (isinstance(bp, np.ndarray) and bp.shape == (B, R)):
            continue
        role = norm_role(tr.get("role"))
        cond = norm_condition(tr.get("condition"))
        cell = cell_code(role, cond)  # canonical
        rows.append({"cell": cell, "bp": bp})
    if not rows:
        continue

    df_eeg = pd.DataFrame(rows)
    df_eeg = df_eeg[df_eeg["cell"].isin(CELL_ORDER)].copy()
    if df_eeg.empty:
        continue
    df_eeg["trial_index"] = df_eeg.groupby("cell").cumcount()

    # inner-join with matched to get RT z and aligned trials
    m_pid = matched[matched["participant_id"] == pid]
    if m_pid.empty:
        continue

    for cell in CELL_ORDER:
        eeg_cell = df_eeg[df_eeg["cell"] == cell][["trial_index","bp"]]
        if eeg_cell.empty:
            continue
        join = pd.merge(
            eeg_cell, m_pid[m_pid["cell"] == cell][["trial_index","rt_z"]],
            on="trial_index", how="inner", validate="one_to_one"
        )
        if join.empty:
            continue
        # append rows
        rc, cc = effect_codes(cell)
        for _, r in join.iterrows():
            Y_rows.append(flatten_band_power(r["bp"]))  # raw; we'll log1p later
            meta_rows.append({
                "participant_id": pid,
                "cell": cell,
                "trial_index": int(r["trial_index"]),
                "rt_z": float(r["rt_z"]),
                "role_code": rc,
                "cond_code": cc,
            })

if not Y_rows:
    raise RuntimeError("No aligned trials assembled for mixed models.")
Y = np.vstack(Y_rows)  # (n_trials, 408)
meta = pd.DataFrame(meta_rows)
print(f"[INFO] Trial matrix: Y={Y.shape}, unique subjects={meta['participant_id'].nunique()}")

# ======================= FIT MIXED MODELS PER FEATURE =========================
# Use log1p transform for robustness (trial-level).
meta_base = meta.copy()
meta_base.rename(columns={"participant_id":"subject"}, inplace=True)

results_rows = []

def fit_and_collect(y_vec, feature_name):
    # Data for this feature
    df = meta_base.copy()
    df["y"] = np.log1p(y_vec)  # log1p power
    # Full model: random intercept + random slope for zRT
    tried = []
    for re_formula in ["1 + rt_z", "1"]:
        try:
            model = smf.mixedlm(
                "y ~ role_code + cond_code + role_code:cond_code + rt_z",
                data=df, groups=df["subject"],
                re_formula=re_formula
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=MAXITER, disp=False)
            tried.append((re_formula, fit))
            full_fit = fit
            full_re  = re_formula
            break
        except Exception:
            continue
    if not tried:
        return [  # return NAs for this feature
            {"feature": feature_name, "effect":"role",           "param":"role_code",
             "coef":np.nan, "z":np.nan, "p_wald":np.nan, "lrt_chi2":np.nan, "lrt_df":np.nan, "p_lrt":np.nan,
             "aic":np.nan, "bic":np.nan, "re_formula":"NA"},
            {"feature": feature_name, "effect":"condition",      "param":"cond_code",
             "coef":np.nan, "z":np.nan, "p_wald":np.nan, "lrt_chi2":np.nan, "lrt_df":np.nan, "p_lrt":np.nan,
             "aic":np.nan, "bic":np.nan, "re_formula":"NA"},
            {"feature": feature_name, "effect":"role:condition", "param":"role_code:cond_code",
             "coef":np.nan, "z":np.nan, "p_wald":np.nan, "lrt_chi2":np.nan, "lrt_df":np.nan, "p_lrt":np.nan,
             "aic":np.nan, "bic":np.nan, "re_formula":"NA"},
            {"feature": feature_name, "effect":"rt_z",           "param":"rt_z",
             "coef":np.nan, "z":np.nan, "p_wald":np.nan, "lrt_chi2":np.nan, "lrt_df":np.nan, "p_lrt":np.nan,
             "aic":np.nan, "bic":np.nan, "re_formula":"NA"},
        ]

    # Wald tests (from full model)
    pv = full_fit.pvalues
    tv = full_fit.tvalues
    co = full_fit.params
    aic = full_fit.aic
    bic = full_fit.bic

    def safe(s): return float(s) if s is not None and np.isfinite(s) else np.nan

    rows = []
    for eff, param in [
        ("role", "role_code"),
        ("condition", "cond_code"),
        ("role:condition", "role_code:cond_code"),
        ("rt_z", "rt_z"),
    ]:
        rows.append({
            "feature": feature_name,
            "effect": eff,
            "param": param,
            "coef": safe(co.get(param, np.nan)),
            "z":    safe(tv.get(param, np.nan)),
            "p_wald": safe(pv.get(param, np.nan)),
            "lrt_chi2": np.nan, "lrt_df": np.nan, "p_lrt": np.nan,
            "aic": aic, "bic": bic, "re_formula": full_re,
        })

    # Likelihood-ratio tests: drop each fixed effect in turn
    def lrt(drop_param):
        # Build reduced formula (remove term)
        terms = ["role_code", "cond_code", "role_code:cond_code", "rt_z"]
        keep = [t for t in terms if t != drop_param]
        formula = "y ~ " + " + ".join(keep)
        try:
            red = smf.mixedlm(formula, data=df, groups=df["subject"], re_formula=full_re)
            red_fit = red.fit(reml=False, method="lbfgs", maxiter=MAXITER, disp=False)
            chi2_stat = 2.0 * (full_fit.llf - red_fit.llf)
            df_diff = red_fit.df_modelwc - full_fit.df_modelwc
            # In MixedLM, dropping one fixed effect ~ df=1
            df_use = abs(int(df_diff)) if df_diff != 0 else 1
            pval = chi2.sf(chi2_stat, df_use)
            return float(chi2_stat), float(df_use), float(pval)
        except Exception:
            return np.nan, np.nan, np.nan

    for r in rows:
        chi2_stat, df_use, pval = lrt(r["param"])
        r["lrt_chi2"] = chi2_stat
        r["lrt_df"]   = df_use
        r["p_lrt"]    = pval

    return rows

# Fit per feature
all_rows = []
for i, fname in enumerate(FEATURES, 1):
    rows = fit_and_collect(Y[:, i-1] if i>0 else Y[:,0], fname)
    all_rows.extend(rows)
    if i % 25 == 0:
        print(f"[..] Fitted {i}/{N_FEATURES} features")

res_df = pd.DataFrame(all_rows)

# ======================= FDR per effect/type ==================================
# Compute BH-FDR separately for each effect and p-value type
out = []
for effect in ["role", "condition", "role:condition", "rt_z"]:
    sub = res_df[res_df["effect"] == effect].copy()
    if sub.empty:
        continue
    for pcol, newcol in [("p_wald","p_wald_fdr"), ("p_lrt","p_lrt_fdr")]:
        if sub[pcol].notna().any():
            p = sub[pcol].to_numpy()
            mask = np.isfinite(p)
            q = np.full_like(p, np.nan, dtype=float)
            if mask.sum() > 0:
                _, qvals, _, _ = multipletests(p[mask], method="fdr_bh")
                q[mask] = qvals
            sub[newcol] = q
            sub[newcol+"_sig"] = sub[newcol] < 0.05
    out.append(sub)

res_fdr = pd.concat(out, ignore_index=True) if out else res_df.copy()

# Save
csv1 = Path(OUT_DIR) / "mixedlm_rt_cov_results.csv"
csv2 = Path(OUT_DIR) / "mixedlm_rt_cov_results_fdr.csv"
res_df.to_csv(csv1, index=False)
res_fdr.to_csv(csv2, index=False)
print(f"[SAVED] {csv1}")
print(f"[SAVED] {csv2}")

# Console summary
def count_sig(df, col):
    return (df[col] < 0.05).groupby(df["effect"]).sum(min_count=1)

print("\n[SUMMARY] FDR<.05 counts (by effect)")
if "p_wald_fdr" in res_fdr:
    print("  Wald:"); print(count_sig(res_fdr, "p_wald_fdr").to_string())
if "p_lrt_fdr" in res_fdr:
    print("  LRT :"); print(count_sig(res_fdr, "p_lrt_fdr").to_string())

print("\nDone.")
