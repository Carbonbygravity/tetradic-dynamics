"""
Script: 01_jaccard_role_invariance_sensitivity.py

Purpose
-------
Assess the robustness of tonic condition effects across roles by refitting the
mixed-effects model independently within Leader and Follower subsets, then
quantifying ROI-level overlap using the Jaccard index.

Pipeline role
-------------
Supplementary sensitivity analysis. This script tests whether tonic condition
effects remain similar when estimated independently within each role rather than
derived from a pooled model.

Model within each role
----------------------
For each ROI × band feature:

    log1p(power) ~ cond_code + rt_z + (1 + rt_z | subject)

If the random-slope model fails, the script falls back to:

    log1p(power) ~ cond_code + rt_z + (1 | subject)

Coding
------
- cond_code: Matching = +0.5, Mirroring = -0.5
- rt_z: within-subject z-scored RT across all aligned trials

Inputs
------
1. Per-participant parcellated power files.
2. matched_rt_eeg_index.csv containing:
   participant_id, cell, trial_index, rt

Outputs
-------
1. independent_role_jaccard_summary.csv
   Summary overlap table for:
   - Leader vs Follower, Matching > Mirroring
   - Leader vs Follower, Mirroring > Matching

2. Optional top-shared-ROI tables, if available:
   - Matching_vs_Mirroring_top_shared_roi_diffs.csv
   - Mirroring_vs_Matching_top_shared_roi_diffs.csv

Notes
-----
- ROI overlap is computed after FDR correction within each role.
- Direction is determined from the sign of the condition coefficient.
- Jaccard is computed at the ROI level, not at the feature level.
"""

import argparse
from pathlib import Path
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2

warnings.simplefilter("ignore", ConvergenceWarning)

# =============================================================================
# User settings
# =============================================================================

parser = argparse.ArgumentParser(description="Run role-specific tonic Jaccard sensitivity.")
parser.add_argument("--eeg-dir", type=Path, required=True)
parser.add_argument("--matched-csv", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
EEG_DIR = args.eeg_dir
MATCHED_CSV = args.matched_csv
OUT_DIR = args.output_dir
if OUT_DIR.exists():
    raise FileExistsError(f"Output directory already exists: {OUT_DIR}")
OUT_DIR.mkdir(parents=True)

ALPHA_FDR = 0.05
MAXITER = 200
TOP_N_SHARED = 15
SAVE_TOP_SHARED_TABLES = True

CELL_ORDER = ["LMR", "LMA", "FMR", "FMA"]

# =============================================================================
# Helper functions
# =============================================================================

def norm_role(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("l"):
        return "Leader"
    if s.startswith("f"):
        return "Follower"
    return str(x)

def norm_condition(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("match") or s == "ma":
        return "Matching"
    if s.startswith("mirror") or s == "mr":
        return "Mirroring"
    return str(x)

def cell_code(role: str, cond: str) -> str:
    r = "L" if role == "Leader" else "F"
    c = "MA" if cond == "Matching" else "MR"
    return f"{r}{c}"

def norm_pid(pid: str) -> str:
    s = str(pid).strip()
    m = re.search(r"(\d+)_P([12])", s, flags=re.IGNORECASE)
    if m:
        return f"{int(m.group(1)):02d}_P{m.group(2)}"
    m = re.search(r"dyad\s*(\d+)\s*_?P([12])", s, flags=re.IGNORECASE)
    if m:
        return f"{int(m.group(1)):02d}_P{m.group(2)}"
    return s

def feature_names_band_major(bands, rois):
    return [f"{roi} | {band}" for band in bands for roi in rois]

def flatten_band_power(bp: np.ndarray) -> np.ndarray:
    return bp.reshape(-1, order="C")

def effect_codes_from_cell(cell: str):
    cell = str(cell).upper().strip()
    role_code = +0.5 if cell.startswith("L") else -0.5
    cond_code = +0.5 if cell.endswith("MA") else -0.5
    return role_code, cond_code

def zscore_within_subject(df: pd.DataFrame, col: str, by: str = "participant_id"):
    g = df.groupby(by)[col]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    z = (df[col] - mu) / sd
    return z.fillna(0.0)

def parse_feature_to_roi_band(feature: str):
    parts = [p.strip() for p in str(feature).split("|")]
    if len(parts) == 2:
        return parts[0], parts[1].lower()
    return str(feature).strip(), ""

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    _, qvals, _, _ = multipletests(p[ok], alpha=ALPHA_FDR, method="fdr_bh")
    q[ok] = qvals
    return q

def cond_direction_from_coef(coef: float) -> str:
    if not np.isfinite(coef):
        return "NA"
    if coef > 0:
        return "Matching > Mirroring"
    if coef < 0:
        return "Mirroring > Matching"
    return "NA"

def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def fit_role_specific_feature(df_role: pd.DataFrame, y_vec: np.ndarray, feature_name: str, role_name: str):
    """
    Fit one feature within one role:
        y ~ cond_code + rt_z + (1 + rt_z | subject)
    fallback -> random intercept only
    """
    d = df_role.copy()
    d["y"] = np.log1p(y_vec)

    full_fit = None
    re_formula_used = None

    for re_formula in ["1 + rt_z", "1"]:
        try:
            model = smf.mixedlm(
                "y ~ cond_code + rt_z",
                data=d,
                groups=d["subject"],
                re_formula=re_formula
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=MAXITER, disp=False)
            full_fit = fit
            re_formula_used = re_formula
            break
        except Exception:
            continue

    if full_fit is None:
        return {
            "role_subset": role_name,
            "feature": feature_name,
            "coef": np.nan,
            "p_lrt": np.nan,
            "re_formula": "NA",
            "n_trials": len(d),
            "n_subjects": d["subject"].nunique(),
        }

    co = full_fit.params

    try:
        red = smf.mixedlm(
            "y ~ rt_z",
            data=d,
            groups=d["subject"],
            re_formula=re_formula_used
        )
        red_fit = red.fit(reml=False, method="lbfgs", maxiter=MAXITER, disp=False)
        lrt_chi2 = 2.0 * (full_fit.llf - red_fit.llf)
        df_diff = red_fit.df_modelwc - full_fit.df_modelwc
        lrt_df = abs(int(df_diff)) if df_diff != 0 else 1
        p_lrt = chi2.sf(lrt_chi2, lrt_df)
    except Exception:
        p_lrt = np.nan

    return {
        "role_subset": role_name,
        "feature": feature_name,
        "coef": safe_float(co.get("cond_code", np.nan)),
        "p_lrt": safe_float(p_lrt),
        "re_formula": re_formula_used,
        "n_trials": len(d),
        "n_subjects": d["subject"].nunique(),
    }

def add_fdr_columns(res_df: pd.DataFrame) -> pd.DataFrame:
    out = res_df.copy()
    out["p_lrt_fdr"] = bh_fdr(out["p_lrt"].to_numpy())
    out["p_lrt_fdr_sig"] = out["p_lrt_fdr"] < ALPHA_FDR
    return out

def survivors_by_direction(res_fdr: pd.DataFrame) -> pd.DataFrame:
    sub = res_fdr[np.isfinite(res_fdr["p_lrt_fdr"]) & (res_fdr["p_lrt_fdr"] < ALPHA_FDR)].copy()
    if sub.empty:
        return sub
    parsed = sub["feature"].apply(parse_feature_to_roi_band)
    sub["roi_full"] = [r for r, _ in parsed]
    sub["band"] = [b for _, b in parsed]
    sub["cond_direction"] = sub["coef"].apply(cond_direction_from_coef)
    return sub

def panel_signature(rows: pd.DataFrame):
    """
    Per direction:
    - ROI set
    - per-ROI representative coefficient = coefficient with max |coef| across bands
    """
    if rows.empty:
        return set(), {}

    rr = rows.copy()
    rr["roi_full"] = rr["roi_full"].str.lower()
    rr["coef"] = pd.to_numeric(rr["coef"], errors="coerce").astype(float)
    rr = rr[np.isfinite(rr["coef"])].copy()

    if rr.empty:
        return set(), {}

    rois = set(rr["roi_full"].unique())
    rr["abs_coef"] = rr["coef"].abs()
    idx = rr.groupby("roi_full")["abs_coef"].idxmax()
    roi_coef = rr.loc[idx, ["roi_full", "coef"]].set_index("roi_full")["coef"].to_dict()
    return rois, roi_coef

def compare_panels(sig_dict: dict, a: str, b: str):
    rois_a, coef_a = sig_dict[a]
    rois_b, coef_b = sig_dict[b]

    union = rois_a | rois_b
    inter = rois_a & rois_b
    jacc = len(inter) / max(1, len(union))

    common = sorted(list(inter))
    coef_diff = [abs(coef_a[r] - coef_b[r]) for r in common if r in coef_a and r in coef_b]

    out = {
        "panel_a": a,
        "panel_b": b,
        "n_rois_a": len(rois_a),
        "n_rois_b": len(rois_b),
        "n_union": len(union),
        "n_intersection": len(inter),
        "jaccard_roi_overlap": float(jacc),
        "n_comparable_common_rois": len(coef_diff),
        "median_abs_coef_diff": float(np.median(coef_diff)) if coef_diff else np.nan,
        "max_abs_coef_diff": float(np.max(coef_diff)) if coef_diff else np.nan,
    }

    top_df = None
    if common:
        rows = []
        for r in common:
            if r in coef_a and r in coef_b:
                rows.append({
                    "roi_full": r,
                    "coef_a": coef_a[r],
                    "coef_b": coef_b[r],
                    "abs_coef_diff": abs(coef_a[r] - coef_b[r]),
                })
        top_df = pd.DataFrame(rows).sort_values("abs_coef_diff", ascending=False).reset_index(drop=True)

    return out, top_df


# =============================================================================
# Load matched RT file
# =============================================================================

print("[INFO] Loading matched RT file...")
matched = pd.read_csv(MATCHED_CSV)

if matched.empty:
    raise RuntimeError("matched_rt_eeg_index.csv is empty.")

matched["participant_id"] = matched["participant_id"].map(norm_pid)
matched["cell"] = matched["cell"].astype(str).str.upper().str.strip()
matched["rt"] = pd.to_numeric(matched["rt"], errors="coerce")
matched = matched.dropna(subset=["rt"]).reset_index(drop=True)
matched["rt_z"] = zscore_within_subject(matched, "rt", by="participant_id")

# =============================================================================
# Discover features
# =============================================================================

pkl_files = sorted(EEG_DIR.glob("*_parcellated_power.pkl"))
if not pkl_files:
    raise FileNotFoundError("No *_parcellated_power.pkl files found in EEG_DIR.")

band_labels = None
roi_labels = None

for p in pkl_files:
    with open(p, "rb") as f:
        trials = pickle.load(f)
    if isinstance(trials, list) and trials:
        bp = trials[0].get("band_power", None)
        if isinstance(bp, np.ndarray) and bp.ndim == 2:
            band_labels = list(trials[0]["band_labels"])
            roi_labels = list(trials[0]["roi_labels"])
            break

if band_labels is None or roi_labels is None:
    raise RuntimeError("Could not recover band/ROI labels.")

FEATURES = feature_names_band_major(band_labels, roi_labels)
N_FEATURES = len(FEATURES)

print(f"[INFO] Features: {N_FEATURES}")

# =============================================================================
# Build aligned trial matrix
# =============================================================================

Y_rows = []
meta_rows = []

for p in pkl_files:
    with open(p, "rb") as f:
        trials = pickle.load(f)

    if not isinstance(trials, list) or len(trials) == 0:
        continue

    pid_raw = trials[0].get(
        "participant_id_norm",
        trials[0].get("participant_id", p.name.replace("_parcellated_power.pkl", ""))
    )
    pid = norm_pid(pid_raw)

    rows = []
    has_trial_index = "trial_index" in trials[0]
    has_cell_field = "cell" in trials[0]

    for tr in trials:
        bp = tr.get("band_power", None)
        if not (isinstance(bp, np.ndarray) and bp.shape == (len(band_labels), len(roi_labels))):
            continue

        if has_cell_field:
            cell = str(tr["cell"]).upper().strip()
        else:
            role = norm_role(tr.get("role"))
            cond = norm_condition(tr.get("condition"))
            cell = cell_code(role, cond)

        trial_index = tr["trial_index"] if has_trial_index else None

        rows.append({
            "cell": cell,
            "bp": bp,
            "trial_index": trial_index,
        })

    if not rows:
        continue

    df_eeg = pd.DataFrame(rows)
    df_eeg = df_eeg[df_eeg["cell"].isin(CELL_ORDER)].copy()

    if df_eeg.empty:
        continue

    if df_eeg["trial_index"].isna().any():
        df_eeg["trial_index"] = df_eeg.groupby("cell").cumcount()

    m_pid = matched[matched["participant_id"] == pid]
    if m_pid.empty:
        continue

    for cell in CELL_ORDER:
        eeg_cell = df_eeg[df_eeg["cell"] == cell][["trial_index", "bp"]]
        if eeg_cell.empty:
            continue

        join = pd.merge(
            eeg_cell,
            m_pid[m_pid["cell"] == cell][["trial_index", "rt_z"]],
            on="trial_index",
            how="inner",
            validate="one_to_one"
        )

        if join.empty:
            continue

        rc, cc = effect_codes_from_cell(cell)

        for _, row in join.iterrows():
            Y_rows.append(flatten_band_power(row["bp"]))
            meta_rows.append({
                "participant_id": pid,
                "cell": cell,
                "trial_index": int(row["trial_index"]),
                "rt_z": float(row["rt_z"]),
                "role_code": rc,
                "cond_code": cc,
            })

if not Y_rows:
    raise RuntimeError("No aligned trials assembled for sensitivity analysis.")

Y = np.vstack(Y_rows)
meta = pd.DataFrame(meta_rows)

meta["subject"] = meta["participant_id"].astype(str)
meta["role_name"] = np.where(meta["role_code"] == 0.5, "Leader", "Follower")

print(f"[INFO] Assembled trial matrix: {Y.shape}")
print(f"[INFO] Unique subjects: {meta['subject'].nunique()}")

# =============================================================================
# Fit role-specific models
# =============================================================================

def fit_role_subset(role_name: str):
    idx = np.where(meta["role_name"].eq(role_name))[0]
    dsub = meta.iloc[idx].copy().reset_index(drop=True)
    ysub = Y[idx, :]

    rows = []
    for i, feature_name in enumerate(FEATURES):
        out = fit_role_specific_feature(dsub, ysub[:, i], feature_name, role_name)
        rows.append(out)

    res = pd.DataFrame(rows)
    res_fdr = add_fdr_columns(res)
    survivors = survivors_by_direction(res_fdr)
    return res_fdr, survivors

leader_res_fdr, leader_surv = fit_role_subset("Leader")
follower_res_fdr, follower_surv = fit_role_subset("Follower")

# =============================================================================
# Jaccard comparison
# =============================================================================

panels = {
    "Leader: Matching > Mirroring": leader_surv[leader_surv["cond_direction"] == "Matching > Mirroring"].copy(),
    "Leader: Mirroring > Matching": leader_surv[leader_surv["cond_direction"] == "Mirroring > Matching"].copy(),
    "Follower: Matching > Mirroring": follower_surv[follower_surv["cond_direction"] == "Matching > Mirroring"].copy(),
    "Follower: Mirroring > Matching": follower_surv[follower_surv["cond_direction"] == "Mirroring > Matching"].copy(),
}

sig = {}
for title, rows in panels.items():
    rois, roi_coef = panel_signature(rows)
    sig[title] = (rois, roi_coef)

pair_defs = [
    ("Leader: Matching > Mirroring", "Follower: Matching > Mirroring"),
    ("Leader: Mirroring > Matching", "Follower: Mirroring > Matching"),
]

jacc_rows = []
top_tables = {}

for a, b in pair_defs:
    summary_row, top_df = compare_panels(sig, a, b)
    jacc_rows.append(summary_row)

    clean_key = (
        a.replace("Leader: ", "")
         .replace("Follower: ", "")
         .replace(" ", "_")
         .replace(">", "vs")
    )
    top_tables[clean_key] = top_df

jacc_df = pd.DataFrame(jacc_rows)

# =============================================================================
# Save outputs
# =============================================================================

jacc_csv = OUT_DIR / "independent_role_jaccard_summary.csv"
jacc_df.to_csv(jacc_csv, index=False)
print(f"[SAVED] {jacc_csv}")

if SAVE_TOP_SHARED_TABLES:
    for key, top_df in top_tables.items():
        if top_df is not None and not top_df.empty:
            top_path = OUT_DIR / f"{key}_top_shared_roi_diffs.csv"
            top_df.head(TOP_N_SHARED).to_csv(top_path, index=False)
            print(f"[SAVED] {top_path}")

# =============================================================================
# Console summary
# =============================================================================

print("\n[SUMMARY] Independent role Jaccard sensitivity analysis")
print(jacc_df.to_string(index=False))

print("\n--- Jaccard sensitivity analysis complete. ---")
