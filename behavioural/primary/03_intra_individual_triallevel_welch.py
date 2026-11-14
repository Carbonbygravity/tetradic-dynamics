"""
Script: 03_intra_individual_triallevel_welch.py

Purpose
-------
Run within-individual trial-level Welch contrasts on cleaned RT
data for the locked participant subset used in this analysis.

Inputs
------
- individual_level_rt_descriptives.csv
  Used only to lock the participant subset.
- reshaped_rt_data_dyads_cleaned.csv

Outputs
-------
- intra_individual_triallevel_welch_NEW32.csv
- intra_individual_triallevel_welch_NEW32_summary.csv

Tests
-----
- Leader: LMR vs LMA
- Follower: FMR vs FMA

Notes
-----
- Participant inclusion is locked by the participant-level descriptives file.
- P-values are Bonferroni-adjusted within participant across available tests.
"""

import argparse
import re
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from pathlib import Path

parser = argparse.ArgumentParser(description="Run within-participant trial-level RT contrasts.")
parser.add_argument("--participant-lock-csv", type=Path, required=True)
parser.add_argument("--trials-csv", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
NEW_IDS_CSV = args.participant_lock_csv
TRIALS_CSV = args.trials_csv
OUT_DIR = args.output_dir
if OUT_DIR.exists():
    raise FileExistsError(f"Output directory already exists: {OUT_DIR}")
OUT_DIR.mkdir(parents=True)
OUT_ROWS = OUT_DIR / "intra_individual_triallevel_welch_NEW32.csv"
OUT_SUMMARY = OUT_DIR / "intra_individual_triallevel_welch_NEW32_summary.csv"
REQUIRED_ID_COLUMNS = {"participant_id"}
REQUIRED_TRIAL_COLUMNS = {"participant_id", "role", "condition", "rt_ms"}

# -------------------- Helpers --------------------
def normalize_cell(role: str, condition: str) -> str:
    """
    Map row to canonical cells {LMR, LMA, FMR, FMA} using role + raw condition label.
    Purpose: robustly unify labels even if condition strings differ across exports.
    """
    role_norm = str(role).strip().title()
    base = re.sub(r'[^A-Z]', '', str(condition).strip().upper())

    if len(base) >= 3 and base[:3] in {"LMR", "LMA", "FMR", "FMA"}:
        return base[:3]
    if "MIRROR" in base or "MR" in base:
        return "LMR" if role_norm == "Leader" else "FMR"
    if "MATCH" in base or "MA" in base:
        return "LMA" if role_norm == "Leader" else "FMA"

    # Fallback on presence of 'R'
    return ("LMR" if role_norm == "Leader" else "FMR") if "R" in base else (
        "LMA" if role_norm == "Leader" else "FMA"
    )

def hedges_g_unpaired(x, y):
    """
    Hedges' g for independent samples with small-sample correction.
    Purpose: unbiased standardized effect size for Welch comparisons.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp2 = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2) if (nx + ny - 2) > 0 else np.nan
    if not np.isfinite(sp2) or sp2 <= 0:
        return np.nan

    d = (np.mean(x) - np.mean(y)) / np.sqrt(sp2)
    J = 1 - 3 / (4 * (nx + ny) - 9) if (nx + ny) > 2 else 1.0
    return d * J

def effect_bin(d):
    """Purpose: simple verbal labels for standardized effects."""
    if not np.isfinite(d):
        return "negligible"
    a = abs(d)
    if a < 0.2:
        return "negligible"
    elif a < 0.5:
        return "small"
    elif a < 0.8:
        return "medium"
    return "large"

# -------------------- 1) Lock to NEW IDs --------------------
if not NEW_IDS_CSV.exists():
    raise FileNotFoundError(
        f"Participant lock file not found: {NEW_IDS_CSV}\n"
        "Edit NEW_IDS_CSV to point to individual_level_rt_descriptives.csv."
    )
if not TRIALS_CSV.exists():
    raise FileNotFoundError(
        f"Trial-level RT dataset not found: {TRIALS_CSV}\n"
        "Edit TRIALS_CSV to point to reshaped_rt_data_dyads_cleaned.csv."
    )

ids_df = pd.read_csv(NEW_IDS_CSV)
missing = REQUIRED_ID_COLUMNS.difference(ids_df.columns)
if missing:
    raise ValueError(f"NEW_IDS_CSV is missing required columns: {sorted(missing)}")

new_ids = (
    ids_df["participant_id"]
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)
new_ids_set = set(new_ids)
print(f"[INFO] NEW IDs loaded: {len(new_ids)} (expected 32)")

# -------------------- 2) Load CLEANED trials & subset --------------------
df = pd.read_csv(TRIALS_CSV)
missing = REQUIRED_TRIAL_COLUMNS.difference(df.columns)
if missing:
    raise ValueError(f"TRIALS_CSV is missing required columns: {sorted(missing)}")

df["participant_id"] = df["participant_id"].astype(str).str.strip()
df = df[df["participant_id"].isin(new_ids_set)].copy()

df["role"] = df["role"].astype(str).str.title().str.strip()
df["cell"] = [normalize_cell(r, c) for r, c in zip(df["role"], df["condition"])]

present = set(df["participant_id"].unique())
missing = sorted(list(new_ids_set - present))
if missing:
    print(f"[WARN] {len(missing)} NEW IDs not found in CLEANED trials: {missing}")
print(f"[INFO] Participants detected in trials after subsetting: {df['participant_id'].nunique()}")

# -------------------- 3) Planned Welch tests per participant --------------------
planned = [
    ("Follower_MIRRORING vs Follower_MATCHING", "FMR", "FMA"),
    ("Leader_MIRRORING vs Leader_MATCHING",     "LMR", "LMA"),
]

rows = []
for pid, sub in df.groupby("participant_id"):
    arr = {
        c: sub.loc[sub["cell"] == c, "rt_ms"].astype(float).to_numpy()
        for c in ["LMR", "LMA", "FMR", "FMA"]
    }

    for label, A, B in planned:
        x = arr.get(A, np.array([]))
        y = arr.get(B, np.array([]))

        if len(x) >= 5 and len(y) >= 5:
            t, p = ttest_ind(x, y, equal_var=False)  # Welch
            d_hedges = hedges_g_unpaired(x, y)

            # Cohen's d (pooled SD) for reference
            s2x, s2y = np.var(x, ddof=1), np.var(y, ddof=1)
            sp = (
                np.sqrt(((len(x) - 1) * s2x + (len(y) - 1) * s2y) / (len(x) + len(y) - 2))
                if (len(x) + len(y) - 2) > 0
                else np.nan
            )
            d_cohen = (np.mean(x) - np.mean(y)) / sp if np.isfinite(sp) and sp > 0 else np.nan

            # SE of difference in means (Welch)
            se_diff = np.sqrt(s2x / len(x) + s2y / len(y)) if (len(x) > 1 and len(y) > 1) else np.nan

            rows.append({
                "participant_id": pid,
                "Comparison": label,
                "cell_A": A,
                "cell_B": B,
                "n_A": int(len(x)),
                "n_B": int(len(y)),
                "Mean_A": float(np.mean(x)),
                "Mean_B": float(np.mean(y)),
                "Delta_ms": float(np.mean(x) - np.mean(y)),  # Mir − Match; negative => MIR faster
                "t_Welch": float(t),
                "p_unc": float(p),
                "Hedges_g": float(d_hedges),
                "Cohen_d": float(d_cohen),
                "SE_diff": float(se_diff),
                "Effect_Size_Interpretation": effect_bin(d_hedges),
                "Note": "Welch (independent trials within condition)",
            })
        else:
            rows.append({
                "participant_id": pid,
                "Comparison": label,
                "cell_A": A,
                "cell_B": B,
                "n_A": int(len(x)),
                "n_B": int(len(y)),
                "Mean_A": float(np.mean(x)) if len(x) else np.nan,
                "Mean_B": float(np.mean(y)) if len(y) else np.nan,
                "Delta_ms": np.nan,
                "t_Welch": np.nan,
                "p_unc": np.nan,
                "Hedges_g": np.nan,
                "Cohen_d": np.nan,
                "SE_diff": np.nan,
                "Effect_Size_Interpretation": "negligible",
                "Note": "insufficient trials",
            })

res = pd.DataFrame(rows)

# Bonferroni within participant across the tests they actually had (max 2)
def bonf(series):
    m = series.notna().sum()
    return np.minimum(series * m, 1.0) if m > 0 else series

res["p_bonf"] = res.groupby("participant_id")["p_unc"].transform(bonf)
res["significant_0.05"] = res["p_bonf"] < 0.05

# -------------------- 4) Save detailed rows --------------------
res.to_csv(OUT_ROWS, index=False)
print(f"✅ Saved per-participant trial-level contrasts → {OUT_ROWS}")

# -------------------- 5) Cohort summary table --------------------
# Compact view: N, significance count, direction (Mirroring faster), and effect-size bins
tmp = res.dropna(subset=["Delta_ms"]).copy()
tmp["MIR_faster"] = tmp["Delta_ms"] < 0  # Delta_ms = MIR − MAT

summ = (
    tmp.groupby("Comparison")
    .agg(
        N=("participant_id", "nunique"),
        Sig_N=("significant_0.05", "sum"),
        MIR_lt_MAT=("MIR_faster", "sum"),
    )
    .assign(MAT_lt_MIR=lambda d: d["N"] - d["MIR_lt_MAT"])
)

# Effect-size counts by Hedges' g bins
bins = (
    tmp.groupby(["Comparison", "Effect_Size_Interpretation"])
    .size()
    .unstack(fill_value=0)
)

for col in ["negligible", "small", "medium", "large"]:
    if col not in bins.columns:
        bins[col] = 0

bins = bins[["negligible", "small", "medium", "large"]]
summ = summ.join(bins)

# Percent columns
summ["% Significant"] = 100 * summ["Sig_N"] / summ["N"]
summ["% MIR < MAT"] = 100 * summ["MIR_lt_MAT"] / summ["N"]
summ["% MAT < MIR"] = 100 * summ["MAT_lt_MIR"] / summ["N"]
summ["Non-Neg"] = summ[["small", "medium", "large"]].sum(axis=1)
summ["% Non-Neg"] = 100 * summ["Non-Neg"] / summ["N"]

# Reorder columns for display
summ = summ[
    [
        "N",
        "Sig_N",
        "MIR_lt_MAT",
        "MAT_lt_MIR",
        "% Significant",
        "% MIR < MAT",
        "% MAT < MIR",
        "negligible",
        "small",
        "medium",
        "large",
        "Non-Neg",
        "% Non-Neg",
    ]
].sort_values("% Significant", ascending=False)

summ.to_csv(OUT_SUMMARY)
print(f"✅ Saved cohort summary → {OUT_SUMMARY}")

print("\n=== Summary of All Comparisons (NEW N=32) ===")
with pd.option_context(
    "display.max_columns", None,
    "display.width", 120,
    "display.float_format", "{:.3f}".format,
):
    print(summ.to_string())

