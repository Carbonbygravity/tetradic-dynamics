"""
Script: 01_rt_qa_outlier_screening.py

Purpose
-------
Performs behavioural reaction-time (RT) quality assurance and outlier screening
on the trial-level dyadic RT dataset.

Pipeline role
-------------
This is the first behavioural preprocessing step. It normalises trial labels
into the four experimental cells, applies prespecified RT exclusion rules within
participant × cell, summarises exclusions and trial balance, and saves the
cleaned dataset used in downstream behavioural analyses.

Inputs
------
A CSV file containing at least the following columns:
    dyad, participant, participant_id, role, condition, trial, rt_ms

Outputs
-------
1. A cleaned behavioural dataset:
       reshaped_rt_data_dyads_cleaned.csv
2. An exclusion log:
       reshaped_rt_data_dyads_exclusions.csv

Exclusion rules
---------------
1. Hard lower RT bound:
       rt_ms < 150 ms
2. High-end distributional outlier within participant × cell:
       rt_ms > mean + 3 * SD
3. Optional robust cap within participant × cell:
       outside the [1st, 99th] percentile range

Notes
-----
- Experimental cells are normalised to:
      LMR = Leader Mirroring
      LMA = Leader Matching
      FMR = Follower Mirroring
      FMA = Follower Matching
- This script also prints descriptive summaries before and after cleaning.
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


# =============================================================================
# User settings
# =============================================================================

parser = argparse.ArgumentParser(description="Screen trial-level dyadic RT data.")
parser.add_argument("--input-csv", required=True, type=Path,
                    help="Trial-level RT input CSV")
parser.add_argument("--output-dir", required=True, type=Path,
                    help="New output directory; must not already exist")
args = parser.parse_args()

INPUT_PATH = args.input_csv
OUT_DIR = args.output_dir
if not INPUT_PATH.is_file():
    parser.error(f"Input CSV does not exist: {INPUT_PATH}")
if OUT_DIR.exists():
    parser.error(f"Output path already exists; choose a new directory: {OUT_DIR}")
OUT_DIR.mkdir(parents=True)

MIN_RT_MS = 150
Z_HI = 3.0
USE_ROBUST_CAPS = True
ROBUST_LOW_P = 1.0
ROBUST_HIGH_P = 99.0

CELLS = ["LMR", "LMA", "FMR", "FMA"]


# =============================================================================
# Helper functions
# =============================================================================

def normalize_condition_to_cell(role: str, condition: str) -> str:
    """
    Map each row to one of the four experimental cells:
    LMR, LMA, FMR, FMA.

    Parameters
    ----------
    role : str
        Participant role ('Leader' or 'Follower').
    condition : str
        Condition label, typically containing values such as LMR1, FMA2, etc.

    Returns
    -------
    str
        One of {'LMR', 'LMA', 'FMR', 'FMA'}.
    """
    role_norm = str(role).strip().title()
    cond_str = str(condition).strip().upper()
    base = re.sub(r"[^A-Z]", "", cond_str)
    base3 = base[:3] if len(base) >= 3 else base

    valid = {"LMR", "LMA", "FMR", "FMA"}
    if base3 in valid:
        return base3

    if "MIRROR" in base or "MR" in base:
        return "LMR" if role_norm == "Leader" else "FMR"
    if "MATCH" in base or "MA" in base:
        return "LMA" if role_norm == "Leader" else "FMA"

    if role_norm == "Leader":
        return "LMR" if "R" in base else "LMA"
    return "FMR" if "R" in base else "FMA"


def sem(values):
    """Compute standard error of the mean, ignoring NaNs."""
    values = np.asarray(values)
    valid_n = np.sum(~np.isnan(values))
    if valid_n <= 1:
        return np.nan
    return np.nanstd(values, ddof=1) / np.sqrt(valid_n)


def cohen_d_within(x, y):
    """
    Within-subject Cohen's d based on difference scores.
    """
    diff = x - y
    sd_diff = np.std(diff, ddof=1)
    if sd_diff == 0:
        return np.nan
    return np.mean(diff) / sd_diff


def flag_mean_plus_3sd(sub_df: pd.DataFrame) -> pd.Series:
    """
    Flag RTs greater than mean + Z_HI * SD within participant × cell.
    """
    mean_rt = sub_df["rt_ms"].mean()
    sd_rt = sub_df["rt_ms"].std(ddof=1)

    if not np.isfinite(sd_rt) or sd_rt == 0:
        return pd.Series(False, index=sub_df.index)

    return sub_df["rt_ms"] > (mean_rt + Z_HI * sd_rt)


def flag_robust_caps(sub_df: pd.DataFrame) -> pd.Series:
    """
    Flag RTs outside the percentile range [ROBUST_LOW_P, ROBUST_HIGH_P]
    within participant × cell.
    """
    low = np.percentile(sub_df["rt_ms"], ROBUST_LOW_P)
    high = np.percentile(sub_df["rt_ms"], ROBUST_HIGH_P)
    return (sub_df["rt_ms"] < low) | (sub_df["rt_ms"] > high)


def summarize_group_means(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Print descriptive RT summaries by role and broad condition.
    """
    cond_map = {
        "LMR": "Mirroring",
        "FMR": "Mirroring",
        "LMA": "Matching",
        "FMA": "Matching",
    }

    summary_df = df_in.assign(
        broad_condition=df_in["cell"].map(cond_map)
    )

    group_summary = (
        summary_df.groupby(["role", "broad_condition"])["rt_ms"]
        .agg(
            mean="mean",
            sd=lambda x: x.std(ddof=1),
            sem=sem,
            n="count"
        )
        .reset_index()
    )

    print(f"\n=== Group Means with SD & SEM ({label}) ===")
    print(group_summary.to_string(index=False))

    return group_summary


def paired_report(wide_df: pd.DataFrame, cell_a: str, cell_b: str, label_a: str, label_b: str):
    """
    Print paired t-test and within-subject Cohen's d for two cells.
    """
    x = wide_df[cell_a].values
    y = wide_df[cell_b].values

    t_stat, p_value = ttest_rel(x, y)
    d_value = cohen_d_within(x, y)
    mean_a = x.mean()
    mean_b = y.mean()
    mean_diff = (x - y).mean()

    print(f"\n[Paired] {label_a} vs {label_b} (n={len(x)}):")
    print(f"  Means: {label_a} = {mean_a:.2f} ms, {label_b} = {mean_b:.2f} ms, Δ = {mean_diff:.2f} ms")
    print(f"  t = {t_stat:.3f}, p = {p_value:.6f}, Cohen's d = {d_value:.3f}")


# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "dyad", "participant", "participant_id", "role", "condition", "trial", "rt_ms"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in {INPUT_PATH}: {missing_columns}"
    )

print(f"\n[INFO] Loaded {len(df):,} trials from: {INPUT_PATH}")


# =============================================================================
# Normalise labels
# =============================================================================

df["role"] = df["role"].astype(str).str.strip().str.title()
df["cell"] = [
    normalize_condition_to_cell(role, condition)
    for role, condition in zip(df["role"], df["condition"])
]

n_before_cell_filter = len(df)
df = df[df["cell"].isin(CELLS)].copy()

if len(df) < n_before_cell_filter:
    print(f"[WARN] Dropped {n_before_cell_filter - len(df)} rows with unmapped or invalid conditions.")


# =============================================================================
# Basic sanity checks
# =============================================================================

n_missing_rt = df["rt_ms"].isna().sum()
if n_missing_rt > 0:
    print(f"[WARN] Found {n_missing_rt} rows with missing rt_ms; these rows will be excluded.")

df = df.dropna(subset=["rt_ms"]).copy()

print(f"[INFO] Trials after dropping missing RTs: {len(df):,}")
print(f"[INFO] RT range: {df['rt_ms'].min():.1f} ms to {df['rt_ms'].max():.1f} ms")


# =============================================================================
# Trial-count balance before cleaning
# =============================================================================

counts_pre = df.groupby(["participant_id", "cell"]).size().unstack("cell", fill_value=0)

print("\n[Balance] Trials per participant × cell (pre-clean):")
print(counts_pre.describe().round(2))

imbalance_pre = (counts_pre.max(axis=1) - counts_pre.min(axis=1)).describe().round(2)
print("\n[Balance] Within-participant imbalance (max - min across the four cells, pre-clean):")
print(imbalance_pre)


# =============================================================================
# Outlier detection within participant × cell
# =============================================================================

df["outlier_reason"] = ""

# Rule 1: hard lower bound
mask_fast = df["rt_ms"] < MIN_RT_MS
df.loc[mask_fast, "outlier_reason"] += "|fast_lt_min"

# Rule 2: mean + 3 SD upper bound within participant × cell
upper_flags = df.groupby(["participant_id", "cell"], group_keys=False).apply(flag_mean_plus_3sd)
df.loc[upper_flags, "outlier_reason"] += "|gt_mean_plus_3sd"

# Rule 3: optional robust percentile caps within participant × cell
if USE_ROBUST_CAPS:
    robust_flags = df.groupby(["participant_id", "cell"], group_keys=False).apply(flag_robust_caps)
    df.loc[robust_flags, "outlier_reason"] += "|outside_p1_p99"

df["is_outlier"] = df["outlier_reason"].str.len() > 0


# =============================================================================
# Exclusion summaries
# =============================================================================

n_outliers = int(df["is_outlier"].sum())
print(f"\n[Exclusions] Total flagged trials: {n_outliers:,} ({100 * n_outliers / len(df):.2f}%)")

if n_outliers > 0:
    reason_counts = (
        df.loc[df["is_outlier"], "outlier_reason"]
        .str.strip("|")
        .str.split("|")
        .explode()
        .value_counts()
        .sort_index()
    )

    print("\n[Exclusions] Reasons (counts):")
    print(reason_counts.to_string())

    exclusions_by_participant_cell = (
        df[df["is_outlier"]]
        .groupby(["participant_id", "cell"])
        .size()
        .sort_values(ascending=False)
    )

    print("\n[Exclusions] By participant × cell (top 20 by count):")
    print(exclusions_by_participant_cell.head(20).to_string())


# =============================================================================
# Save exclusion log and cleaned dataset
# =============================================================================

output_dir = OUT_DIR
base_name = INPUT_PATH.stem

exclusions_path = output_dir / f"{base_name}_exclusions.csv"
cleaned_path = output_dir / f"{base_name}_cleaned.csv"

df[df["is_outlier"]].to_csv(exclusions_path, index=False)

clean = df[~df["is_outlier"]].copy()
clean.to_csv(cleaned_path, index=False)

print(f"\n[Saved] Exclusions log: {exclusions_path}")
print(f"[Saved] Cleaned dataset: {cleaned_path}")
print(f"[INFO] Trials retained: {len(clean):,} (removed {n_outliers:,})")


# =============================================================================
# Trial-count balance after cleaning
# =============================================================================

counts_post = clean.groupby(["participant_id", "cell"]).size().unstack("cell", fill_value=0)

print("\n[Balance] Trials per participant × cell (post-clean):")
print(counts_post.describe().round(2))

imbalance_post = (counts_post.max(axis=1) - counts_post.min(axis=1)).describe().round(2)
print("\n[Balance] Within-participant imbalance (max - min across the four cells, post-clean):")
print(imbalance_post)


# =============================================================================
# Descriptive summaries before and after cleaning
# =============================================================================

_ = summarize_group_means(df, "pre-clean")
_ = summarize_group_means(clean, "post-clean")


# =============================================================================
# Paired descriptive comparisons on participant means after cleaning
# =============================================================================

wide = (
    clean.pivot_table(
        index="participant_id",
        columns="cell",
        values="rt_ms",
        aggfunc="mean"
    )
    .reindex(columns=CELLS)
)

if wide.isna().any().any():
    missing_cells = wide.columns[wide.isna().any()].tolist()
    print(f"\n[WARN] Some participants lack data in cells: {missing_cells}. These participants will be dropped for paired comparisons.")

wide = wide.dropna(axis=0, how="any")
print(f"\n[Paired descriptive comparisons on participant means, post-clean] n = {wide.shape[0]}")

paired_report(wide, "LMA", "LMR", "Leader_MATCHING", "Leader_MIRRORING")
paired_report(wide, "FMA", "FMR", "Follower_MATCHING", "Follower_MIRRORING")
paired_report(wide, "LMA", "FMA", "Leader_MATCHING", "Follower_MATCHING")
paired_report(wide, "LMR", "FMR", "Leader_MIRRORING", "Follower_MIRRORING")

print("\n--- RT QA complete. Use the cleaned CSV for downstream behavioural and neural analyses. ---")
