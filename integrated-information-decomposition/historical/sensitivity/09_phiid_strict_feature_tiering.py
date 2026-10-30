"""
Script: 09_phiid_strict_feature_tiering.py

Purpose
-------
Build a strict feature-level robustness summary for ΦID targets using previously
computed dyad-level surrogate results.

This script:
1. Loads:
   - full_targets_geometry.csv
   - phiid_p_trialshuffle_true_full_geometry_by_dyad.csv
   - phiid_p_timeshift_true_full_geometry_by_dyad.csv
2. Aggregates dyad-level robustness to the feature level
3. Assigns strict Tier A / Tier B / Tier C labels
4. Writes tier-specific CSV files and map-ready summaries
5. Prints a summary of counts and support levels

Pipeline role
-------------
Primary downstream robustness-classification script for ΦID features.

Strict tier definitions
-----------------------
Tier A:
    trial-shuffle support >= MIN_DYADS_TRIAL
    AND time-shift support >= MIN_DYADS_TS

Tier B:
    trial-shuffle support >= MIN_DYADS_TRIAL
    AND time-shift support < MIN_DYADS_TS
    OR no time-shift row exists

Tier C:
    original target present but not Tier A or Tier B

Notes
-----
- This script does not recompute ΦID.
- It preserves full feature coordinates:
    partition, contrast, atom, band, roi, win_start_ms, win_stop_ms
- Outputs can be used directly for tables, maps, and restricted follow-up analyses.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# =============================================================================
# User configuration
# =============================================================================

BASE_DIR = Path("PATH_TO_PHIID_FULL_GEOMETRIC_ROBUSTNESS")

TARGETS_CSV = BASE_DIR / "full_targets_geometry.csv"
P_TRIAL_DYAD_CSV = BASE_DIR / "phiid_p_trialshuffle_true_full_geometry_by_dyad.csv"
P_TS_DYAD_CSV = BASE_DIR / "phiid_p_timeshift_true_full_geometry_by_dyad.csv"

RUNSIG_JSON = BASE_DIR / "phiid_full_geometry_run_signature.json"
SUMMARY_JSON = BASE_DIR / "phiid_full_geometry_summary.json"

OUT_FEATURE_SUMMARY = BASE_DIR / "feature_level_robustness_summary_strict.csv"
OUT_TIER_A = BASE_DIR / "tier_A_features_strict.csv"
OUT_TIER_B = BASE_DIR / "tier_B_features_strict.csv"
OUT_TIER_C = BASE_DIR / "tier_C_features_strict.csv"

OUT_MAP_A = BASE_DIR / "map_ready_summary_tier_A_strict.csv"
OUT_MAP_B = BASE_DIR / "map_ready_summary_tier_B_strict.csv"
OUT_TIME_A = BASE_DIR / "time_window_summary_tier_A_strict.csv"
OUT_TIME_B = BASE_DIR / "time_window_summary_tier_B_strict.csv"
OUT_TIER_COUNTS = BASE_DIR / "tier_counts_summary_strict.csv"

ALPHA_TRIAL = 0.05
ALPHA_TS = 0.05

MIN_DYADS_TRIAL = 4
MIN_DYADS_TS = 3

TOP_N = 30
MAX_PRINT_ROWS = 200

# =============================================================================
# Helpers
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

def print_subheader(title: str):
    print("\n" + "-" * 120)
    print(title)
    print("-" * 120)

def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")

def read_csv_checked(path: Path, label: str) -> pd.DataFrame:
    assert_exists(path, label)
    df = pd.read_csv(path)
    print(f"[PASS] Loaded {label}: rows={len(df):,}, cols={len(df.columns)}")
    return df

def check_required_cols(df: pd.DataFrame, required: list[str], label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    print(f"[PASS] {label} contains required columns.")

def pretty_print_df(df: pd.DataFrame, max_rows: int = 20):
    if df.empty:
        print("(empty)")
    else:
        print(df.head(max_rows).to_string(index=False))

def vc_table(df: pd.DataFrame, cols: list[str], count_name="n"):
    if df.empty:
        return pd.DataFrame(columns=cols + [count_name])
    out = df.groupby(cols).size().reset_index(name=count_name)
    return out.sort_values(cols).reset_index(drop=True)

def feature_key_df(df: pd.DataFrame) -> pd.Series:
    return (
        df["partition"].astype(str) + "||" +
        df["contrast"].astype(str) + "||" +
        df["atom"].astype(str) + "||" +
        df["band"].astype(str) + "||" +
        df["roi"].astype(str) + "||" +
        df["win_start_ms"].astype(str) + "||" +
        df["win_stop_ms"].astype(str)
    )

# =============================================================================
# Load
# =============================================================================

print_header("LOADING FILES")

df_targets = read_csv_checked(TARGETS_CSV, "TARGETS")
df_trial = read_csv_checked(P_TRIAL_DYAD_CSV, "P-TRIAL BY DYAD")
df_ts = read_csv_checked(P_TS_DYAD_CSV, "P-TIME-SHIFT BY DYAD") if P_TS_DYAD_CSV.exists() else pd.DataFrame()

run_sig = None
if RUNSIG_JSON.exists():
    with open(RUNSIG_JSON, "r", encoding="utf-8") as f:
        run_sig_obj = json.load(f)
    run_sig = run_sig_obj.get("sig_hash", None)
    print(f"[PASS] Loaded RUN SIGNATURE JSON: {run_sig}")

summary_obj = None
if SUMMARY_JSON.exists():
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary_obj = json.load(f)
    print("[PASS] Loaded SUMMARY JSON.")

# =============================================================================
# Schema validation
# =============================================================================

print_header("SCHEMA VALIDATION")

targets_req = [
    "target_id", "partition", "contrast", "atom", "band", "roi",
    "win_start_ms", "win_stop_ms", "window_rule"
]
trial_req = [
    "dyad", "target_id", "partition", "contrast", "atom", "band", "roi",
    "win_start_ms", "win_stop_ms", "obs_value", "p_trialshuffle_dyad"
]
ts_req = [
    "dyad", "target_id", "partition", "contrast", "atom", "band", "roi",
    "win_start_ms", "win_stop_ms", "obs_value", "p_timeshift_dyad"
]

check_required_cols(df_targets, targets_req, "TARGETS")
check_required_cols(df_trial, trial_req, "P-TRIAL BY DYAD")
if not df_ts.empty:
    check_required_cols(df_ts, ts_req, "P-TIME-SHIFT BY DYAD")

# =============================================================================
# Run-signature filtering
# =============================================================================

print_header("RUN SIGNATURE FILTERING")

if run_sig is not None:
    if "run_sig" in df_trial.columns:
        df_trial = df_trial[df_trial["run_sig"] == run_sig].copy()
    if not df_ts.empty and "run_sig" in df_ts.columns:
        df_ts = df_ts[df_ts["run_sig"] == run_sig].copy()
    print(f"[PASS] Filtered dyad-level files to run signature: {run_sig}")
else:
    print("[INFO] No run signature found. Proceeding without filtering.")

# =============================================================================
# Feature-level aggregation
# =============================================================================

print_header("BUILDING STRICT FEATURE-LEVEL SUMMARY")

feature_cols = [
    "target_id", "partition", "contrast", "atom", "band", "roi",
    "win_start_ms", "win_stop_ms", "window_rule"
]

trial_feature = (
    df_trial
    .assign(trial_pass=df_trial["p_trialshuffle_dyad"] <= ALPHA_TRIAL)
    .groupby(feature_cols, as_index=False)
    .agg(
        n_dyads_trial=("dyad", "nunique"),
        n_pass_trial=("trial_pass", "sum"),
        min_p_trial=("p_trialshuffle_dyad", "min"),
        mean_obs_trial=("obs_value", "mean"),
        mean_abs_obs_trial=("obs_value", lambda x: float(np.nanmean(np.abs(x))))
    )
)

if not df_ts.empty:
    ts_feature = (
        df_ts
        .assign(ts_pass=df_ts["p_timeshift_dyad"] <= ALPHA_TS)
        .groupby(feature_cols, as_index=False)
        .agg(
            n_dyads_ts=("dyad", "nunique"),
            n_pass_ts=("ts_pass", "sum"),
            min_p_ts=("p_timeshift_dyad", "min"),
            mean_obs_ts=("obs_value", "mean"),
            mean_abs_obs_ts=("obs_value", lambda x: float(np.nanmean(np.abs(x))))
        )
    )
else:
    ts_feature = pd.DataFrame(columns=feature_cols + [
        "n_dyads_ts", "n_pass_ts", "min_p_ts", "mean_obs_ts", "mean_abs_obs_ts"
    ])

df_feature = (
    df_targets[feature_cols]
    .drop_duplicates()
    .merge(trial_feature, on=feature_cols, how="left")
    .merge(ts_feature, on=feature_cols, how="left")
)

for col in ["n_dyads_trial", "n_pass_trial", "n_dyads_ts", "n_pass_ts"]:
    if col in df_feature.columns:
        df_feature[col] = df_feature[col].fillna(0).astype(int)

df_feature["trial_supported"] = df_feature["n_pass_trial"] >= MIN_DYADS_TRIAL
df_feature["ts_supported"] = df_feature["n_pass_ts"] >= MIN_DYADS_TS
df_feature["has_ts_rows"] = df_feature["n_dyads_ts"] > 0

df_feature["tier"] = "Tier_C"
df_feature.loc[df_feature["trial_supported"] & (~df_feature["ts_supported"]), "tier"] = "Tier_B"
df_feature.loc[df_feature["trial_supported"] & df_feature["ts_supported"], "tier"] = "Tier_A"
df_feature.loc[df_feature["trial_supported"] & (~df_feature["has_ts_rows"]), "tier"] = "Tier_B"

df_feature["feature_key"] = feature_key_df(df_feature)

df_feature = df_feature.sort_values(
    ["tier", "partition", "contrast", "atom", "band", "roi", "win_start_ms", "win_stop_ms"]
).reset_index(drop=True)

df_feature.to_csv(OUT_FEATURE_SUMMARY, index=False)

df_tier_A = df_feature[df_feature["tier"] == "Tier_A"].copy()
df_tier_B = df_feature[df_feature["tier"] == "Tier_B"].copy()
df_tier_C = df_feature[df_feature["tier"] == "Tier_C"].copy()

df_tier_A.to_csv(OUT_TIER_A, index=False)
df_tier_B.to_csv(OUT_TIER_B, index=False)
df_tier_C.to_csv(OUT_TIER_C, index=False)

print(f"[PASS] Wrote full strict feature summary:\n{OUT_FEATURE_SUMMARY}")
print(f"[PASS] Wrote strict Tier A features:\n{OUT_TIER_A}")
print(f"[PASS] Wrote strict Tier B features:\n{OUT_TIER_B}")
print(f"[PASS] Wrote strict Tier C features:\n{OUT_TIER_C}")

# =============================================================================
# Console summary
# =============================================================================

print_header("STRICT FEATURE-LEVEL ROBUSTNESS SUMMARY")

print(f"[INFO] Alpha trial-shuffle          : {ALPHA_TRIAL}")
print(f"[INFO] Alpha time-shift            : {ALPHA_TS}")
print(f"[INFO] Min dyads for trial support : {MIN_DYADS_TRIAL}")
print(f"[INFO] Min dyads for time support  : {MIN_DYADS_TS}")

print_subheader("Overall dimensions")
print(f"Original targets                  : {len(df_targets):,}")
print(f"Unique feature targets            : {len(df_feature):,}")
print(f"Strict Tier A features            : {len(df_tier_A):,}")
print(f"Strict Tier B features            : {len(df_tier_B):,}")
print(f"Strict Tier C features            : {len(df_tier_C):,}")

if summary_obj is not None:
    print_subheader("Cross-check with summary JSON")
    print(f"n_targets_total (json)            : {summary_obj.get('n_targets_total')}")
    print(f"n_surrogates (json)               : {summary_obj.get('n_surrogates')}")
    print(f"proceed_timeshift (json)          : {summary_obj.get('proceed_timeshift')}")
    print(f"alpha_timeshift_gate (json)       : {summary_obj.get('alpha_timeshift_gate')}")

tier_counts_partition = vc_table(df_feature, ["tier", "partition"], "n_features")
tier_counts_atom = vc_table(df_feature, ["tier", "atom"], "n_features")
tier_counts_contrast = vc_table(df_feature, ["tier", "contrast"], "n_features")
tier_counts_band = vc_table(df_feature, ["tier", "band"], "n_features")
tier_counts_partition_atom = vc_table(df_feature, ["tier", "partition", "atom"], "n_features")
tier_counts_partition_contrast_atom = vc_table(df_feature, ["tier", "partition", "contrast", "atom"], "n_features")

print_subheader("Tier counts by partition")
pretty_print_df(tier_counts_partition, max_rows=100)

print_subheader("Tier counts by atom")
pretty_print_df(tier_counts_atom, max_rows=100)

print_subheader("Tier counts by contrast")
pretty_print_df(tier_counts_contrast, max_rows=200)

print_subheader("Tier counts by band")
pretty_print_df(tier_counts_band, max_rows=100)

print_subheader("Tier counts by partition × atom")
pretty_print_df(tier_counts_partition_atom, max_rows=200)

print_subheader("Tier counts by partition × contrast × atom")
pretty_print_df(tier_counts_partition_contrast_atom, max_rows=MAX_PRINT_ROWS)

tier_counts_summary = pd.concat([
    tier_counts_partition.assign(summary_type="tier_by_partition"),
    tier_counts_atom.assign(summary_type="tier_by_atom"),
    tier_counts_contrast.assign(summary_type="tier_by_contrast"),
    tier_counts_band.assign(summary_type="tier_by_band"),
], ignore_index=True, sort=False)
tier_counts_summary.to_csv(OUT_TIER_COUNTS, index=False)

print_subheader("Support summary by tier")
support_tab = (
    df_feature.groupby("tier", as_index=False)
    .agg(
        n_features=("feature_key", "nunique"),
        mean_trial_pass=("n_pass_trial", "mean"),
        max_trial_pass=("n_pass_trial", "max"),
        mean_ts_pass=("n_pass_ts", "mean"),
        max_ts_pass=("n_pass_ts", "max"),
        mean_abs_obs_trial=("mean_abs_obs_trial", "mean"),
    )
)
pretty_print_df(support_tab, max_rows=20)

print_subheader("Top strict Tier A features by time-shift dyad support")
top_A = (
    df_tier_A.sort_values(
        ["n_pass_ts", "n_pass_trial", "mean_abs_obs_trial"],
        ascending=[False, False, False]
    )[[
        "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms",
        "n_pass_trial", "n_pass_ts",
        "min_p_trial", "min_p_ts",
        "mean_obs_trial", "mean_abs_obs_trial"
    ]]
)
pretty_print_df(top_A, max_rows=TOP_N)

print_subheader("Top strict Tier B features by trial-shuffle dyad support")
top_B = (
    df_tier_B.sort_values(
        ["n_pass_trial", "mean_abs_obs_trial"],
        ascending=[False, False]
    )[[
        "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms",
        "n_pass_trial", "n_pass_ts",
        "min_p_trial", "min_p_ts",
        "mean_obs_trial", "mean_abs_obs_trial", "has_ts_rows"
    ]]
)
pretty_print_df(top_B, max_rows=TOP_N)

print_subheader("Representative strict Tier C features")
top_C = (
    df_tier_C.sort_values(
        ["partition", "contrast", "atom", "band", "roi", "win_start_ms"]
    )[[
        "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms",
        "n_pass_trial", "n_pass_ts",
        "min_p_trial", "min_p_ts",
        "mean_obs_trial", "mean_abs_obs_trial"
    ]]
)
pretty_print_df(top_C, max_rows=TOP_N)

for tier_name, df_tier in [("Tier_A", df_tier_A), ("Tier_B", df_tier_B), ("Tier_C", df_tier_C)]:
    print_subheader(f"{tier_name}: top ROIs")
    pretty_print_df(
        vc_table(df_tier, ["roi"], "n_features").sort_values("n_features", ascending=False),
        max_rows=TOP_N
    )

    print_subheader(f"{tier_name}: top bands")
    pretty_print_df(
        vc_table(df_tier, ["band"], "n_features").sort_values("n_features", ascending=False),
        max_rows=TOP_N
    )

    print_subheader(f"{tier_name}: top atoms")
    pretty_print_df(
        vc_table(df_tier, ["atom"], "n_features").sort_values("n_features", ascending=False),
        max_rows=TOP_N
    )

print_subheader("Map-ready summary: strict Tier A by ROI × band × atom")
map_A = (
    df_tier_A.groupby(["roi", "band", "atom"], as_index=False)
    .agg(
        n_features=("feature_key", "nunique"),
        mean_trial_support=("n_pass_trial", "mean"),
        mean_ts_support=("n_pass_ts", "mean"),
        min_time_ms=("win_start_ms", "min"),
        max_time_ms=("win_stop_ms", "max")
    )
    .sort_values(["n_features", "mean_ts_support"], ascending=[False, False])
)
pretty_print_df(map_A, max_rows=MAX_PRINT_ROWS)
map_A.to_csv(OUT_MAP_A, index=False)

print_subheader("Map-ready summary: strict Tier B by ROI × band × atom")
map_B = (
    df_tier_B.groupby(["roi", "band", "atom"], as_index=False)
    .agg(
        n_features=("feature_key", "nunique"),
        mean_trial_support=("n_pass_trial", "mean"),
        min_time_ms=("win_start_ms", "min"),
        max_time_ms=("win_stop_ms", "max")
    )
    .sort_values(["n_features", "mean_trial_support"], ascending=[False, False])
)
pretty_print_df(map_B, max_rows=MAX_PRINT_ROWS)
map_B.to_csv(OUT_MAP_B, index=False)

for tier_name, df_tier, out_path in [
    ("Tier_A", df_tier_A, OUT_TIME_A),
    ("Tier_B", df_tier_B, OUT_TIME_B),
]:
    print_subheader(f"{tier_name}: time-window span summary")
    if df_tier.empty:
        print("(empty)")
    else:
        tw = (
            df_tier.groupby(["partition", "contrast", "atom", "band"], as_index=False)
            .agg(
                n_features=("feature_key", "nunique"),
                min_time_ms=("win_start_ms", "min"),
                max_time_ms=("win_stop_ms", "max"),
                mean_window_start=("win_start_ms", "mean"),
                mean_window_stop=("win_stop_ms", "mean"),
            )
            .sort_values(["n_features", "partition"], ascending=[False, True])
        )
        pretty_print_df(tw, max_rows=MAX_PRINT_ROWS)
        tw.to_csv(out_path, index=False)

print_header("INTERPRETIVE NOTES")
print(
    "Strict Tier A = feature-level support under both trial-shuffle and time-shift criteria,\n"
    "with higher recurrence requirements across dyads.\n"
    "Strict Tier B = feature-level support under trial-shuffle only,\n"
    "with higher recurrence requirements across dyads.\n"
    "Strict Tier C = original targets not meeting Strict Tier A or Strict Tier B.\n"
)

print(
    "These strict tiers preserve ROI, band, atom, contrast, partition, and time window,\n"
    "so they can be used directly for:\n"
    "  - stricter brain maps\n"
        "  - publication-ready ROI frequency tables\n"
    "  - atom-specific summaries under tighter recurrence criteria\n"
    "  - time-window restricted follow-up analyses\n"
)

print_header("FILES WRITTEN")
print(f"[WRITE] {OUT_FEATURE_SUMMARY}")
print(f"[WRITE] {OUT_TIER_A}")
print(f"[WRITE] {OUT_TIER_B}")
print(f"[WRITE] {OUT_TIER_C}")
print(f"[WRITE] {OUT_MAP_A}")
print(f"[WRITE] {OUT_MAP_B}")
print(f"[WRITE] {OUT_TIME_A}")
print(f"[WRITE] {OUT_TIME_B}")
print(f"[WRITE] {OUT_TIER_COUNTS}")

print_header("DONE")
print("[INFO] Strict feature-level tiering complete.")
