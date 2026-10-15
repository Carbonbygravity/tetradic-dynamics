"""
Script: 10_phiid_dyad_recurrence_summary.py

Purpose
-------
Build the dyad-consistency recurrence layer for ΦID targets using previously
computed feature-level and dyad-level robustness outputs.

This script:
1. Loads:
   - full_targets_geometry.csv
   - phiid_obs_full_geometry.csv
   - phiid_p_trialshuffle_true_full_geometry_by_dyad.csv
   - phiid_p_timeshift_true_full_geometry_by_dyad.csv
2. Computes recurrence across dyads for each target
3. Calculates exact one-sided binomial recurrence p-values
4. Applies BH-FDR correction
5. Quantifies sign consistency across dyads
6. Assigns recurrence classes
7. Writes summary tables and JSON outputs

Pipeline role
-------------
Primary ΦID recurrence-inference script for across-dyad recurrence analysis.

Recurrence classes
------------------
Recurrence_Class_1 = dual-null recurrent
Recurrence_Class_2 = trial-recurrent
Recurrence_Class_3 = partial or heterogeneous recurrence
Recurrence_Class_4 = weak or idiosyncratic

Notes
-----
- This script does not recompute ΦID.
- It evaluates whether features recur across dyads more than expected by chance.
- Trial-shuffle and time-shift recurrence are corrected separately by BH-FDR.
"""

from pathlib import Path
import json
from math import comb
import numpy as np
import pandas as pd

# =============================================================================
# User configuration
# =============================================================================

BASE_DIR = Path("PATH_TO_PHIID_FULL_GEOMETRIC_ROBUSTNESS")

TARGETS_CSV = BASE_DIR / "full_targets_geometry.csv"
OBS_CSV = BASE_DIR / "phiid_obs_full_geometry.csv"
P_TRIAL_DYAD_CSV = BASE_DIR / "phiid_p_trialshuffle_true_full_geometry_by_dyad.csv"
P_TS_DYAD_CSV = BASE_DIR / "phiid_p_timeshift_true_full_geometry_by_dyad.csv"
SUMMARY_JSON = BASE_DIR / "phiid_full_geometry_summary.json"
RUNSIG_JSON = BASE_DIR / "phiid_full_geometry_run_signature.json"

OUT_MAIN = BASE_DIR / "phiid_target_recurrence_summary.csv"
OUT_SIG = BASE_DIR / "phiid_recurrent_features_significant.csv"
OUT_PART = BASE_DIR / "phiid_recurrent_features_counts_by_partition_atom.csv"
OUT_CONTRAST = BASE_DIR / "phiid_recurrent_features_counts_by_contrast_atom.csv"
OUT_JSON = BASE_DIR / "phiid_target_recurrence_summary.json"

ALPHAS = [0.05, 0.10]

P0_TRIAL = 0.05
P0_TS = 0.05

MIN_SIGN_CONSISTENCY = 0.70
MIN_DYADS_STRONG = 4
MIN_DYADS_MODERATE = 3

# =============================================================================
# Helpers
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)

def print_sub(title: str):
    print("\n" + "-" * 108)
    print(title)
    print("-" * 108)

def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

def load_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[PASS] Loaded {label}: rows={len(df):,}, cols={len(df.columns)}")
    return df

def check_required(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    print(f"[PASS] {label} contains required columns.")

def exact_binom_sf(k: int, n: int, p: float) -> float:
    """
    Exact one-sided binomial survival:
        P(K >= k | n, p)
    """
    if n <= 0:
        return np.nan
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    s = 0.0
    for i in range(k, n + 1):
        s += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return float(min(max(s, 0.0), 1.0))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    NaNs are preserved.
    """
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out

    p = pvals[ok]
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        q[i] = prev

    q_back = np.empty(m, dtype=float)
    q_back[order] = np.clip(q, 0.0, 1.0)
    out[ok] = q_back
    return out

def majority_sign(values: np.ndarray):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())
    if n_pos > n_neg:
        return 1
    if n_neg > n_pos:
        return -1
    return 0

def sign_consistency(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())
    same = max(n_pos, n_neg)
    return float(same / vals.size)

def recurrence_class(row):
    trial_q = row["q_recur_trial_005"]
    ts_q = row["q_recur_ts_005"]
    n_trial = row["n_pass_trial_005"]
    n_ts = row["n_pass_ts_005"]
    signc = row["sign_consistency_all_dyads"]
    ts_available = row["n_dyads_ts_available"] > 0

    trial_sig = pd.notna(trial_q) and (trial_q < 0.05)
    ts_sig = pd.notna(ts_q) and (ts_q < 0.05)

    if (
        trial_sig
        and ts_available
        and ts_sig
        and (n_trial >= MIN_DYADS_STRONG)
        and (n_ts >= 2)
        and pd.notna(signc)
        and (signc >= MIN_SIGN_CONSISTENCY)
    ):
        return "Recurrence_Class_1_dual_null_recurrent"

    if (
        trial_sig
        and (n_trial >= MIN_DYADS_MODERATE)
        and pd.notna(signc)
        and (signc >= MIN_SIGN_CONSISTENCY)
    ):
        return "Recurrence_Class_2_trial_recurrent"

    if (
        (n_trial >= 2)
        and pd.notna(signc)
        and (signc >= 0.60)
    ):
        return "Recurrence_Class_3_partial_or_heterogeneous"

    return "Recurrence_Class_4_weak_or_idiosyncratic"

# =============================================================================
# Load
# =============================================================================

def load_all():
    print_header("LOADING FILES")

    assert_exists(TARGETS_CSV, "TARGETS_CSV")
    assert_exists(OBS_CSV, "OBS_CSV")
    assert_exists(P_TRIAL_DYAD_CSV, "P_TRIAL_DYAD_CSV")
    assert_exists(P_TS_DYAD_CSV, "P_TS_DYAD_CSV")
    assert_exists(SUMMARY_JSON, "SUMMARY_JSON")
    assert_exists(RUNSIG_JSON, "RUNSIG_JSON")

    df_targets = load_csv(TARGETS_CSV, "TARGETS")
    df_obs = load_csv(OBS_CSV, "OBSERVED")
    df_trial = load_csv(P_TRIAL_DYAD_CSV, "TRIAL P BY DYAD")
    df_ts = load_csv(P_TS_DYAD_CSV, "TIME-SHIFT P BY DYAD")

    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)
    print("[PASS] Loaded SUMMARY JSON.")

    with open(RUNSIG_JSON, "r", encoding="utf-8") as f:
        runsig = json.load(f)
    print(f"[PASS] Loaded RUN SIGNATURE JSON: {runsig.get('sig_hash', 'NA')}")

    return df_targets, df_obs, df_trial, df_ts, summary, runsig

# =============================================================================
# Validate
# =============================================================================

def validate(df_targets, df_obs, df_trial, df_ts):
    print_header("SCHEMA VALIDATION")

    check_required(
        df_targets,
        ["target_id", "partition", "contrast", "atom", "band", "roi", "win_start_ms", "win_stop_ms"],
        "TARGETS"
    )
    check_required(
        df_obs,
        ["dyad", "target_id", "partition", "contrast", "atom", "band", "roi",
         "win_start_ms", "win_stop_ms", "obs_value", "run_sig"],
        "OBSERVED"
    )
    check_required(
        df_trial,
        ["dyad", "target_id", "partition", "contrast", "atom", "band", "roi",
         "win_start_ms", "win_stop_ms", "obs_value", "p_trialshuffle_dyad", "run_sig"],
        "TRIAL P BY DYAD"
    )
    check_required(
        df_ts,
        ["dyad", "target_id", "partition", "contrast", "atom", "band", "roi",
         "win_start_ms", "win_stop_ms", "obs_value", "p_timeshift_dyad", "run_sig"],
        "TIME-SHIFT P BY DYAD"
    )

# =============================================================================
# Filter to run signature
# =============================================================================

def filter_run_sig(df_obs, df_trial, df_ts, runsig):
    print_header("RUN SIGNATURE FILTERING")
    sig = runsig["sig_hash"]

    df_obs = df_obs[df_obs["run_sig"] == sig].copy()
    df_trial = df_trial[df_trial["run_sig"] == sig].copy()
    df_ts = df_ts[df_ts["run_sig"] == sig].copy()

    print(f"[PASS] Filtered OBSERVED/TRIAL/TS to run signature: {sig}")
    return df_obs, df_trial, df_ts, sig

# =============================================================================
# Build target-level recurrence table
# =============================================================================

def build_target_recurrence(df_targets, df_obs, df_trial, df_ts):
    print_header("BUILDING TARGET-LEVEL DYAD RECURRENCE SUMMARY")

    dyads_all = sorted(df_obs["dyad"].dropna().astype(int).unique().tolist())
    print(f"[INFO] Total dyads in observed layer: {len(dyads_all)}")
    print(f"[INFO] Dyad list: {dyads_all}")

    obs_agg = (
        df_obs.groupby("target_id")
        .agg(
            n_dyads_obs=("dyad", "nunique"),
            mean_obs=("obs_value", "mean"),
            median_obs=("obs_value", "median"),
            sd_obs=("obs_value", "std"),
            min_obs=("obs_value", "min"),
            max_obs=("obs_value", "max"),
        )
        .reset_index()
    )

    sign_rows = []
    for tid, sub in df_obs.groupby("target_id"):
        vals = sub["obs_value"].to_numpy(dtype=float)
        sign_rows.append({
            "target_id": tid,
            "n_pos_dyads": int((vals > 0).sum()),
            "n_neg_dyads": int((vals < 0).sum()),
            "n_zero_dyads": int((vals == 0).sum()),
            "majority_sign": majority_sign(vals),
            "sign_consistency_all_dyads": sign_consistency(vals),
        })
    df_sign = pd.DataFrame(sign_rows)

    trial_rows = []
    for tid, sub in df_trial.groupby("target_id"):
        p = sub["p_trialshuffle_dyad"].to_numpy(dtype=float)
        trial_rows.append({
            "target_id": tid,
            "n_dyads_trial_available": int(sub["dyad"].nunique()),
            "n_pass_trial_005": int(np.sum(np.isfinite(p) & (p <= 0.05))),
            "n_pass_trial_010": int(np.sum(np.isfinite(p) & (p <= 0.10))),
        })
    df_trial_counts = pd.DataFrame(trial_rows)

    ts_rows = []
    for tid, sub in df_ts.groupby("target_id"):
        p = sub["p_timeshift_dyad"].to_numpy(dtype=float)
        ts_rows.append({
            "target_id": tid,
            "n_dyads_ts_available": int(sub["dyad"].nunique()),
            "n_pass_ts_005": int(np.sum(np.isfinite(p) & (p <= 0.05))),
            "n_pass_ts_010": int(np.sum(np.isfinite(p) & (p <= 0.10))),
        })
    df_ts_counts = pd.DataFrame(ts_rows)

    out = df_targets.copy()
    out = out.merge(obs_agg, on="target_id", how="left")
    out = out.merge(df_sign, on="target_id", how="left")
    out = out.merge(df_trial_counts, on="target_id", how="left")
    out = out.merge(df_ts_counts, on="target_id", how="left")

    for c in ["n_dyads_ts_available", "n_pass_ts_005", "n_pass_ts_010"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    for c in ["n_dyads_trial_available", "n_pass_trial_005", "n_pass_trial_010"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    out["p_recur_trial_005"] = [
        exact_binom_sf(int(r["n_pass_trial_005"]), int(r["n_dyads_trial_available"]), P0_TRIAL)
        if int(r["n_dyads_trial_available"]) > 0 else np.nan
        for _, r in out.iterrows()
    ]
    out["p_recur_trial_010"] = [
        exact_binom_sf(int(r["n_pass_trial_010"]), int(r["n_dyads_trial_available"]), 0.10)
        if int(r["n_dyads_trial_available"]) > 0 else np.nan
        for _, r in out.iterrows()
    ]
    out["p_recur_ts_005"] = [
        exact_binom_sf(int(r["n_pass_ts_005"]), int(r["n_dyads_ts_available"]), P0_TS)
        if int(r["n_dyads_ts_available"]) > 0 else np.nan
        for _, r in out.iterrows()
    ]
    out["p_recur_ts_010"] = [
        exact_binom_sf(int(r["n_pass_ts_010"]), int(r["n_dyads_ts_available"]), 0.10)
        if int(r["n_dyads_ts_available"]) > 0 else np.nan
        for _, r in out.iterrows()
    ]

    out["q_recur_trial_005"] = bh_fdr(out["p_recur_trial_005"].to_numpy(float))
    out["q_recur_trial_010"] = bh_fdr(out["p_recur_trial_010"].to_numpy(float))
    out["q_recur_ts_005"] = bh_fdr(out["p_recur_ts_005"].to_numpy(float))
    out["q_recur_ts_010"] = bh_fdr(out["p_recur_ts_010"].to_numpy(float))

    out["recurrence_class"] = out.apply(recurrence_class, axis=1)
    out["flag_trial_recurrent_fdr005"] = out["q_recur_trial_005"] < 0.05
    out["flag_ts_recurrent_fdr005"] = out["q_recur_ts_005"] < 0.05

    out = out.sort_values(
        ["partition", "contrast", "atom", "band", "roi", "win_start_ms", "win_stop_ms"]
    ).reset_index(drop=True)

    print(f"[PASS] Built target-level recurrence table: rows={len(out):,}")
    return out

# =============================================================================
# Console summary
# =============================================================================

def print_console_summary(df_inf, summary):
    print_header("COMPREHENSIVE TARGET-LEVEL DYAD RECURRENCE SUMMARY")

    print_sub("File dimensions")
    print(f"Targets total                     : {len(df_inf):,}")
    print(f"n_surrogates (json)              : {summary.get('n_surrogates', 'NA')}")
    print(f"proceed_timeshift (json)         : {summary.get('proceed_timeshift', 'NA')}")
    print(f"alpha_timeshift_gate (json)      : {summary.get('alpha_timeshift_gate', 'NA')}")

    print_sub("Target counts by partition × atom")
    print(
        df_inf.groupby(["partition", "atom"])
        .size()
        .reset_index(name="n_targets")
        .sort_values(["partition", "atom"])
        .to_string(index=False)
    )

    print_sub("Target counts by contrast × atom")
    print(
        df_inf.groupby(["contrast", "atom"])
        .size()
        .reset_index(name="n_targets")
        .sort_values(["contrast", "atom"])
        .to_string(index=False)
    )

    print_sub("Recurrence-significant targets by partition × atom")
    print(
        df_inf.assign(
            trial_sig=df_inf["q_recur_trial_005"] < 0.05,
            ts_sig=df_inf["q_recur_ts_005"] < 0.05
        )
        .groupby(["partition", "atom"])
        .agg(
            n_targets=("target_id", "size"),
            n_trial_recurrent=("trial_sig", "sum"),
            n_ts_recurrent=("ts_sig", "sum"),
        )
        .reset_index()
        .sort_values(["partition", "atom"])
        .to_string(index=False)
    )

    print_sub("Recurrence-significant targets by contrast × atom")
    print(
        df_inf.assign(
            trial_sig=df_inf["q_recur_trial_005"] < 0.05,
            ts_sig=df_inf["q_recur_ts_005"] < 0.05
        )
        .groupby(["contrast", "atom"])
        .agg(
            n_targets=("target_id", "size"),
            n_trial_recurrent=("trial_sig", "sum"),
            n_ts_recurrent=("ts_sig", "sum"),
        )
        .reset_index()
        .sort_values(["contrast", "atom"])
        .to_string(index=False)
    )

    print_sub("Recurrence class counts")
    print(
        df_inf["recurrence_class"]
        .value_counts(dropna=False)
        .rename_axis("recurrence_class")
        .reset_index(name="n_targets")
        .to_string(index=False)
    )

    print_sub("Top trial-recurrent targets (smallest q_recur_trial_005)")
    cols = [
        "target_id", "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms",
        "n_pass_trial_005", "n_dyads_trial_available",
        "p_recur_trial_005", "q_recur_trial_005",
        "mean_obs", "median_obs", "sign_consistency_all_dyads", "recurrence_class"
    ]
    print(
        df_inf.sort_values(
            ["q_recur_trial_005", "p_recur_trial_005", "n_pass_trial_005"],
            ascending=[True, True, False]
        ).head(30)[cols].to_string(index=False)
    )

    print_sub("Top time-shift recurrent targets (smallest q_recur_ts_005)")
    cols = [
        "target_id", "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms",
        "n_pass_ts_005", "n_dyads_ts_available",
        "p_recur_ts_005", "q_recur_ts_005",
        "mean_obs", "median_obs", "sign_consistency_all_dyads", "recurrence_class"
    ]
    print(
        df_inf[df_inf["n_dyads_ts_available"] > 0]
        .sort_values(
            ["q_recur_ts_005", "p_recur_ts_005", "n_pass_ts_005"],
            ascending=[True, True, False]
        ).head(30)[cols].to_string(index=False)
    )

    print_sub("Targets with strongest dyad consistency of sign")
    cols = [
        "target_id", "partition", "contrast", "atom", "band", "roi",
        "n_pos_dyads", "n_neg_dyads", "sign_consistency_all_dyads",
        "mean_obs", "median_obs", "recurrence_class"
    ]
    print(
        df_inf.sort_values(
            ["sign_consistency_all_dyads", "n_pass_trial_005", "q_recur_trial_005"],
            ascending=[False, False, True]
        ).head(30)[cols].to_string(index=False)
    )

    print_header("RECURRENCE CLASS DEFINITIONS")
    print("Recurrence_Class_1_dual_null_recurrent")
    print("  = recurrent across dyads under both trial-shuffle and time-shift criteria")
    print("Recurrence_Class_2_trial_recurrent")
    print("  = recurrent across dyads under trial-shuffle, but not time-shift")
    print("Recurrence_Class_3_partial_or_heterogeneous")
    print("  = some dyadic support and/or directional consistency, but not full recurrence")
    print("Recurrence_Class_4_weak_or_idiosyncratic")
    print("  = weak, sparse, or dyad-specific signal")

# =============================================================================
# Write
# =============================================================================

def write_outputs(df_inf):
    print_header("WRITING OUTPUTS")

    df_inf.to_csv(OUT_MAIN, index=False)
    print(f"[WRITE] {OUT_MAIN}")

    sig = df_inf[
        (df_inf["q_recur_trial_005"] < 0.05) |
        (df_inf["q_recur_ts_005"] < 0.05)
    ].copy()
    sig.to_csv(OUT_SIG, index=False)
    print(f"[WRITE] {OUT_SIG}")

    part = (
        df_inf.assign(
            trial_recurrent=df_inf["q_recur_trial_005"] < 0.05,
            ts_recurrent=df_inf["q_recur_ts_005"] < 0.05
        )
        .groupby(["partition", "atom"])
        .agg(
            n_targets=("target_id", "size"),
            n_trial_recurrent=("trial_recurrent", "sum"),
            n_ts_recurrent=("ts_recurrent", "sum"),
        )
        .reset_index()
        .sort_values(["partition", "atom"])
    )
    part.to_csv(OUT_PART, index=False)
    print(f"[WRITE] {OUT_PART}")

    contrast = (
        df_inf.assign(
            trial_recurrent=df_inf["q_recur_trial_005"] < 0.05,
            ts_recurrent=df_inf["q_recur_ts_005"] < 0.05
        )
        .groupby(["contrast", "atom"])
        .agg(
            n_targets=("target_id", "size"),
            n_trial_recurrent=("trial_recurrent", "sum"),
            n_ts_recurrent=("ts_recurrent", "sum"),
        )
        .reset_index()
        .sort_values(["contrast", "atom"])
    )
    contrast.to_csv(OUT_CONTRAST, index=False)
    print(f"[WRITE] {OUT_CONTRAST}")

    out_json = {
        "n_targets_total": int(len(df_inf)),
        "n_trial_recurrent_fdr005": int((df_inf["q_recur_trial_005"] < 0.05).sum()),
        "n_ts_recurrent_fdr005": int((df_inf["q_recur_ts_005"] < 0.05).sum()),
        "recurrence_class_counts": df_inf["recurrence_class"].value_counts(dropna=False).to_dict(),
        "files": {
            "main": str(OUT_MAIN),
            "recurrent_features_significant": str(OUT_SIG),
            "by_partition_atom": str(OUT_PART),
            "by_contrast_atom": str(OUT_CONTRAST),
        }
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"[WRITE] {OUT_JSON}")

# =============================================================================
# Main
# =============================================================================

def main():
    df_targets, df_obs, df_trial, df_ts, summary, runsig = load_all()
    validate(df_targets, df_obs, df_trial, df_ts)
    df_obs, df_trial, df_ts, sig = filter_run_sig(df_obs, df_trial, df_ts, runsig)

    df_inf = build_target_recurrence(df_targets, df_obs, df_trial, df_ts)
    print_console_summary(df_inf, summary)
    write_outputs(df_inf)

    print_header("DONE")
    print(f"[INFO] Run signature used: {sig}")
    print(f"[INFO] Main output: {OUT_MAIN}")

if __name__ == "__main__":
    main()
