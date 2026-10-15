"""
Script: 08_phiid_overlap_homologous_vs_crosshemi.py

Purpose
-------
Compare significant ΦID clusters derived from homologous and axis geometries.

This script:
1. Loads each geometry's significant_clusters_summary.csv
2. Recomputes peak metrics from the original group tensors
3. Prints summary outputs for each geometry
4. Quantifies overlap between homologous and axis clusters for matched
   (contrast, atom, band, roi) features using inclusive millisecond windows
5. Writes enriched outputs and overlap tables to disk

Pipeline role
-------------
Primary comparative ΦID analysis script for homologous-versus-axis overlap.

Required inputs
---------------
For each geometry:
- significant_clusters_summary.csv
- group_phiid_tensor.dat
- group_phiid_labels.pkl

Assumptions
-----------
- Blocks:
    ["Leader-Mirroring","Leader-Matching","Follower-Matching","Follower-Mirroring"]
- Atom labels:
    ["rtr","xtx","yty","sts"]
- Contrast sign convention:
    always lhs - rhs, as defined in the upstream cluster scripts
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# =============================================================================
# User configuration
# =============================================================================

HOMOL_GROUP_DIR = Path("PATH_TO_HOMOLOGOUS_GROUP_RESULTS")
AXIS_GROUP_DIR  = Path("PATH_TO_AXIS_GROUP_RESULTS")

HOMOL_RESULTS_DIR = HOMOL_GROUP_DIR / "cluster_results"
AXIS_RESULTS_DIR  = AXIS_GROUP_DIR / "cluster_results"

HOMOL_CSV = HOMOL_RESULTS_DIR / "significant_clusters_summary.csv"
AXIS_CSV  = AXIS_RESULTS_DIR  / "significant_clusters_summary.csv"

OUT_DIR = AXIS_RESULTS_DIR / "grand_summary_vs_homol"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP_MS = 1.0
TOPK_PEAKS = 10

ATOM_FILTER = None
BAND_FILTER = None
CONTRAST_FILTER = None

REQUIRED_COLS = [
    "contrast", "atom", "band", "roi",
    "t_start_ms", "t_end_ms", "p_value",
    "mean_effect_in_cluster", "cluster_stat",
    "n_dyads", "n_times_in_cluster",
]

# =============================================================================
# Utility functions
# =============================================================================

def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

def read_csv_strict(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing} in: {path}")
    return df

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def interval_overlap_inclusive(a_start, a_end, b_start, b_end):
    inter = min(a_end, b_end) - max(a_start, b_start) + 1.0
    overlap = max(0.0, inter)
    if overlap > 0:
        union = max(a_end, b_end) - min(a_start, b_start) + 1.0
        jacc = overlap / max(1e-9, union)
    else:
        jacc = 0.0
    return overlap, jacc

def sign(x):
    if np.isnan(x) or x == 0:
        return 0
    return 1 if x > 0 else -1

def contrast_direction_label(contrast_name: str, effect_value: float) -> str:
    s = sign(effect_value)

    if contrast_name == "Matching_minus_Mirroring_pooled":
        return "Matching > Mirroring" if s > 0 else ("Mirroring > Matching" if s < 0 else "Zero")
    if contrast_name == "Mirroring_minus_Matching_pooled":
        return "Mirroring > Matching" if s > 0 else ("Matching > Mirroring" if s < 0 else "Zero")

    if contrast_name == "Leader_minus_Follower_Matching":
        return "Leader > Follower" if s > 0 else ("Follower > Leader" if s < 0 else "Zero")
    if contrast_name == "Leader_minus_Follower_Mirroring":
        return "Leader > Follower" if s > 0 else ("Follower > Leader" if s < 0 else "Zero")

    if contrast_name == "RoleDiff_Matching_minus_RoleDiff_Mirroring":
        return "(L-F Matching) > (L-F Mirroring)" if s > 0 else (
            "(L-F Mirroring) > (L-F Matching)" if s < 0 else "Zero"
        )

    return "Positive effect" if s > 0 else ("Negative effect" if s < 0 else "Zero")

def build_feature_key(df: pd.DataFrame) -> pd.Series:
    return df[["contrast", "atom", "band", "roi"]].astype(str).agg("||".join, axis=1)

# =============================================================================
# Tensor loading and contrast helpers
# =============================================================================

def load_group_tensor(group_dir: Path):
    tensor_path = group_dir / "group_phiid_tensor.dat"
    labels_path = group_dir / "group_phiid_labels.pkl"

    assert_exists(tensor_path, "Tensor")
    assert_exists(labels_path, "Labels")

    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    shape = tuple(labels["shape"])
    if len(shape) != 6:
        raise RuntimeError(f"Expected 6D tensor shape, got {shape} in {labels_path}")

    X = np.memmap(tensor_path, dtype=np.float32, mode="r", shape=shape)

    block_labels = labels["block_labels"]
    roi_labels   = labels["roi_labels"]
    band_labels  = labels["band_labels"]
    atom_labels  = labels["atom_labels"]
    dyad_labels  = labels["dyad_labels"]
    sfreq        = float(labels.get("sfreq", 1000.0))

    block_to_idx = {b: i for i, b in enumerate(block_labels)}
    required_blocks = [
        "Leader-Mirroring",
        "Leader-Matching",
        "Follower-Matching",
        "Follower-Mirroring",
    ]
    missing = [b for b in required_blocks if b not in block_to_idx]
    if missing:
        raise RuntimeError(f"Missing required blocks in labels: {missing}")

    b_LM = block_to_idx["Leader-Mirroring"]
    b_LK = block_to_idx["Leader-Matching"]
    b_FK = block_to_idx["Follower-Matching"]
    b_FM = block_to_idx["Follower-Mirroring"]

    contrasts = {
        "Mirroring_minus_Matching_pooled": dict(lhs=[b_LM, b_FM], rhs=[b_LK, b_FK]),
        "Matching_minus_Mirroring_pooled": dict(lhs=[b_LK, b_FK], rhs=[b_LM, b_FM]),
        "Leader_minus_Follower_Mirroring": dict(lhs=[b_LM], rhs=[b_FM]),
        "Leader_minus_Follower_Matching": dict(lhs=[b_LK], rhs=[b_FK]),
        "RoleDiff_Matching_minus_RoleDiff_Mirroring": dict(
            lhs=[b_LK], rhs=[b_FK], extra_lhs=[b_FM], extra_rhs=[b_LM]
        ),
    }

    return dict(
        X=X,
        labels=labels,
        sfreq=sfreq,
        roi_to_idx={r: i for i, r in enumerate(roi_labels)},
        band_to_idx={b: i for i, b in enumerate(band_labels)},
        atom_to_idx={a: i for i, a in enumerate(atom_labels)},
        dyad_labels=dyad_labels,
        block_labels=block_labels,
        band_labels=band_labels,
        roi_labels=roi_labels,
        atom_labels=atom_labels,
        CONTRASTS=contrasts,
    )

def safe_mean_over_blocks(arr_dbt, block_indices):
    sub = arr_dbt[:, block_indices, :]
    return np.nanmean(sub, axis=1)

def build_contrast_timeseries(arr_dbt, contrast_def):
    lhs_mean = safe_mean_over_blocks(arr_dbt, contrast_def["lhs"])
    rhs_mean = safe_mean_over_blocks(arr_dbt, contrast_def["rhs"])
    D = lhs_mean - rhs_mean
    if "extra_lhs" in contrast_def and "extra_rhs" in contrast_def:
        el = safe_mean_over_blocks(arr_dbt, contrast_def["extra_lhs"])
        er = safe_mean_over_blocks(arr_dbt, contrast_def["extra_rhs"])
        D = D - (el - er)
    return D

def compute_peak_metrics_for_row(row, G, t0_ms=0.0):
    X = G["X"]
    sfreq = G["sfreq"]
    dt_ms = 1000.0 / sfreq

    contrast = str(row["contrast"])
    atom = str(row["atom"])
    band = str(row["band"])
    roi = str(row["roi"])

    if contrast not in G["CONTRASTS"]:
        return dict(
            peak_effect=np.nan, peak_time_ms=np.nan,
            peak_d=np.nan, peak_d_time_ms=np.nan,
            mean_effect_in_window=np.nan,
            window_n_times=np.nan,
            flag="missing_contrast_def",
        )

    if atom not in G["atom_to_idx"] or band not in G["band_to_idx"] or roi not in G["roi_to_idx"]:
        return dict(
            peak_effect=np.nan, peak_time_ms=np.nan,
            peak_d=np.nan, peak_d_time_ms=np.nan,
            mean_effect_in_window=np.nan,
            window_n_times=np.nan,
            flag="missing_label_index",
        )

    roi_idx  = G["roi_to_idx"][roi]
    band_idx = G["band_to_idx"][band]
    atom_idx = G["atom_to_idx"][atom]

    arr_dbt = X[:, :, roi_idx, band_idx, :, atom_idx]
    D = build_contrast_timeseries(arr_dbt, G["CONTRASTS"][contrast])

    t_start = safe_float(row["t_start_ms"])
    t_end   = safe_float(row["t_end_ms"])
    if np.isnan(t_start) or np.isnan(t_end):
        return dict(
            peak_effect=np.nan, peak_time_ms=np.nan,
            peak_d=np.nan, peak_d_time_ms=np.nan,
            mean_effect_in_window=np.nan,
            window_n_times=np.nan,
            flag="bad_time_window",
        )

    i0 = int(np.clip(np.round((t_start - t0_ms) / dt_ms), 0, D.shape[1] - 1))
    i1 = int(np.clip(np.round((t_end - t0_ms) / dt_ms), 0, D.shape[1] - 1))
    if i1 < i0:
        i0, i1 = i1, i0

    Dw = D[:, i0:i1 + 1]

    mean_ts = np.nanmean(Dw, axis=0)
    if mean_ts.size == 0 or np.all(np.isnan(mean_ts)):
        return dict(
            peak_effect=np.nan, peak_time_ms=np.nan,
            peak_d=np.nan, peak_d_time_ms=np.nan,
            mean_effect_in_window=np.nan,
            window_n_times=int(Dw.shape[1]),
            flag="all_nan_window",
        )

    peak_idx_rel = int(np.nanargmax(np.abs(mean_ts)))
    peak_effect = float(mean_ts[peak_idx_rel])
    peak_time_ms = float(t0_ms + (i0 + peak_idx_rel) * dt_ms)
    mean_effect_in_window = float(np.nanmean(mean_ts))

    mu = np.nanmean(Dw, axis=0)
    sd = np.nanstd(Dw, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ts = np.where(sd > 0, mu / sd, np.nan)

    if np.all(np.isnan(d_ts)):
        peak_d = np.nan
        peak_d_time_ms = np.nan
    else:
        peak_d_rel = int(np.nanargmax(np.abs(d_ts)))
        peak_d = float(d_ts[peak_d_rel])
        peak_d_time_ms = float(t0_ms + (i0 + peak_d_rel) * dt_ms)

    return dict(
        peak_effect=peak_effect,
        peak_time_ms=peak_time_ms,
        peak_d=peak_d,
        peak_d_time_ms=peak_d_time_ms,
        mean_effect_in_window=mean_effect_in_window,
        window_n_times=int(Dw.shape[1]),
        flag="ok",
    )

# =============================================================================
# Enrichment and summaries
# =============================================================================

def enrich_summary(df, G, geometry_label: str):
    df = df.copy()

    if ATOM_FILTER is not None:
        df = df[df["atom"].astype(str) == str(ATOM_FILTER)].copy()
    if BAND_FILTER is not None:
        df = df[df["band"].astype(str) == str(BAND_FILTER)].copy()
    if CONTRAST_FILTER is not None:
        df = df[df["contrast"].astype(str) == str(CONTRAST_FILTER)].copy()

    df["t_start_ms"] = df["t_start_ms"].astype(float)
    df["t_end_ms"]   = df["t_end_ms"].astype(float)
    df["p_value"]    = df["p_value"].astype(float)
    df["mean_effect_in_cluster"] = df["mean_effect_in_cluster"].astype(float)

    df["feature_key"] = build_feature_key(df)
    df["direction"] = [
        contrast_direction_label(c, e)
        for c, e in zip(df["contrast"].astype(str), df["mean_effect_in_cluster"].astype(float))
    ]

    peak_rows = [compute_peak_metrics_for_row(r, G, t0_ms=0.0) for _, r in df.iterrows()]
    peak_df = pd.DataFrame(peak_rows)

    out = pd.concat([df.reset_index(drop=True), peak_df.reset_index(drop=True)], axis=1)
    out["geometry"] = geometry_label
    out["abs_peak_effect"] = out["peak_effect"].abs()
    out["abs_peak_d"] = out["peak_d"].abs()
    return out

def print_console_summary(enriched_df: pd.DataFrame, title: str):
    df = enriched_df.copy()

    print("\n" + "=" * 90)
    print(f"Significant Cluster Summary ({title}) - {len(df)} clusters")
    print("=" * 90)
    print(f"ROIs with clusters   : {df['roi'].nunique()}")
    print(f"Atoms present        : {', '.join(sorted(df['atom'].astype(str).unique()))}")
    print(f"Bands present        : {', '.join(sorted(df['band'].astype(str).unique()))}")

    print("\n-- Counts by Contrast x Direction --")
    ctab = (
        df.groupby(["contrast", "direction"])
          .size()
          .reset_index(name="n")
          .sort_values(["contrast", "n"], ascending=[True, False])
    )
    for contrast in ctab["contrast"].unique():
        print(f"\n[{contrast}]")
        sub = ctab[ctab["contrast"] == contrast]
        for _, r in sub.iterrows():
            print(f"  {str(r['direction']):28s}: {int(r['n'])} clusters")

    print("\n-- Top ROIs by number of significant clusters --")
    for roi, n in df["roi"].value_counts().head(15).items():
        print(f"  {roi:30s}: {int(n)}")

    print("\n-- Band distribution --")
    for band, n in df["band"].value_counts().items():
        print(f"  {band:10s}: {int(n)}")

    print(f"\n-- Top {TOPK_PEAKS} strongest peaks by |peak_d| --")
    top_d = df.sort_values("abs_peak_d", ascending=False).head(TOPK_PEAKS)
    for _, r in top_d.iterrows():
        print(
            f"  {r['roi']:26s} | {r['band']:9s} | {r['atom']:3s} | "
            f"{r['contrast'][:30]:30s} | {r['direction'][:26]:26s} | "
            f"peak d={r['peak_d']:+.3f} @ {r['peak_d_time_ms']:.0f} ms | "
            f"p={r['p_value']:.4f} | {r['t_start_ms']:.0f}-{r['t_end_ms']:.0f} ms"
        )

# =============================================================================
# Overlap matching
# =============================================================================

def match_overlap(homol_df: pd.DataFrame, axis_df: pd.DataFrame):
    key_cols = ["contrast", "atom", "band", "roi"]

    homol = homol_df.copy()
    axis = axis_df.copy()

    homol["_key"] = homol[key_cols].astype(str).agg("||".join, axis=1)
    axis["_key"]  = axis[key_cols].astype(str).agg("||".join, axis=1)

    homol_groups = {k: g for k, g in homol.groupby("_key")}
    axis_groups  = {k: g for k, g in axis.groupby("_key")}

    matches = []
    homol_seen = set()
    axis_seen = set()

    for key, gH in homol_groups.items():
        gA = axis_groups.get(key)
        if gA is None:
            continue

        for idxH, rH in gH.iterrows():
            for idxA, rA in gA.iterrows():
                ov, jacc = interval_overlap_inclusive(
                    float(rH["t_start_ms"]), float(rH["t_end_ms"]),
                    float(rA["t_start_ms"]), float(rA["t_end_ms"]),
                )

                if ov >= OVERLAP_MS:
                    matches.append({
                        "contrast": rH["contrast"],
                        "atom": rH["atom"],
                        "band": rH["band"],
                        "roi": rH["roi"],
                        "overlap_ms": ov,
                        "jaccard_time": jacc,

                        "h_p": rH["p_value"],
                        "h_t_start": rH["t_start_ms"],
                        "h_t_end": rH["t_end_ms"],
                        "h_direction": rH["direction"],
                        "h_mean_effect": rH["mean_effect_in_cluster"],
                        "h_peak_effect": rH["peak_effect"],
                        "h_peak_time": rH["peak_time_ms"],
                        "h_peak_d": rH["peak_d"],
                        "h_peak_d_time": rH["peak_d_time_ms"],

                        "a_p": rA["p_value"],
                        "a_t_start": rA["t_start_ms"],
                        "a_t_end": rA["t_end_ms"],
                        "a_direction": rA["direction"],
                        "a_mean_effect": rA["mean_effect_in_cluster"],
                        "a_peak_effect": rA["peak_effect"],
                        "a_peak_time": rA["peak_time_ms"],
                        "a_peak_d": rA["peak_d"],
                        "a_peak_d_time": rA["peak_d_time_ms"],

                        "sign_agree_mean_effect": int(sign(rH["mean_effect_in_cluster"]) == sign(rA["mean_effect_in_cluster"])),
                        "sign_agree_peak_effect": int(sign(rH["peak_effect"]) == sign(rA["peak_effect"])),
                        "sign_agree_peak_d": int(sign(rH["peak_d"]) == sign(rA["peak_d"])),
                    })
                    homol_seen.add(idxH)
                    axis_seen.add(idxA)

    matched_df = pd.DataFrame(matches)

    if len(matched_df):
        matched_df["abs_h_peak_d"] = matched_df["h_peak_d"].abs()
        matched_df["abs_a_peak_d"] = matched_df["a_peak_d"].abs()
        matched_df["max_abs_peak_d"] = matched_df[["abs_h_peak_d", "abs_a_peak_d"]].max(axis=1)
        matched_df = matched_df.sort_values(
            ["overlap_ms", "max_abs_peak_d", "jaccard_time"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    homol_only = homol.loc[~homol.index.isin(homol_seen)].copy()
    axis_only  = axis.loc[~axis.index.isin(axis_seen)].copy()

    return matched_df, homol_only, axis_only

# =============================================================================
# Main
# =============================================================================

def main():
    assert_exists(HOMOL_CSV, "HOMOLOGOUS significant_clusters_summary.csv")
    assert_exists(AXIS_CSV,  "AXIS significant_clusters_summary.csv")

    homol_raw = read_csv_strict(HOMOL_CSV)
    axis_raw  = read_csv_strict(AXIS_CSV)

    G_h = load_group_tensor(HOMOL_GROUP_DIR)
    G_a = load_group_tensor(AXIS_GROUP_DIR)

    homol = enrich_summary(homol_raw, G_h, geometry_label="homol_roleXY")
    axis  = enrich_summary(axis_raw,  G_a, geometry_label="axis_roleXY")

    homol_out = OUT_DIR / "enriched_significant_clusters_homol.csv"
    axis_out  = OUT_DIR / "enriched_significant_clusters_axis.csv"
    homol.to_csv(homol_out, index=False)
    axis.to_csv(axis_out, index=False)

    print_console_summary(homol, title="HOMOLOGOUS")
    print_console_summary(axis, title="AXIS")

    matched_df, homol_only, axis_only = match_overlap(homol, axis)

    matched_out = OUT_DIR / "matched_pairs_homol_vs_axis.csv"
    homol_only_out = OUT_DIR / "homol_only.csv"
    axis_only_out  = OUT_DIR / "axis_only.csv"

    matched_df.to_csv(matched_out, index=False)
    homol_only.to_csv(homol_only_out, index=False)
    axis_only.to_csv(axis_only_out, index=False)

    print("\n" + "=" * 90)
    print("Overlap Summary (HOMOLOGOUS vs AXIS)")
    print("=" * 90)
    print(f"HOMOLOGOUS clusters (input): {len(homol)}")
    print(f"AXIS clusters (input)      : {len(axis)}")
    print(f"Matched pairs (overlap >= {OVERLAP_MS:.0f} ms, inclusive): {len(matched_df)}")
    print(f"HOMOLOGOUS-only clusters   : {len(homol_only)}")
    print(f"AXIS-only clusters         : {len(axis_only)}")

    if len(matched_df):
        print(f"\nSign agreement (mean_effect_in_cluster): {100.0 * float(matched_df['sign_agree_mean_effect'].mean()):.1f}%")
        print(f"Sign agreement (peak_effect)          : {100.0 * float(matched_df['sign_agree_peak_effect'].mean()):.1f}%")
        print(f"Sign agreement (peak_d)               : {100.0 * float(matched_df['sign_agree_peak_d'].mean()):.1f}%")

    cd = (
        pd.concat([homol, axis], ignore_index=True)
          .groupby(["geometry", "contrast", "direction"])
          .size()
          .reset_index(name="n")
          .sort_values(["geometry", "contrast", "n"], ascending=[True, True, False])
    )
    cd.to_csv(OUT_DIR / "counts_contrast_direction_by_geometry.csv", index=False)

    acd = (
        pd.concat([homol, axis], ignore_index=True)
          .groupby(["geometry", "atom", "contrast", "direction"])
          .size()
          .reset_index(name="n")
          .sort_values(["geometry", "atom", "contrast", "n"], ascending=[True, True, True, False])
    )
    acd.to_csv(OUT_DIR / "counts_atom_contrast_direction_by_geometry.csv", index=False)

    roi_counts = (
        pd.concat([homol, axis], ignore_index=True)
          .groupby(["geometry", "roi"])
          .size()
          .reset_index(name="n")
          .sort_values(["geometry", "n"], ascending=[True, False])
    )
    roi_counts.to_csv(OUT_DIR / "roi_cluster_counts_by_geometry.csv", index=False)

    top_d = (
        pd.concat([homol, axis], ignore_index=True)
          .sort_values("abs_peak_d", ascending=False)
          .head(200)
    )
    top_d.to_csv(OUT_DIR / "top_peaks_by_abs_peak_d.csv", index=False)

    top_e = (
        pd.concat([homol, axis], ignore_index=True)
          .sort_values("abs_peak_effect", ascending=False)
          .head(200)
    )
    top_e.to_csv(OUT_DIR / "top_peaks_by_abs_peak_effect.csv", index=False)

    print("\n" + "=" * 90)
    print("FILES WRITTEN")
    print("=" * 90)
    print(f"[WRITE] {homol_out}")
    print(f"[WRITE] {axis_out}")
    print(f"[WRITE] {matched_out}")
    print(f"[WRITE] {homol_only_out}")
    print(f"[WRITE] {axis_only_out}")
    print(f"[WRITE] {OUT_DIR / 'counts_contrast_direction_by_geometry.csv'}")
    print(f"[WRITE] {OUT_DIR / 'counts_atom_contrast_direction_by_geometry.csv'}")
    print(f"[WRITE] {OUT_DIR / 'roi_cluster_counts_by_geometry.csv'}")
    print(f"[WRITE] {OUT_DIR / 'top_peaks_by_abs_peak_d.csv'}")
    print(f"[WRITE] {OUT_DIR / 'top_peaks_by_abs_peak_effect.csv'}")
    print("\nDone.")

if __name__ == "__main__":
    main()
