# -*- coding: utf-8 -*-
"""
Script: 11_phiid_role_swap_validation.py

Purpose
-------
Validate whether strict surviving ΦID features remain stable across the role
swap built into the 2 × 2 design.

This script evaluates both:
1. Block-half stability for the features retained after strict feature filtering
2. Role-swap stability for unique-information atoms (xtx, yty)

Design logic
------------
Block structure:
    Block 1 = Leader + Mirroring
    Block 2 = Leader + Matching
    Block 3 = Follower + Matching
    Block 4 = Follower + Mirroring

Role-consistent XY mapping:
    Blocks 1-2: X = P1(Leader), Y = P2(Follower)
    Blocks 3-4: X = P2(Leader), Y = P1(Follower)

Interpretation:
    xtx = Leader-unique information
    yty = Follower-unique information

Tests performed
---------------
1. Block-half stability (features retained after filtering)
   early_half = mean(blocks 1, 2)
   late_half  = mean(blocks 3, 4)

2. Role-swap stability (xtx and yty only)
   xtx:
       P1-Leader (blocks 1-2) vs P2-Leader (blocks 3-4)
   yty:
       P2-Follower (blocks 1-2) vs P1-Follower (blocks 3-4)

Important limitation
--------------------
This script does not reconstruct a deeper participant-native tensor. It operates
on the existing roleXY tensor level.

Label recovery
--------------
Some dyad tensor files may not store roi_labels, band_labels, or atom_labels.
This script therefore:
1. uses dyad-file labels if present
2. otherwise falls back to group labels files

Inputs
------
1. Strict survivor CSVs
   - tier_A_features_strict.csv
   - tier_B_features_strict.csv

2. Dyad-level roleXY tensors
   - dyad_*_tensor_roleXY.pkl
   - dyad_*_tensor_axis_roleXY.pkl

3. Group labels files
   - group_phiid_labels.pkl (homologous)
   - group_phiid_labels.pkl (axis)

Outputs
-------
Writes:
- Tier A per-dyad values + feature summary
- Tier B per-dyad values + feature summary
- Combined per-dyad values + feature summary

Default analysis setting
------------------------
By default, the script restricts the analysed features to xtx and yty via
RUN_ONLY_UNIQUE_ATOMS = True. This matches the intended interpretation of
role-specific atoms.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# =============================================================================
# USER SETTINGS
# =============================================================================

ROBUSTNESS_DIR = r"PATH_TO_PHIID_FULL_GEOMETRIC_ROBUSTNESS"
DYAD_TENSOR_DIR = r"PATH_TO_DYAD_PHIID_TENSORS"

TIER_A_CSV = os.path.join(ROBUSTNESS_DIR, "tier_A_features_strict.csv")
TIER_B_CSV = os.path.join(ROBUSTNESS_DIR, "tier_B_features_strict.csv")

# Dyad tensor files
HOMOL_GLOB = os.path.join(DYAD_TENSOR_DIR, "dyad_*_tensor_roleXY.pkl")
AXIS_GLOB = os.path.join(DYAD_TENSOR_DIR, "dyad_*_tensor_axis_roleXY.pkl")

# Group label files for fallback
GROUP_LABELS_HOMOL = r"PATH_TO_GROUP_LABELS_HOMOLOGOUS"
GROUP_LABELS_AXIS = r"PATH_TO_GROUP_LABELS_AXIS"

# Filters
RUN_ONLY_UNIQUE_ATOMS = True
RESTRICT_PARTITIONS = None   # e.g. ["overlap"] or ["axis_only", "overlap"] or None

# Time axis assumption: 1999 samples from -1000 ms in 1 ms increments
TIME_START_MS_DEFAULT = -1000

ALPHA = 0.05

# Output folder
OUT_DIR = os.path.join(ROBUSTNESS_DIR, "role_swap_validation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PER_DYAD_A = os.path.join(OUT_DIR, "role_swap_validation_per_dyad_tierA.csv")
OUT_SUMMARY_A = os.path.join(OUT_DIR, "role_swap_validation_summary_tierA.csv")

OUT_PER_DYAD_B = os.path.join(OUT_DIR, "role_swap_validation_per_dyad_tierB.csv")
OUT_SUMMARY_B = os.path.join(OUT_DIR, "role_swap_validation_summary_tierB.csv")

OUT_PER_DYAD_ALL = os.path.join(OUT_DIR, "role_swap_validation_per_dyad_combined.csv")
OUT_SUMMARY_ALL = os.path.join(OUT_DIR, "role_swap_validation_summary_combined.csv")

# =============================================================================
# HELPERS
# =============================================================================

def build_time_vector(n_times: int, tmin_ms: int = TIME_START_MS_DEFAULT) -> np.ndarray:
    return np.arange(tmin_ms, tmin_ms + n_times)


def find_time_indices(time_vec: np.ndarray, start_ms: float, stop_ms: float) -> np.ndarray:
    idx = np.where((time_vec >= start_ms) & (time_vec <= stop_ms))[0]
    if len(idx) == 0:
        raise ValueError(f"No time points found between {start_ms} and {stop_ms} ms.")
    return idx


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tensor_path_maps(homol_glob: str, axis_glob: str):
    homol_paths = sorted(glob.glob(homol_glob))
    axis_paths = sorted(glob.glob(axis_glob))

    if not homol_paths:
        raise FileNotFoundError(f"No homologous roleXY tensor files found: {homol_glob}")
    if not axis_paths:
        raise FileNotFoundError(f"No axis roleXY tensor files found: {axis_glob}")

    homol_map = {}
    axis_map = {}

    for p in homol_paths:
        obj = load_pickle(p)
        if "dyad_id" not in obj:
            raise KeyError(f"'dyad_id' missing from file: {p}")
        homol_map[obj["dyad_id"]] = p

    for p in axis_paths:
        obj = load_pickle(p)
        if "dyad_id" not in obj:
            raise KeyError(f"'dyad_id' missing from file: {p}")
        axis_map[obj["dyad_id"]] = p

    return homol_map, axis_map


def pick_tensor_path(partition: str, dyad_id: int, homol_map: dict, axis_map: dict) -> str:
    if partition in ("axis_only", "overlap"):
        if dyad_id not in axis_map:
            raise KeyError(f"Dyad {dyad_id} missing from axis tensor map.")
        return axis_map[dyad_id]

    if partition == "homol_only":
        if dyad_id not in homol_map:
            raise KeyError(f"Dyad {dyad_id} missing from homologous tensor map.")
        return homol_map[dyad_id]

    raise ValueError(f"Unknown partition: {partition}")


def _extract_labels_from_obj(obj):
    """
    Try several common layouts for labels in a pickle object.
    """
    labels = {}

    if isinstance(obj, dict):
        if "roi_labels" in obj:
            labels["roi_labels"] = list(obj["roi_labels"])
        if "band_labels" in obj:
            labels["band_labels"] = list(obj["band_labels"])
        if "atom_labels" in obj:
            labels["atom_labels"] = list(obj["atom_labels"])

        for nested_key in ["labels", "meta", "metadata"]:
            if nested_key in obj and isinstance(obj[nested_key], dict):
                nested = obj[nested_key]
                if "roi_labels" in nested and "roi_labels" not in labels:
                    labels["roi_labels"] = list(nested["roi_labels"])
                if "band_labels" in nested and "band_labels" not in labels:
                    labels["band_labels"] = list(nested["band_labels"])
                if "atom_labels" in nested and "atom_labels" not in labels:
                    labels["atom_labels"] = list(nested["atom_labels"])

    return labels


def load_group_labels(path: str):
    """
    Load fallback labels from a group labels file.
    Atom labels default to the known ΦID atom order if missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Group labels file not found: {path}")

    obj = load_pickle(path)
    labels = _extract_labels_from_obj(obj)

    if "roi_labels" not in labels or "band_labels" not in labels:
        raise KeyError(f"Could not recover roi_labels/band_labels from group labels file: {path}")

    if "atom_labels" not in labels:
        labels["atom_labels"] = ["rtr", "xtx", "yty", "sts"]

    return labels


def get_labels_for_obj(obj: dict, partition: str, axis_group_labels: dict, homol_group_labels: dict):
    """
    Use dyad-file labels if present; otherwise use partition-appropriate group labels.
    """
    local_labels = _extract_labels_from_obj(obj)

    if all(k in local_labels for k in ["roi_labels", "band_labels", "atom_labels"]):
        return local_labels

    if partition in ("axis_only", "overlap"):
        return axis_group_labels

    if partition == "homol_only":
        return homol_group_labels

    raise ValueError(f"Unknown partition for label fallback: {partition}")


def validate_feature_labels(labels: dict, roi_name: str, band_name: str, atom_name: str):
    roi_labels = labels["roi_labels"]
    band_labels = labels["band_labels"]
    atom_labels = labels["atom_labels"]

    if roi_name not in roi_labels:
        raise ValueError(f"ROI '{roi_name}' not found in labels. Example labels: {roi_labels[:10]}")
    if band_name not in band_labels:
        raise ValueError(f"Band '{band_name}' not found in labels. Labels: {band_labels}")
    if atom_name not in atom_labels:
        raise ValueError(f"Atom '{atom_name}' not found in labels. Labels: {atom_labels}")


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) == 0:
        return {
            "n": 0,
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_diff": np.nan,
            "sd_diff": np.nan,
            "t": np.nan,
            "p": np.nan,
            "mean_abs_diff": np.nan,
        }

    if len(a) == 1:
        diff = a - b
        return {
            "n": 1,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(diff)),
            "sd_diff": np.nan,
            "t": np.nan,
            "p": np.nan,
            "mean_abs_diff": float(np.mean(np.abs(diff))),
        }

    diff = a - b
    tt = ttest_rel(a, b, nan_policy="omit")

    return {
        "n": int(len(a)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "sd_diff": float(np.std(diff, ddof=1)),
        "t": float(tt.statistic),
        "p": float(tt.pvalue),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def extract_feature_from_tensor(
    obj: dict,
    labels: dict,
    roi_name: str,
    band_name: str,
    atom_name: str,
    win_start_ms: float,
    win_stop_ms: float,
):
    tensor = obj["phiid_tensor"]
    validate_feature_labels(labels, roi_name, band_name, atom_name)

    roi_idx = labels["roi_labels"].index(roi_name)
    band_idx = labels["band_labels"].index(band_name)
    atom_idx = labels["atom_labels"].index(atom_name)

    n_times = tensor.shape[3]
    time_vec = build_time_vector(n_times, TIME_START_MS_DEFAULT)
    tidx = find_time_indices(time_vec, win_start_ms, win_stop_ms)

    # Early half: blocks 1-2
    # Late half:  blocks 3-4
    early_vals = tensor[[0, 1], roi_idx, band_idx][:, tidx, atom_idx]
    late_vals = tensor[[2, 3], roi_idx, band_idx][:, tidx, atom_idx]

    early_mean = float(np.mean(early_vals))
    late_mean = float(np.mean(late_vals))

    out = {
        "early_mean": early_mean,
        "late_mean": late_mean,
        "P1_Leader": np.nan,
        "P2_Leader": np.nan,
        "P1_Follower": np.nan,
        "P2_Follower": np.nan,
    }

    if atom_name == "xtx":
        out["P1_Leader"] = early_mean
        out["P2_Leader"] = late_mean

    elif atom_name == "yty":
        out["P2_Follower"] = early_mean
        out["P1_Follower"] = late_mean

    return out


def load_and_filter_features(csv_path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{label} CSV not found: {csv_path}")

    features = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {label}: {len(features)} rows")

    required_cols = [
        "target_id", "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms", "tier", "feature_key"
    ]
    missing = [c for c in required_cols if c not in features.columns]
    if missing:
        raise KeyError(f"{label} CSV missing required columns: {missing}")

    if RESTRICT_PARTITIONS is not None:
        features = features[features["partition"].isin(RESTRICT_PARTITIONS)].copy()
        print(f"[INFO] {label} after partition filter: {len(features)} rows")

    if RUN_ONLY_UNIQUE_ATOMS:
        features = features[features["atom"].isin(["xtx", "yty"])].copy()
        print(f"[INFO] {label} after unique-atom filter (xtx/yty): {len(features)} rows")

    features = features.reset_index(drop=True)
    features["analysis_set"] = label
    return features


def run_one_feature_set(
    features: pd.DataFrame,
    set_label: str,
    homol_map: dict,
    axis_map: dict,
    common_dyads: list,
    axis_group_labels: dict,
    homol_group_labels: dict,
):
    rows = []
    n_total = len(features) * len(common_dyads)
    counter = 0

    print("=" * 100)
    print(f"RUNNING: {set_label}")
    print("=" * 100)
    print(f"[INFO] Features in set: {len(features)}")
    print(f"[INFO] Common dyads: {common_dyads}")
    print()

    print("[INFO] Extracting per-dyad feature values...")

    for _, feat in features.iterrows():
        for dyad_id in common_dyads:
            tensor_path = pick_tensor_path(
                partition=feat["partition"],
                dyad_id=dyad_id,
                homol_map=homol_map,
                axis_map=axis_map
            )

            obj = load_pickle(tensor_path)

            labels = get_labels_for_obj(
                obj=obj,
                partition=feat["partition"],
                axis_group_labels=axis_group_labels,
                homol_group_labels=homol_group_labels
            )

            vals = extract_feature_from_tensor(
                obj=obj,
                labels=labels,
                roi_name=feat["roi"],
                band_name=feat["band"],
                atom_name=feat["atom"],
                win_start_ms=float(feat["win_start_ms"]),
                win_stop_ms=float(feat["win_stop_ms"]),
            )

            row = {
                "analysis_set": set_label,
                "target_id": feat["target_id"],
                "feature_key": feat["feature_key"],
                "tier": feat["tier"],
                "partition": feat["partition"],
                "contrast": feat["contrast"],
                "atom": feat["atom"],
                "band": feat["band"],
                "roi": feat["roi"],
                "win_start_ms": float(feat["win_start_ms"]),
                "win_stop_ms": float(feat["win_stop_ms"]),
                "dyad_id": dyad_id,
                "tensor_file": os.path.basename(tensor_path),
                **vals
            }
            rows.append(row)

            counter += 1
            if counter % 50 == 0 or counter == n_total:
                print(f"  [PROGRESS] {counter}/{n_total}")

    per_dyad = pd.DataFrame(rows)
    print(f"[PASS] {set_label} per-dyad extraction complete: {len(per_dyad)} rows")
    print()

    group_cols = [
        "analysis_set", "target_id", "feature_key", "tier", "partition", "contrast",
        "atom", "band", "roi", "win_start_ms", "win_stop_ms"
    ]

    summary_rows = []
    print(f"[INFO] Summarizing {set_label} feature-level statistics...")

    for keys, g in per_dyad.groupby(group_cols, dropna=False):
        g = g.sort_values("dyad_id").reset_index(drop=True)
        key_dict = dict(zip(group_cols, keys))
        atom = key_dict["atom"]

        row = {**key_dict}

        early_vs_late = paired_stats(g["early_mean"].values, g["late_mean"].values)
        row.update({
            "block_half_test": "early_vs_late",
            "block_half_n": early_vs_late["n"],
            "block_half_mean_early": early_vs_late["mean_a"],
            "block_half_mean_late": early_vs_late["mean_b"],
            "block_half_mean_diff": early_vs_late["mean_diff"],
            "block_half_sd_diff": early_vs_late["sd_diff"],
            "block_half_t": early_vs_late["t"],
            "block_half_p": early_vs_late["p"],
            "block_half_mean_abs_diff": early_vs_late["mean_abs_diff"],
        })

        if atom == "xtx":
            role_swap = paired_stats(g["P1_Leader"].values, g["P2_Leader"].values)
            row.update({
                "role_swap_test": "P1_Leader_vs_P2_Leader",
                "role_swap_n": role_swap["n"],
                "role_swap_mean_a": role_swap["mean_a"],
                "role_swap_mean_b": role_swap["mean_b"],
                "role_swap_mean_diff": role_swap["mean_diff"],
                "role_swap_sd_diff": role_swap["sd_diff"],
                "role_swap_t": role_swap["t"],
                "role_swap_p": role_swap["p"],
                "role_swap_mean_abs_diff": role_swap["mean_abs_diff"],
                "role_swap_interpretation_target": "Leader-stability-across-swap",
            })

        elif atom == "yty":
            role_swap = paired_stats(g["P2_Follower"].values, g["P1_Follower"].values)
            row.update({
                "role_swap_test": "P2_Follower_vs_P1_Follower",
                "role_swap_n": role_swap["n"],
                "role_swap_mean_a": role_swap["mean_a"],
                "role_swap_mean_b": role_swap["mean_b"],
                "role_swap_mean_diff": role_swap["mean_diff"],
                "role_swap_sd_diff": role_swap["sd_diff"],
                "role_swap_t": role_swap["t"],
                "role_swap_p": role_swap["p"],
                "role_swap_mean_abs_diff": role_swap["mean_abs_diff"],
                "role_swap_interpretation_target": "Follower-stability-across-swap",
            })

        else:
            row.update({
                "role_swap_test": "",
                "role_swap_n": np.nan,
                "role_swap_mean_a": np.nan,
                "role_swap_mean_b": np.nan,
                "role_swap_mean_diff": np.nan,
                "role_swap_sd_diff": np.nan,
                "role_swap_t": np.nan,
                "role_swap_p": np.nan,
                "role_swap_mean_abs_diff": np.nan,
                "role_swap_interpretation_target": "",
            })

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print(f"[PASS] {set_label} feature-level summary complete: {len(summary)} rows")
    print()

    return per_dyad, summary


def print_console_summary(summary: pd.DataFrame, label: str):
    print("=" * 100)
    print(f"CONSOLE SUMMARY: {label}")
    print("=" * 100)

    print(f"[INFO] Total summarized features: {len(summary)}")
    print(f"[INFO] Unique-role features (xtx/yty): {int(summary['atom'].isin(['xtx', 'yty']).sum())}")
    print(f"[INFO] Joint features (sts/rtr): {int(summary['atom'].isin(['sts', 'rtr']).sum())}")
    print()

    tmp = summary[summary["atom"].isin(["xtx", "yty"])].copy()
    if len(tmp) > 0:
        stable_n = int((tmp["role_swap_p"] >= ALPHA).sum())
        vulnerable_n = int((tmp["role_swap_p"] < ALPHA).sum())
        print(f"[INFO] Unique-role stable (p >= {ALPHA}): {stable_n}")
        print(f"[INFO] Unique-role vulnerable (p < {ALPHA}): {vulnerable_n}")
        print()

        print("TOP UNIQUE-ROLE FEATURES BY ROLE-SWAP STABILITY")
        top_cols = [
            "target_id", "tier", "partition", "contrast", "atom", "band", "roi",
            "win_start_ms", "win_stop_ms",
            "role_swap_test", "role_swap_mean_abs_diff",
            "role_swap_t", "role_swap_p",
            "block_half_mean_abs_diff", "block_half_t", "block_half_p"
        ]
        tmp = tmp.sort_values(
            by=["role_swap_p", "role_swap_mean_abs_diff", "block_half_mean_abs_diff"],
            ascending=[True, True, True],
            na_position="last"
        )
        print(tmp[top_cols].to_string(index=False))
        print()

    print("ALL FEATURES BY BLOCK-HALF STABILITY")
    all_cols = [
        "target_id", "tier", "partition", "contrast", "atom", "band", "roi",
        "win_start_ms", "win_stop_ms",
        "block_half_mean_abs_diff", "block_half_t", "block_half_p",
        "role_swap_mean_abs_diff", "role_swap_t", "role_swap_p"
    ]
    tmp_all = summary.sort_values(
        by=["block_half_p", "block_half_mean_abs_diff"],
        ascending=[True, True],
        na_position="last"
    )
    print(tmp_all[all_cols].to_string(index=False))
    print()

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("ROLE-SWAP VALIDATION FOR STRICT ΦID SURVIVORS")
    print("=" * 100)
    print(f"Robustness directory: {ROBUSTNESS_DIR}")
    print(f"Dyad tensor directory: {DYAD_TENSOR_DIR}")
    print(f"Tier A CSV: {TIER_A_CSV}")
    print(f"Tier B CSV: {TIER_B_CSV}")
    print(f"Homologous group labels: {GROUP_LABELS_HOMOL}")
    print(f"Axis group labels:       {GROUP_LABELS_AXIS}")
    print(f"Run only unique atoms: {RUN_ONLY_UNIQUE_ATOMS}")
    print(f"Restrict partitions: {RESTRICT_PARTITIONS}")
    print(f"Alpha threshold: {ALPHA}")
    print()

    required_paths = [
        TIER_A_CSV,
        TIER_B_CSV,
        GROUP_LABELS_HOMOL,
        GROUP_LABELS_AXIS,
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    features_A = load_and_filter_features(TIER_A_CSV, "Tier_A")
    features_B = load_and_filter_features(TIER_B_CSV, "Tier_B")

    combined_features = pd.concat([features_A, features_B], ignore_index=True)
    print(f"[INFO] Combined features: {len(combined_features)}")
    print()

    homol_map, axis_map = load_tensor_path_maps(HOMOL_GLOB, AXIS_GLOB)

    axis_group_labels = load_group_labels(GROUP_LABELS_AXIS)
    homol_group_labels = load_group_labels(GROUP_LABELS_HOMOL)

    print(
        f"[INFO] Axis fallback labels loaded: "
        f"{len(axis_group_labels['roi_labels'])} ROIs, "
        f"{len(axis_group_labels['band_labels'])} bands, "
        f"{len(axis_group_labels['atom_labels'])} atoms"
    )
    print(
        f"[INFO] Homol fallback labels loaded: "
        f"{len(homol_group_labels['roi_labels'])} ROIs, "
        f"{len(homol_group_labels['band_labels'])} bands, "
        f"{len(homol_group_labels['atom_labels'])} atoms"
    )
    print()

    common_dyads = sorted(set(homol_map.keys()) & set(axis_map.keys()))
    if len(common_dyads) == 0:
        raise RuntimeError("No dyads found in common between homologous and axis tensor maps.")

    print(f"[INFO] Homologous dyads: {sorted(homol_map.keys())}")
    print(f"[INFO] Axis dyads:       {sorted(axis_map.keys())}")
    print(f"[INFO] Common dyads:     {common_dyads}")
    print()

    per_dyad_A, summary_A = run_one_feature_set(
        features_A, "Tier_A", homol_map, axis_map, common_dyads,
        axis_group_labels, homol_group_labels
    )

    per_dyad_B, summary_B = run_one_feature_set(
        features_B, "Tier_B", homol_map, axis_map, common_dyads,
        axis_group_labels, homol_group_labels
    )

    per_dyad_ALL, summary_ALL = run_one_feature_set(
        combined_features, "Combined", homol_map, axis_map, common_dyads,
        axis_group_labels, homol_group_labels
    )

    print_console_summary(summary_A, "Tier A")
    print_console_summary(summary_B, "Tier B")
    print_console_summary(summary_ALL, "Combined Tier A + Tier B")

    per_dyad_A.to_csv(OUT_PER_DYAD_A, index=False)
    summary_A.to_csv(OUT_SUMMARY_A, index=False)

    per_dyad_B.to_csv(OUT_PER_DYAD_B, index=False)
    summary_B.to_csv(OUT_SUMMARY_B, index=False)

    per_dyad_ALL.to_csv(OUT_PER_DYAD_ALL, index=False)
    summary_ALL.to_csv(OUT_SUMMARY_ALL, index=False)

    print("=" * 100)
    print("FILES WRITTEN")
    print("=" * 100)
    print(f"[WRITE] {OUT_PER_DYAD_A}")
    print(f"[WRITE] {OUT_SUMMARY_A}")
    print(f"[WRITE] {OUT_PER_DYAD_B}")
    print(f"[WRITE] {OUT_SUMMARY_B}")
    print(f"[WRITE] {OUT_PER_DYAD_ALL}")
    print(f"[WRITE] {OUT_SUMMARY_ALL}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
