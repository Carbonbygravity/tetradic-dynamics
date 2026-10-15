"""
Script: 04_intra_brain_homotopic_te.py

Purpose
-------
Compute bidirectional intra-brain transfer entropy (TE) between homotopic
left- and right-hemisphere ROI time series for each participant, trial, and band.

Pipeline role
-------------
Primary statistical analysis script for intra-brain homotopic transfer
entropy.

Inputs
------
dyadic_homologous_sourcelevel_time_resolved.pkl containing, per dyad:
- p1_data / p2_data: trials × rois × bands × time
- p1_info / p2_info: trial metadata
- roi_labels
- band_labels

Outputs
-------
One CSV per participant per dyad:
    dyad_<id>_<p1/p2>_te_homotopic.csv

Each file contains:
- dyad
- participant
- trial
- band
- roi
- direction
- role
- condition
- te_fw
- te_bw
- net_te

Notes
-----
- TE is computed using the JIDT Kraskov TransferEntropyCalculatorKraskov.
- Time series are z-scored before TE estimation.
- Homotopic pairs are matched by ROI stem (e.g., roi-lh with roi-rh).
- Each row stores both directions for a left-right homotopic pair:
      te_fw = TE(left -> right)
      te_bw = TE(right -> left)
- Net TE is defined as te_fw - te_bw.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import jpype
import jpype.imports
from jpype.types import JArray, JDouble

# =============================================================================
# User settings
# =============================================================================

JIDT_JAR_PATH = Path("PATH_TO_JIDT_JAR") / "infodynamics.jar"
INPUT_PATH = Path("PATH_TO_DYADIC_SOURCE_DATA") / "dyadic_homologous_sourcelevel_time_resolved.pkl"
OUTPUT_DIR = Path("PATH_TO_TE_OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BANDS = ["theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "gamma"]
KRASKOV_K = 4
ZSCORE_STD_THRESHOLD = 1e-9

# =============================================================================
# JVM + JIDT setup
# =============================================================================

if not JIDT_JAR_PATH.exists():
    raise FileNotFoundError(
        f"JIDT jar not found: {JIDT_JAR_PATH}\n"
        "Edit JIDT_JAR_PATH to point to infodynamics.jar."
    )
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dyadic source file not found: {INPUT_PATH}\n"
        "Edit INPUT_PATH to point to dyadic_homologous_sourcelevel_time_resolved.pkl."
    )

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "-ea",
        f"-Djava.class.path={JIDT_JAR_PATH}"
    )

TE_CALCULATOR = jpype.JClass(
    "infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov"
)

print("[INFO] JVM and JIDT Kraskov TE calculator initialized.")

# =============================================================================
# Helper functions
# =============================================================================

def safe_zscore(arr):
    """
    Z-score a 1D array and flag near-zero variance traces as invalid.
    """
    arr = np.asarray(arr, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr)
    if std <= ZSCORE_STD_THRESHOLD:
        return None, float(mean), float(std), False
    return (arr - mean) / std, float(mean), float(std), True


def compute_te(x, y, k=4):
    """
    Compute transfer entropy from x -> y using the JIDT Kraskov estimator.
    """
    x_java = JArray(JDouble)(x)
    y_java = JArray(JDouble)(y)

    calc = TE_CALCULATOR()
    calc.setProperty("k", str(k))
    calc.initialise()
    calc.setObservations(x_java, y_java)

    return calc.computeAverageLocalOfObservations()


def run_te_intrabrain(data, info, participant_label, dyad_id, label_names, band_labels):
    """
    Compute bidirectional TE for all homotopic left-right ROI pairs.

    Parameters
    ----------
    data : ndarray
        Shape: trials × rois × bands × time
    info : list-like
        Trial metadata entries
    participant_label : str
        'p1' or 'p2'
    dyad_id : int
        Dyad identifier
    label_names : list of str
        ROI labels
    band_labels : list of str
        Frequency band labels

    Returns
    -------
    list of dict
        Row-wise TE results
    """
    n_trials, _, _, _ = data.shape

    left_rois = {
        roi.replace("-lh", ""): i
        for i, roi in enumerate(label_names)
        if roi.endswith("-lh")
    }
    right_rois = {
        roi.replace("-rh", ""): i
        for i, roi in enumerate(label_names)
        if roi.endswith("-rh")
    }

    homotopic_pairs = [
        (roi_name, left_rois[roi_name], right_rois[roi_name])
        for roi_name in left_rois
        if roi_name in right_rois
    ]

    results = []
    excluded_rows = []
    stats = {
        "pairs_checked": 0,
        "pairs_excluded_near_zero_var": 0,
        "left_excluded": 0,
        "right_excluded": 0,
        "te_errors": 0,
    }

    for b_idx, band in enumerate(band_labels):
        if band not in TARGET_BANDS:
            continue

        for trial_idx in range(n_trials):
            meta = info[trial_idx]
            trial_number = meta.get("trial", trial_idx)

            for roi_name, lh_idx, rh_idx in homotopic_pairs:
                stats["pairs_checked"] += 1
                try:
                    ts_l, mean_l, std_l, ok_l = safe_zscore(data[trial_idx, lh_idx, b_idx, :])
                    ts_r, mean_r, std_r, ok_r = safe_zscore(data[trial_idx, rh_idx, b_idx, :])

                    if not ok_l or not ok_r:
                        stats["pairs_excluded_near_zero_var"] += 1
                        stats["left_excluded"] += int(not ok_l)
                        stats["right_excluded"] += int(not ok_r)
                        excluded_rows.append({
                            "dyad": dyad_id,
                            "participant": participant_label,
                            "trial": trial_number,
                            "band": band,
                            "roi": roi_name,
                            "role": meta["role"],
                            "condition": meta["condition"],
                            "left_mean": mean_l,
                            "left_std": std_l,
                            "left_invalid": not ok_l,
                            "right_mean": mean_r,
                            "right_std": std_r,
                            "right_invalid": not ok_r,
                            "reason": "near_zero_variance",
                        })
                        continue

                    te_fw = compute_te(ts_l, ts_r, k=KRASKOV_K)  # left -> right
                    te_bw = compute_te(ts_r, ts_l, k=KRASKOV_K)  # right -> left

                    results.append({
                        "dyad": dyad_id,
                        "participant": participant_label,
                        "trial": trial_number,
                        "band": band,
                        "roi": roi_name,
                        "direction": "left↔right",
                        "role": meta["role"],
                        "condition": meta["condition"],
                        "te_fw": te_fw,
                        "te_bw": te_bw,
                        "net_te": te_fw - te_bw,
                    })

                except Exception as e:
                    stats["te_errors"] += 1
                    tqdm.write(
                        f"[WARN] TE error (dyad={dyad_id}, participant={participant_label}, roi={roi_name}, band={band}, trial={trial_number}): {e}"
                    )

    return results, excluded_rows, stats

# =============================================================================
# Load input and run
# =============================================================================

with open(INPUT_PATH, "rb") as f:
    dyads = pickle.load(f)

print(f"[INFO] Loaded {len(dyads)} dyads for TE computation.")

for batch in tqdm(dyads, desc="Dyads"):
    dyad_id = batch["dyad"]
    label_names = batch["roi_labels"]
    band_labels = batch["band_labels"]

    for participant_key in ["p1", "p2"]:
        data = batch[f"{participant_key}_data"]
        info = batch[f"{participant_key}_info"]
        participant_label = participant_key

        out_csv = OUTPUT_DIR / f"dyad_{dyad_id:02d}_{participant_label}_te_homotopic.csv"
        excluded_csv = OUTPUT_DIR / f"dyad_{dyad_id:02d}_{participant_label}_te_homotopic_excluded.csv"

        if out_csv.exists():
            tqdm.write(f"[INFO] Skipping dyad {dyad_id} {participant_label}: file exists.")
            continue

        result, excluded_rows, stats = run_te_intrabrain(
            data=data,
            info=info,
            participant_label=participant_label,
            dyad_id=dyad_id,
            label_names=label_names,
            band_labels=band_labels,
        )

        if result:
            df = pd.DataFrame(result)
            df.to_csv(out_csv, index=False)
            tqdm.write(f"[SAVED] TE for dyad {dyad_id} {participant_label}: {out_csv.name}")
        else:
            tqdm.write(f"[WARN] No TE results for dyad {dyad_id} {participant_label}")

        if excluded_rows:
            pd.DataFrame(excluded_rows).to_csv(excluded_csv, index=False)
            tqdm.write(
                f"[INFO] Excluded {stats['pairs_excluded_near_zero_var']} near-zero-variance pairs "
                f"for dyad {dyad_id} {participant_label}: {excluded_csv.name}"
            )

        tqdm.write(
            f"[INFO] Dyad {dyad_id} {participant_label}: checked={stats['pairs_checked']}, "
            f"excluded={stats['pairs_excluded_near_zero_var']}, "
            f"left_invalid={stats['left_excluded']}, right_invalid={stats['right_excluded']}, "
            f"te_errors={stats['te_errors']}"
        )

print("\n--- Intra-brain homotopic TE computation complete. ---")
