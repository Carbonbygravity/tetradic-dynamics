"""Report the normative phasic cluster results.

This script does not run a cluster test or modify cluster-result pickles. It
requires all 18 band-by-contrast files, checks their cell order and contrasts, and
writes new CSV reports outside the cluster-results directory.
"""

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ALPHA = 0.05

BANDS = ("theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "gamma")
CONTRASTS = ("Condition", "Role", "Interaction")
EXPECTED_CELL_ORDER = ["LMA", "LMR", "FMA", "FMR"]
EXPECTED_DEFINITIONS = {
    "Condition": "(LMA+FMA) - (LMR+FMR)",
    "Role": "(LMR+LMA) - (FMR+FMA)",
    "Interaction": "(LMR+FMA) - (LMA+FMR)",
}


def cluster_to_mask_paired(cluster, shape):
    """Decode either an MNE boolean mask or paired time/ROI indices."""
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.shape == shape:
        return cluster
    if isinstance(cluster, (tuple, list)) and len(cluster) == 2:
        time_indices = np.asarray(cluster[0], dtype=int)
        roi_indices = np.asarray(cluster[1], dtype=int)
        if time_indices.shape != roi_indices.shape:
            raise ValueError("Paired cluster index arrays have different shapes.")
        if np.any(time_indices < 0) or np.any(time_indices >= shape[0]):
            raise ValueError("Cluster time index is out of range.")
        if np.any(roi_indices < 0) or np.any(roi_indices >= shape[1]):
            raise ValueError("Cluster ROI index is out of range.")
        mask = np.zeros(shape, dtype=bool)
        mask[time_indices, roi_indices] = True
        return mask
    raise ValueError("Unrecognized MNE cluster representation.")


def get_mask(result, index):
    saved_masks = result.get("cluster_masks")
    if saved_masks is not None and saved_masks[index] is not None:
        mask = np.asarray(saved_masks[index])
        if mask.dtype != bool or mask.shape != result["t_obs"].shape:
            raise ValueError("Saved cluster mask has the wrong dtype or shape.")
        return mask
    return cluster_to_mask_paired(result["clusters"][index], result["t_obs"].shape)


def direction_for(contrast, signed_mass):
    if signed_mass == 0:
        return "Undetermined (zero signed mass)"
    positive = signed_mass > 0
    if contrast == "Condition":
        return "Matching > Mirroring" if positive else "Mirroring > Matching"
    if contrast == "Role":
        return "Leader > Follower" if positive else "Follower > Leader"
    if contrast == "Interaction":
        return (
            "(LMR+FMA) > (LMA+FMR)" if positive
            else "(LMA+FMR) > (LMR+FMA)"
        )
    raise ValueError(f"Unknown contrast: {contrast}")


def validate_result(result, band, contrast, filename):
    if result.get("band") != band or result.get("contrast") != contrast:
        raise ValueError(f"Band/contrast mismatch in {filename}")
    if result.get("cell_order") != EXPECTED_CELL_ORDER:
        raise ValueError(f"Unverified cell order in {filename}")
    if result.get("contrast_definition") != EXPECTED_DEFINITIONS:
        raise ValueError(f"Unexpected contrast definitions in {filename}")
    if result.get("cluster_mask_parse_errors", 0) != 0:
        raise ValueError(f"Cluster-mask parsing errors recorded in {filename}")

    t_obs = np.asarray(result["t_obs"])
    pvals = np.asarray(result["cluster_p_values"], dtype=float)
    time_ms = np.asarray(result["time_vector"], dtype=float)
    roi_labels = result["roi_labels"]
    n_subjects = int(result["n_subjects"])
    if t_obs.ndim != 2 or t_obs.shape != (len(time_ms), len(roi_labels)):
        raise ValueError(f"Time/ROI dimensions do not match in {filename}")
    if len(result["clusters"]) != len(pvals):
        raise ValueError(f"Cluster and p-value counts differ in {filename}")
    if result.get("cluster_masks") is not None and len(result["cluster_masks"]) != len(pvals):
        raise ValueError(f"Saved mask and p-value counts differ in {filename}")
    if n_subjects <= 1 or not np.all(np.isfinite(time_ms)):
        raise ValueError(f"Invalid subject count or time vector in {filename}")
    if not np.all(np.diff(time_ms) > 0):
        raise ValueError(f"Time vector is not strictly increasing in {filename}")
    if np.nanmax(np.abs(time_ms)) <= 10:
        raise ValueError(f"Time vector appears to be seconds, not milliseconds, in {filename}")
    return t_obs, pvals, time_ms, roi_labels, n_subjects


def summarize_cluster(result, band, contrast, index, t_obs, pval, time_ms, roi_labels, n_subjects):
    mask = get_mask(result, index)
    values = t_obs[mask]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Cluster {index} in {band}/{contrast} is empty or non-finite")

    time_indices = np.flatnonzero(mask.any(axis=1))
    roi_indices = np.flatnonzero(mask.any(axis=0))
    coverage = mask.sum(axis=0)
    top_indices = sorted(roi_indices, key=lambda i: (-int(coverage[i]), str(roi_labels[i])))[:10]
    mean_t = float(np.mean(values))
    peak_t = float(values[np.argmax(np.abs(values))])
    mass = float(np.sum(values))
    return {
        "band": band,
        "contrast": contrast,
        "cluster_index_zero_based": int(index),
        "cluster_p_fwer": float(pval),
        "direction": direction_for(contrast, mass),
        "n_subjects": n_subjects,
        "cluster_mass_signed": mass,
        "mean_t_signed": mean_t,
        "peak_t_signed": peak_t,
        "estimated_mean_dz_signed": mean_t / np.sqrt(n_subjects),
        "estimated_peak_abs_dz": abs(peak_t) / np.sqrt(n_subjects),
        "time_start_ms": float(time_ms[time_indices[0]]),
        "time_end_ms": float(time_ms[time_indices[-1]]),
        "n_time_samples": int(len(time_indices)),
        "n_time_segments": int(1 + np.sum(np.diff(time_indices) > 1)),
        "n_rois": int(len(roi_indices)),
        "n_time_roi_points": int(mask.sum()),
        "roi_names": "; ".join(str(roi_labels[i]) for i in roi_indices),
        "top_rois_by_time_sample_coverage": "; ".join(
            f"{roi_labels[i]} ({int(coverage[i])})" for i in top_indices
        ),
        "source_tensor_path": result.get("source_tensor_path", ""),
        "time_vector_status": result.get("time_vector_status", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Report saved response-period-onset-aligned phasic clusters.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir
    report_dir = args.report_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if report_dir.exists():
        raise FileExistsError(f"Report directory exists; no files will be overwritten: {report_dir}")

    expected_files = {
        f"source_cluster_results_{band}_{contrast}.pkl"
        for band in BANDS for contrast in CONTRASTS
    }
    actual_files = {p.name for p in results_dir.glob("source_cluster_results_*.pkl")}
    if actual_files != expected_files:
        raise RuntimeError(
            f"Expected exactly 18 cluster files. "
            f"Missing: {sorted(expected_files - actual_files)}; "
            f"Unexpected: {sorted(actual_files - expected_files)}"
        )

    rows = []
    file_rows = []
    subject_counts = set()
    for band in BANDS:
        for contrast in CONTRASTS:
            filename = f"source_cluster_results_{band}_{contrast}.pkl"
            with (results_dir / filename).open("rb") as f:
                result = pickle.load(f)
            t_obs, pvals, time_ms, roi_labels, n_subjects = validate_result(
                result, band, contrast, filename
            )
            subject_counts.add(n_subjects)
            significant = np.flatnonzero(pvals < ALPHA)
            file_rows.append({
                "file": filename,
                "band": band,
                "contrast": contrast,
                "n_subjects": n_subjects,
                "n_clusters_tested": int(len(pvals)),
                "n_significant_clusters": int(len(significant)),
                "minimum_cluster_p": float(pvals.min()) if len(pvals) else np.nan,
                "time_vector_status": result.get("time_vector_status", ""),
            })
            for index in significant:
                rows.append(summarize_cluster(
                    result, band, contrast, int(index), t_obs, pvals[index],
                    time_ms, roi_labels, n_subjects
                ))
            print(f"{band:>10} | {contrast:<11} | {len(significant)} significant cluster(s)")

    if len(subject_counts) != 1:
        raise ValueError(f"Subject counts differ across files: {sorted(subject_counts)}")

    report_dir.mkdir(parents=True, exist_ok=False)
    summary_path = report_dir / "cluster_file_summary.csv"
    clusters_path = report_dir / "significant_cluster_details.csv"
    notes_path = report_dir / "report_notes.txt"
    pd.DataFrame(file_rows).to_csv(summary_path, index=False)
    pd.DataFrame(rows).to_csv(clusters_path, index=False)
    notes_path.write_text(
        "Source: normative cluster files with verified cell order and contrasts.\n"
        "Verified cell order: LMA, LMR, FMA, FMR.\n"
        "Condition direction: Matching minus Mirroring.\n"
        "Role direction: Leader minus Follower (sampled response-period label).\n"
        "Interaction direction: [(LMR+FMA) minus (LMA+FMR)] / 2.\n"
        "Cluster p-values are within-band, within-contrast permutation results.\n"
        "Time start/end span the cluster's earliest and latest included samples; "
        "n_time_segments indicates whether that span is discontinuous.\n"
        "ROI lists describe cluster coverage, not separate ROI-level significance.\n"
        "Estimated dz = local or cluster-mean t / sqrt(n_subjects); these are "
        "descriptive and not independently selected effect-size estimates.\n",
        encoding="utf-8",
    )
    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {clusters_path}")
    print(f"[SAVED] {notes_path}")
    print(f"[SUMMARY] {len(file_rows)} files; {len(rows)} significant clusters")


if __name__ == "__main__":
    main()
