"""Display the normative Space x Frequency cluster results.

Run directly with Python or in Jupyter using ``%run``. This script does not
rerun permutations. It validates cell order and rejects incomplete result
sets; it can also write a report CSV to a new output directory.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd


ALPHA = 0.05
WRITE_CSV = True
CELL_ORDER = ("LMA", "LMR", "FMA", "FMR")
CONTRASTS = ("Condition", "Role", "Interaction")
EXPECTED_DEFINITIONS = {
    "Condition": "(LMA+FMA) - (LMR+FMR)",
    "Role": "(LMA+LMR) - (FMA+FMR)",
    "Interaction": "(LMR+FMA) - (LMA+FMR)",
}


def cluster_to_mask_1d(cluster, length):
    """Accept MNE's 1-D boolean mask or index-list cluster representation."""
    if isinstance(cluster, (tuple, list)) and len(cluster) == 1:
        cluster = cluster[0]
    arr = np.asarray(cluster)
    if arr.dtype == bool and arr.size == length:
        return arr.reshape(length)
    if np.issubdtype(arr.dtype, np.integer):
        indices = arr.reshape(-1)
        if (indices < 0).any() or (indices >= length).any():
            raise ValueError("Cluster contains out-of-range feature indices")
        mask = np.zeros(length, dtype=bool)
        mask[indices] = True
        return mask
    raise ValueError(f"Unrecognized cluster shape/dtype: {arr.shape}/{arr.dtype}")


def direction_for(contrast, signed_mass):
    if contrast == "Condition":
        positive = "Matching > Mirroring"
        negative = "Mirroring > Matching"
    elif contrast == "Role":
        positive = "Leader > Follower"
        negative = "Follower > Leader"
    elif contrast == "Interaction":
        # Positive (LMR+FMA)-(LMA+FMR) means the Follower Matching-minus-
        # Mirroring effect exceeds the Leader Matching-minus-Mirroring effect.
        positive = "Follower (Matching-Mirroring) > Leader (Matching-Mirroring)"
        negative = "Leader (Matching-Mirroring) > Follower (Matching-Mirroring)"
    else:
        raise ValueError(f"Unexpected contrast: {contrast}")
    return positive if signed_mass > 0 else negative if signed_mass < 0 else "Zero signed mass"


def verify_provenance(result, contrast, path):
    if tuple(result.get("cell_order", ())) != CELL_ORDER:
        raise ValueError(f"Wrong or missing normative cell-order metadata in {path}")
    if result.get("contrast_name") != contrast:
        raise ValueError(f"Contrast name and file disagree in {path}")
    definitions = result.get("contrast_definition", {})
    if definitions.get(contrast) != EXPECTED_DEFINITIONS[contrast]:
        raise ValueError(f"Wrong or missing contrast definition in {path}")
    if int(result.get("n_subjects", 0)) < 2:
        raise ValueError(f"Invalid subject count in {path}")


def main():
    parser = argparse.ArgumentParser(description="Verify saved normative space-frequency clusters.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir
    CSV_PATH = os.path.join(args.output_dir, "space_frequency_cluster_report.csv")
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(
            f"Results folder not found: {RESULTS_DIR}\n"
            "Run normative_space_frequency_cluster.py first."
        )

    # Require all three contrasts before presenting a report.
    files = {
        contrast: os.path.join(RESULTS_DIR, f"space_frequency_cluster_{contrast}.pkl")
        for contrast in CONTRASTS
    }
    missing = [path for path in files.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Results are incomplete; missing: {missing}")

    all_rows = []
    for contrast in CONTRASTS:
        path = files[contrast]
        with open(path, "rb") as stream:
            result = pickle.load(stream)
        verify_provenance(result, contrast, path)

        t_obs = np.asarray(result["t_obs"], dtype=float).reshape(-1)
        clusters = result["clusters"]
        pvals = np.asarray(result["cluster_p_values"], dtype=float)
        roi_labels = list(result["roi_labels"])
        band_labels = list(result["band_labels"])
        n_subjects = int(result["n_subjects"])
        n_rois, n_bands = len(roi_labels), len(band_labels)
        if len(t_obs) != n_rois * n_bands:
            raise ValueError(f"ROI x band dimensions do not match t_obs in {path}")
        if len(clusters) != len(pvals):
            raise ValueError(f"Cluster count and p-value count disagree in {path}")

        significant = np.flatnonzero(pvals < ALPHA)
        print("\n" + "=" * 80)
        print(f"Normative Space x Frequency | {contrast} | {os.path.basename(path)}")
        print(f"N={n_subjects}; {n_rois} ROIs x {n_bands} bands; significant clusters={len(significant)}")
        print("=" * 80)
        if not len(significant):
            print(f"No clusters with cluster-level p < {ALPHA}.")
            continue

        for display_index, cluster_index in enumerate(significant, start=1):
            mask = cluster_to_mask_1d(clusters[cluster_index], len(t_obs))
            indices = np.flatnonzero(mask)
            if not len(indices):
                raise ValueError(f"Significant cluster {cluster_index} has an empty mask in {path}")
            values = t_obs[indices]
            mass = float(np.sum(values))
            mean_t = float(np.mean(values))
            peak_index = int(indices[np.argmax(np.abs(values))])
            peak_t = float(t_obs[peak_index])
            roi_indices = indices // n_bands
            band_indices = indices % n_bands
            peak_roi = roi_labels[peak_index // n_bands]
            peak_band = band_labels[peak_index % n_bands]
            rois = list(dict.fromkeys(roi_labels[index] for index in roi_indices))
            bands = list(dict.fromkeys(band_labels[index] for index in band_indices))
            roi_counts = np.bincount(roi_indices, minlength=n_rois)
            band_counts = np.bincount(band_indices, minlength=n_bands)
            top_roi_indices = np.argsort(-roi_counts, kind="stable")[:10]
            top_band_indices = np.argsort(-band_counts, kind="stable")
            top_rois = ", ".join(
                f"{roi_labels[index]} (n={roi_counts[index]})"
                for index in top_roi_indices if roi_counts[index] > 0
            )
            top_bands = ", ".join(
                f"{band_labels[index]} (n={band_counts[index]})"
                for index in top_band_indices if band_counts[index] > 0
            )
            direction = direction_for(contrast, mass)
            mean_dz = mean_t / np.sqrt(n_subjects)
            peak_abs_dz = abs(peak_t) / np.sqrt(n_subjects)

            print(f"\n--- Significant cluster {display_index} (original index {cluster_index}) ---")
            print(f"  Cluster p (FWER)   : {pvals[cluster_index]:.4f}")
            print(f"  Direction          : {direction}")
            print(f"  Cluster mass (sum t): {mass:.2f}")
            print(f"  Mean t             : {mean_t:.2f} | descriptive mean pointwise dz~{mean_dz:.2f}")
            print(f"  Peak |t|           : {abs(peak_t):.2f} | peak local |dz|~{peak_abs_dz:.2f}")
            print(f"  Peak ROI / band    : {peak_roi} / {peak_band}")
            print(f"  ROI x band elements: {len(indices)} of {n_rois * n_bands}")
            print(f"  Bands involved     : {', '.join(bands)}")
            print(f"  ROIs involved      : {', '.join(rois)}")
            print(f"  Top ROIs           : {top_rois}")
            print(f"  Top bands          : {top_bands}")

            all_rows.append({
                "file": os.path.basename(path),
                "contrast": contrast,
                "cluster_index": int(cluster_index),
                "cluster_p_fwer": float(pvals[cluster_index]),
                "direction": direction,
                "cluster_mass_sum_t": mass,
                "mean_t": mean_t,
                "peak_abs_t": abs(peak_t),
                "descriptive_mean_pointwise_dz": mean_dz,
                "peak_local_abs_dz": peak_abs_dz,
                "n_elements": int(len(indices)),
                "peak_roi": peak_roi,
                "peak_band": peak_band,
                "rois_involved": "; ".join(rois),
                "bands_involved": "; ".join(bands),
                "top_rois": top_rois,
                "top_bands": top_bands,
                "n_subjects": n_subjects,
            })

    if WRITE_CSV:
        os.makedirs(args.output_dir)
        frame = pd.DataFrame(all_rows)
        frame.to_csv(CSV_PATH, index=False)
        print(f"\n[SAVED] CSV report: {CSV_PATH}")
    print("\n[NOTE] Cluster p-values apply to the cluster, not to each listed ROI or band.")
    print("[DONE] Normative Space x Frequency verification complete.")


if __name__ == "__main__":
    main()
