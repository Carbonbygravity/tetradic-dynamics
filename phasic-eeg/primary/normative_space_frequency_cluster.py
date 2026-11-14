"""Normative source-level Space x Frequency cluster analysis.

The source tensor stacks cells as LMA, LMR, FMA, FMR. The analysis applies
log1p power, time averaging, ROI-centroid adjacency, and 5,000 permutations.

Run as a Python script or in a Jupyter cell using ``%run``. Outputs go into a
new folder; this script refuses to overwrite an existing results folder.
"""

import argparse
import os
import pickle

import numpy as np
from mne import create_info, read_labels_from_annot
from mne.channels import find_ch_adjacency, make_dig_montage
from mne.stats import permutation_cluster_1samp_test
from scipy import sparse


# Analysis settings.
SUBJECT = "fsaverage"
PARC = "aparc"
N_PERMUTATIONS = 5000
CONTRASTS = ("Condition", "Role", "Interaction")

# Empirically checked against participant-level parcellation averages. This
# order is defined by the normative tensor builder. The tensor may lack
# explicit cell-order metadata.
CELL_ORDER = ("LMA", "LMR", "FMA", "FMR")
IDX = {name: index for index, name in enumerate(CELL_ORDER)}


def collapse_cells(tensor, names):
    return np.mean(tensor[:, [IDX[name] for name in names], ...], axis=1)


def make_contrast_time_averaged(tensor, contrast):
    """Return time-averaged subjects x ROIs x bands."""
    if contrast == "Condition":
        a = collapse_cells(tensor, ("LMA", "FMA"))
        b = collapse_cells(tensor, ("LMR", "FMR"))
    elif contrast == "Role":
        a = collapse_cells(tensor, ("LMA", "LMR"))
        b = collapse_cells(tensor, ("FMA", "FMR"))
    elif contrast == "Interaction":
        # Interaction: Follower Condition difference minus Leader difference.
        a = collapse_cells(tensor, ("LMR", "FMA"))
        b = collapse_cells(tensor, ("LMA", "FMR"))
    else:
        raise ValueError(f"Unknown contrast: {contrast}")
    return a.mean(axis=-1) - b.mean(axis=-1)


def main():
    parser = argparse.ArgumentParser(description="Run normative space-frequency cluster tests.")
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects-dir", required=True)
    args = parser.parse_args()
    TENSOR_PATH = args.tensor
    OUTPUT_DIR = args.output_dir
    SUBJECTS_DIR = args.subjects_dir
    if not os.path.isfile(TENSOR_PATH):
        raise FileNotFoundError(TENSOR_PATH)
    if os.path.exists(OUTPUT_DIR):
        raise FileExistsError(
            f"Refusing to overwrite existing results folder: {OUTPUT_DIR}\n"
            "Choose a new OUTPUT_DIR for a further run."
        )

    with open(TENSOR_PATH, "rb") as stream:
        source = pickle.load(stream)

    raw = np.asarray(source["tensor"])
    subjects = source["subjects"]
    roi_labels = list(source["roi_labels"])
    band_labels = list(source["band_labels"])
    if raw.ndim != 5 or raw.shape[1] != 4:
        raise ValueError(f"Expected subjects x 4 cells x ROIs x bands x time, got {raw.shape}")
    if len(subjects) != raw.shape[0] or len(roi_labels) != raw.shape[2] or len(band_labels) != raw.shape[3]:
        raise ValueError("Tensor dimensions and subject/ROI/band labels disagree")
    if not np.isfinite(raw).all() or (raw <= -1).any():
        raise ValueError("Input contains non-finite values or values invalid for log1p")

    declared_order = source.get("cell_order")
    if declared_order is not None and tuple(declared_order) != CELL_ORDER:
        raise ValueError(
            f"Input declares cell order {declared_order}, expected {CELL_ORDER}. "
            "Do not use this script on a different tensor."
        )
    if declared_order is None:
        print("[INFO] Input has no cell_order metadata; using the independently checked normative builder order:", CELL_ORDER)

    # Apply log1p before contrast formation.
    tensor = np.log1p(raw)
    n_subjects, _, n_rois, n_bands, _ = tensor.shape

    # Use ROI-centroid/Delaunay adjacency and identity
    # band adjacency. This does not connect adjacent frequency bands.
    labels = read_labels_from_annot(subject=SUBJECT, parc=PARC, subjects_dir=SUBJECTS_DIR)
    centroids = {label.name: label.pos.mean(axis=0).tolist() for label in labels if label.name in roi_labels}
    missing = [roi for roi in roi_labels if roi not in centroids]
    if missing:
        raise ValueError(f"Missing ROI centroids: {missing}")
    info = create_info(ch_names=roi_labels, sfreq=1000, ch_types="eeg")
    info.set_montage(make_dig_montage(ch_pos={roi: centroids[roi] for roi in roi_labels}, coord_frame="head"))
    adjacency_rois, _ = find_ch_adjacency(info, ch_type="eeg")
    adjacency_full = sparse.kron(adjacency_rois, sparse.eye(n_bands, format="csr"), format="csr")
    if adjacency_full.shape != (n_rois * n_bands, n_rois * n_bands):
        raise ValueError(f"Unexpected adjacency shape: {adjacency_full.shape}")

    os.makedirs(OUTPUT_DIR)
    log_tensor_path = os.path.join(OUTPUT_DIR, "log_transformed_tensor.pkl")
    with open(log_tensor_path, "wb") as stream:
        pickle.dump(
            {
                "tensor": tensor,
                "subjects": subjects,
                "roi_labels": roi_labels,
                "band_labels": band_labels,
                "cell_order": list(CELL_ORDER),
                "source_tensor_path": TENSOR_PATH,
            },
            stream,
        )
    print(f"[SAVED] {log_tensor_path}")

    definitions = {
        "Condition": "(LMA+FMA) - (LMR+FMR)",
        "Role": "(LMA+LMR) - (FMA+FMR)",
        "Interaction": "(LMR+FMA) - (LMA+FMR)",
    }
    for contrast in CONTRASTS:
        difference = make_contrast_time_averaged(tensor, contrast)
        participant_means = np.nanmean(difference, axis=(1, 2))
        mean_difference = float(np.nanmean(participant_means))
        sd_difference = float(np.nanstd(participant_means, ddof=1))
        dz = mean_difference / sd_difference if sd_difference != 0 else np.nan
        print(f"[QA] {contrast}: mean(A-B)={mean_difference:+.6e}; d~{dz:+.3f}")

        x = difference.reshape(n_subjects, -1)
        t_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
            x,
            n_permutations=N_PERMUTATIONS,
            threshold=None,
            tail=0,
            adjacency=adjacency_full,
            out_type="mask",
            n_jobs=-1,
        )
        out_path = os.path.join(OUTPUT_DIR, f"space_frequency_cluster_{contrast}.pkl")
        with open(out_path, "wb") as stream:
            pickle.dump(
                {
                    "contrast_name": contrast,
                    "t_obs": t_obs,
                    "clusters": clusters,
                    "cluster_p_values": cluster_p_values,
                    "H0": h0,
                    "n_subjects": n_subjects,
                    "roi_labels": roi_labels,
                    "band_labels": band_labels,
                    "adjacency": adjacency_full,
                    "cell_order": list(CELL_ORDER),
                    "contrast_definition": definitions,
                    "source_tensor_path": TENSOR_PATH,
                },
                stream,
            )
        print(f"[SAVED] {out_path}; significant clusters (p < .05): {int(np.sum(cluster_p_values < 0.05))}")

    print(f"[DONE] Normative Space x Frequency outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
