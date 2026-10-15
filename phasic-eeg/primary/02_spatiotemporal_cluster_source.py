"""Normative source-level spatiotemporal cluster tests with verified cell order.

Input tensor order (verified against participant-level trial averages):
    0 = Leader Matching (LMA)
    1 = Leader Mirroring (LMR)
    2 = Follower Matching (FMA)
    3 = Follower Mirroring (FMR)

The analysis applies log1p power, ROI-centroid adjacency, and 5,000
permutations per band-by-contrast map. Outputs go to a new directory and
never overwrite an existing result set.
"""

import argparse
import os
import pickle

import numpy as np
from mne import create_info, read_labels_from_annot
from mne.channels import find_ch_adjacency, make_dig_montage
from mne.stats import spatio_temporal_cluster_1samp_test


# --- 1. Paths / I/O ---
parser = argparse.ArgumentParser(description="Run normative response-locked cluster tests.")
parser.add_argument("--tensor", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--subjects-dir", required=True)
args = parser.parse_args()
tensor_path = args.tensor
output_dir = args.output_dir
log_tensor_path = os.path.join(output_dir, "log_transformed_tensor.pkl")

subjects_dir = args.subjects_dir
subject = "fsaverage"
parc = "aparc"

N_PERMUTATIONS = 5000
ALPHA = 0.05
CONTRASTS_TO_TEST = ["Condition", "Role", "Interaction"]


# --- 2. Helper functions (time + cluster format) ---
def ensure_time_ms(time_vector, n_time):
    """Return a time vector in milliseconds."""
    if time_vector is None:
        return np.linspace(-1000.0, 1000.0, n_time), "generated_ms"

    tv = np.asarray(time_vector).astype(float)
    if tv.shape[0] != n_time:
        return np.linspace(-1000.0, 1000.0, n_time), "generated_ms_bad_len"

    max_abs = float(np.nanmax(np.abs(tv)))
    if max_abs <= 10.0:
        return tv * 1000.0, "converted_s_to_ms"
    return tv, "kept_ms"


def cluster_to_mask_paired(cluster, t_obs_shape):
    """Convert an MNE cluster to a time x ROI mask using paired indices."""
    n_time, n_roi = t_obs_shape
    if (
        isinstance(cluster, np.ndarray)
        and cluster.dtype == bool
        and cluster.shape == t_obs_shape
    ):
        return cluster

    if isinstance(cluster, (tuple, list)) and len(cluster) == 2:
        time_inds = np.asarray(cluster[0], dtype=int)
        roi_inds = np.asarray(cluster[1], dtype=int)
        if time_inds.size != roi_inds.size:
            raise ValueError("Cluster index arrays differ in length.")

        mask = np.zeros(t_obs_shape, dtype=bool)
        if time_inds.size > 0:
            valid = (
                (time_inds >= 0)
                & (time_inds < n_time)
                & (roi_inds >= 0)
                & (roi_inds < n_roi)
            )
            mask[time_inds[valid], roi_inds[valid]] = True
        return mask

    raise ValueError("Unrecognized cluster format.")


# --- 3. Protect previous results and load the source tensor ---
if not os.path.isfile(tensor_path):
    raise FileNotFoundError(tensor_path)
if os.path.exists(output_dir):
    raise FileExistsError(
        f"Output directory already exists; choose a new run name: {output_dir}"
    )

with open(tensor_path, "rb") as f:
    data = pickle.load(f)

tensor_raw = data["tensor"]  # subjects x 4 x rois x bands x time
subjects = data.get("subjects")
roi_labels = data.get("roi_labels")
band_labels = data.get("band_labels")
time_vector = data.get("time_vector")

if tensor_raw.ndim != 5 or tensor_raw.shape[1] != 4:
    raise ValueError(f"Expected subjects x 4 x rois x bands x time; got {tensor_raw.shape}")
if len(roi_labels) != tensor_raw.shape[2]:
    raise ValueError("ROI-label count does not match tensor shape.")
if len(band_labels) != tensor_raw.shape[3]:
    raise ValueError("Band-label count does not match tensor shape.")

n_time = tensor_raw.shape[-1]
time_vector_ms, time_vector_status = ensure_time_ms(time_vector, n_time)
tensor = np.log1p(tensor_raw)

# Create the output only after the input has passed basic checks.
os.makedirs(output_dir, exist_ok=False)
with open(log_tensor_path, "wb") as f:
    pickle.dump(
        {
            "tensor": tensor,
            "subjects": subjects,
            "roi_labels": roi_labels,
            "band_labels": band_labels,
            "time_vector": time_vector_ms,
            "time_vector_status": time_vector_status,
            "cell_order": ["LMA", "LMR", "FMA", "FMR"],
            "source_tensor_path": tensor_path,
        },
        f,
    )

print(f"[SAVED] {log_tensor_path}")
print(f"[INFO] Tensor shape: {tensor.shape}")
print(
    f"[INFO] Time vector: {time_vector_status}; "
    f"{time_vector_ms[0]:.1f} to {time_vector_ms[-1]:.1f} ms"
)


# --- 4. Build ROI-centroid adjacency ---
labels = read_labels_from_annot(subject=subject, parc=parc, subjects_dir=subjects_dir)
roi_centroids = {}
for label in labels:
    if label.name in roi_labels:
        roi_centroids[label.name] = label.pos.mean(axis=0).tolist()

missing = [roi for roi in roi_labels if roi not in roi_centroids]
if missing:
    raise ValueError(f"Missing ROI centroids: {missing}")

roi_positions = {roi: roi_centroids[roi] for roi in roi_labels}
info = create_info(ch_names=list(roi_positions.keys()), sfreq=1000, ch_types="eeg")
montage = make_dig_montage(ch_pos=roi_positions, coord_frame="head")
info.set_montage(montage)
adjacency, _ = find_ch_adjacency(info, ch_type="eeg")
print(f"[INFO] Adjacency matrix constructed for {len(roi_labels)} ROIs.")


# --- 5. Contrasts using the verified saved-tensor order ---
IDX = {"LMA": 0, "LMR": 1, "FMA": 2, "FMR": 3}


def collapse_cells(X, indices):
    return np.mean(X[:, indices, ...], axis=1)


def make_contrast(X, band_idx, kind):
    """Return subjects x time x ROIs for the selected contrast and band."""
    if kind == "Condition":
        # Matching - Mirroring
        A = collapse_cells(X, [IDX["LMA"], IDX["FMA"]])
        B = collapse_cells(X, [IDX["LMR"], IDX["FMR"]])
    elif kind == "Role":
        # Leader - Follower (response-event role)
        A = collapse_cells(X, [IDX["LMR"], IDX["LMA"]])
        B = collapse_cells(X, [IDX["FMR"], IDX["FMA"]])
    elif kind == "Interaction":
        # (Leader Mirroring + Follower Matching)
        # - (Leader Matching + Follower Mirroring)
        A = collapse_cells(X, [IDX["LMR"], IDX["FMA"]])
        B = collapse_cells(X, [IDX["LMA"], IDX["FMR"]])
    else:
        raise ValueError(f"Unknown contrast: {kind}")

    return (A[:, :, band_idx, :] - B[:, :, band_idx, :]).transpose(0, 2, 1)


# --- 6. Run cluster tests and save only to the new directory ---
for band_idx, band in enumerate(band_labels):
    for contrast in CONTRASTS_TO_TEST:
        print(f"\n[RUNNING] Band: {band} | Contrast: {contrast}")
        X = make_contrast(tensor, band_idx, contrast)
        print(f"[INFO] Input shape: {X.shape}")

        t_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
            X,
            adjacency=adjacency,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            threshold=None,
            n_jobs=-1,
        )

        cluster_masks = []
        mask_errors = 0
        for cluster in clusters:
            try:
                cluster_masks.append(cluster_to_mask_paired(cluster, t_obs.shape))
            except Exception:
                mask_errors += 1
                cluster_masks.append(None)

        out_path = os.path.join(
            output_dir, f"source_cluster_results_{band}_{contrast}.pkl"
        )
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "contrast": contrast,
                    "band": band,
                    "t_obs": t_obs,
                    "clusters": clusters,
                    "cluster_masks": cluster_masks,
                    "cluster_p_values": cluster_p_values,
                    "H0": H0,
                    "n_subjects": X.shape[0],
                    "adjacency": adjacency,
                    "roi_labels": roi_labels,
                    "time_vector": time_vector_ms,
                    "time_vector_status": time_vector_status,
                    "cluster_mask_parse_errors": mask_errors,
                    "cell_order": ["LMA", "LMR", "FMA", "FMR"],
                    "contrast_definition": {
                        "Condition": "(LMA+FMA) - (LMR+FMR)",
                        "Role": "(LMR+LMA) - (FMR+FMA)",
                        "Interaction": "(LMR+FMA) - (LMA+FMR)",
                    },
                    "source_tensor_path": tensor_path,
                },
                f,
            )

        n_sig = int(np.sum(cluster_p_values < ALPHA))
        print(
            f"[SAVED] {out_path} | Significant clusters (p < {ALPHA}): "
            f"{n_sig} | mask_parse_errors={mask_errors}"
        )

print("\n--- Source-level cluster analysis complete. ---")
