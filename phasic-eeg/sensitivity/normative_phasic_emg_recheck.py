"""Recheck normative phasic clusters after two explicit EMG-ROI exclusions.

This creates a new output directory; it never changes the primary tensor,
saved cluster files, or manuscript. The input is the verified 30-person
log-power tensor with cell order LMA, LMR, FMA, FMR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Keep generated JIT cache files outside the analysis code directory.
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "tetradic-phasic-numba-cache")
)

import mne
import numpy as np
from mne import create_info, read_labels_from_annot
from mne.channels import find_ch_adjacency, make_dig_montage
from mne.stats import spatio_temporal_cluster_1samp_test


CELL_ORDER = ["LMA", "LMR", "FMA", "FMR"]
CELL = {name: index for index, name in enumerate(CELL_ORDER)}
BANDS = ["high_beta", "gamma"]
CONTRASTS = ["Condition", "Role", "Interaction"]
MASKS = {
    "manuscript": [
        "temporalpole", "inferiortemporal", "bankssts", "transversetemporal"
    ],
    "alternative": [
        "temporalpole", "fusiform", "inferiortemporal", "middletemporal"
    ],
}
FIELDNAMES = [
    "mask", "band", "contrast", "cluster_index_zero_based", "p_fwer",
    "direction", "cluster_mass_signed", "mean_t_signed", "peak_abs_t",
    "estimated_mean_dz_signed", "estimated_peak_abs_dz", "start_ms",
    "end_ms", "n_rois", "n_time_roi_points", "roi_names",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_mask(cluster, shape):
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.shape == shape:
        return cluster
    if isinstance(cluster, (tuple, list)) and len(cluster) == 2:
        times = np.asarray(cluster[0], dtype=int)
        rois = np.asarray(cluster[1], dtype=int)
        if times.size != rois.size:
            raise ValueError("Cluster time/ROI index arrays differ in length")
        if np.any(times < 0) or np.any(times >= shape[0]):
            raise ValueError("Cluster time index out of range")
        if np.any(rois < 0) or np.any(rois >= shape[1]):
            raise ValueError("Cluster ROI index out of range")
        result = np.zeros(shape, dtype=bool)
        result[times, rois] = True
        return result
    raise TypeError(f"Unrecognized cluster representation: {type(cluster)}")


def make_contrast(tensor, band_index: int, kind: str):
    if kind == "Condition":
        positive, negative = ("LMA", "FMA"), ("LMR", "FMR")
    elif kind == "Role":
        positive, negative = ("LMA", "LMR"), ("FMA", "FMR")
    elif kind == "Interaction":
        positive, negative = ("LMR", "FMA"), ("LMA", "FMR")
    else:
        raise ValueError(kind)
    band_tensor = tensor[:, :, :, band_index, :]
    a = np.take(band_tensor, [CELL[x] for x in positive], axis=1).mean(axis=1)
    b = np.take(band_tensor, [CELL[x] for x in negative], axis=1).mean(axis=1)
    return (a - b).transpose(0, 2, 1)


def direction(kind: str, mass: float) -> str:
    if kind == "Condition":
        return "Matching > Mirroring" if mass > 0 else "Mirroring > Matching"
    if kind == "Role":
        return "Leader > Follower" if mass > 0 else "Follower > Leader"
    return (
        "Follower (Match-Mirror) > Leader (Match-Mirror)"
        if mass > 0 else
        "Leader (Match-Mirror) > Follower (Match-Mirror)"
    )


def run(args):
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError(f"Cannot resume missing output: {args.output}")
    if not TENSOR_PATH.is_file():
        raise FileNotFoundError(TENSOR_PATH)
    with TENSOR_PATH.open("rb") as handle:
        source = pickle.load(handle)
    tensor = np.asarray(source["tensor"])
    roi_labels = list(source["roi_labels"])
    band_labels = list(source["band_labels"])
    times_ms = np.asarray(source["time_vector"], dtype=float)
    subjects = list(source["subjects"])
    if tensor.shape != (30, 4, 68, 6, 2001):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    if list(source.get("cell_order", [])) != CELL_ORDER:
        raise ValueError("Saved cell order is not verified order")
    if len(subjects) != 30 or len(set(subjects)) != 30:
        raise ValueError("Expected 30 unique participants")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("Tensor includes non-finite values")
    if not np.allclose(times_ms[[0, -1]], [-1000, 1000]):
        raise ValueError("Time vector is not in expected milliseconds")
    if not np.all(np.diff(times_ms) > 0):
        raise ValueError("Time vector is not increasing")
    if any(b not in band_labels for b in args.bands):
        raise ValueError("Requested band absent from tensor")

    labels = read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=str(SUBJECTS_DIR), verbose=False)
    centroids = {label.name: label.pos.mean(axis=0).tolist() for label in labels}
    if any(roi not in centroids for roi in roi_labels):
        raise ValueError("Atlas centroids do not cover every tensor ROI")

    all_exclusions = {}
    for mask_name in args.masks:
        exclusions = [f"{stem}-{hemi}" for stem in MASKS[mask_name] for hemi in ("lh", "rh")]
        if any(roi not in roi_labels for roi in exclusions):
            raise ValueError(f"{mask_name} exclusion absent from tensor ROI labels")
        all_exclusions[mask_name] = exclusions

    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "run_manifest.json"
    original_manifest = None
    if args.resume:
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "source_tensor_sha256": sha256(TENSOR_PATH),
            "masks": all_exclusions,
            "bands": args.bands,
            "contrasts": args.contrasts,
            "permutations": args.permutations,
            "seed": args.seed,
        }
        for key, value in expected.items():
            if original_manifest.get(key) != value:
                raise ValueError(f"Resume manifest mismatch for {key}")
        if original_manifest.get("status") == "complete":
            raise ValueError("Refusing to resume a completed run")
    manifest = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_tensor": str(TENSOR_PATH),
        "source_tensor_sha256": sha256(TENSOR_PATH),
        "saved_cell_order": CELL_ORDER,
        "n_participants": len(subjects),
        "participant_ids": subjects,
        "masks": all_exclusions,
        "bands": args.bands,
        "contrasts": args.contrasts,
        "permutations": args.permutations,
        "seed": args.seed,
        "tail": 0,
        "threshold": None,
        "n_jobs": args.jobs,
        "alpha": 0.05,
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
        "correction_family": "within each band x contrast map",
    }
    if original_manifest is not None:
        manifest["created_utc"] = original_manifest["created_utc"]
        manifest["resume_history"] = original_manifest.get("resume_history", []) + [{
            "resumed_utc": datetime.now(timezone.utc).isoformat(),
            "previous_status": original_manifest["status"],
            "previous_n_jobs": original_manifest["n_jobs"],
            "n_jobs": args.jobs,
            "reason": "Single-job worker interrupted to increase execution parallelism; completed test files reused",
        }]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    try:
        for mask_name in args.masks:
            exclusions = set(all_exclusions[mask_name])
            kept_indices = [i for i, roi in enumerate(roi_labels) if roi not in exclusions]
            kept_labels = [roi_labels[i] for i in kept_indices]
            if len(kept_labels) != 60:
                raise ValueError(f"Expected 60 kept ROIs for {mask_name}")
            positions = {roi: centroids[roi] for roi in kept_labels}
            info = create_info(list(positions), sfreq=1000, ch_types="eeg")
            info.set_montage(make_dig_montage(ch_pos=positions, coord_frame="head"))
            adjacency, _ = find_ch_adjacency(info, ch_type="eeg")
            mask_dir = args.output / mask_name
            mask_dir.mkdir(exist_ok=args.resume)
            selected = tensor[:, :, kept_indices, :, :]
            print(f"[MASK] {mask_name}: {len(exclusions)} excluded, {len(kept_labels)} retained", flush=True)
            for band in args.bands:
                band_index = band_labels.index(band)
                for kind in args.contrasts:
                    result_path = mask_dir / f"source_cluster_results_{band}_{kind}_EMG.pkl"
                    if result_path.exists():
                        if not args.resume:
                            raise FileExistsError(result_path)
                        with result_path.open("rb") as handle:
                            prior = pickle.load(handle)
                        if (
                            prior.get("n_subjects") != len(subjects)
                            or prior.get("source_tensor_sha256") != manifest["source_tensor_sha256"]
                            or prior.get("roi_labels") != kept_labels
                            or prior.get("contrast") != kind
                            or prior.get("band") != band
                        ):
                            raise ValueError(f"Existing test result inconsistent: {result_path}")
                        observed = prior["t_obs"]
                        clusters = prior["clusters"]
                        pvalues = prior["cluster_p_values"]
                        print(f"[REUSED] {mask_name} | {band} | {kind}", flush=True)
                    else:
                        print(f"[RUN] {mask_name} | {band} | {kind}", flush=True)
                        contrast = make_contrast(selected, band_index, kind)
                        expected_shape = (len(subjects), len(times_ms), len(kept_labels))
                        if contrast.shape != expected_shape:
                            raise ValueError(
                                f"Contrast shape {contrast.shape} is not {expected_shape}"
                            )
                        observed, clusters, pvalues, h0 = spatio_temporal_cluster_1samp_test(
                            contrast,
                            adjacency=adjacency,
                            n_permutations=args.permutations,
                            tail=0,
                            threshold=None,
                            n_jobs=args.jobs,
                            seed=args.seed,
                            out_type="indices",
                            verbose=False,
                        )
                        with result_path.open("xb") as handle:
                            pickle.dump({
                                "contrast": kind, "band": band, "t_obs": observed,
                                "clusters": clusters, "cluster_p_values": pvalues,
                                "H0": h0, "n_subjects": len(subjects),
                                "adjacency": adjacency, "roi_labels": kept_labels,
                                "time_vector_ms": times_ms, "cell_order": CELL_ORDER,
                                "mask_name": mask_name, "excluded_rois": sorted(exclusions),
                                "source_tensor_sha256": manifest["source_tensor_sha256"],
                            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    significant = np.flatnonzero(np.asarray(pvalues) < 0.05)
                    print(f"[DONE] {len(significant)} significant cluster(s)", flush=True)
                    for index in significant:
                        mask = cluster_mask(clusters[index], observed.shape)
                        values = observed[mask]
                        roi_indices = np.flatnonzero(mask.any(axis=0))
                        time_indices = np.flatnonzero(mask.any(axis=1))
                        mass = float(values.sum())
                        row = {
                            "mask": mask_name, "band": band, "contrast": kind,
                            "cluster_index_zero_based": int(index),
                            "p_fwer": float(pvalues[index]),
                            "direction": direction(kind, mass),
                            "cluster_mass_signed": mass,
                            "mean_t_signed": float(values.mean()),
                            "peak_abs_t": float(np.max(np.abs(values))),
                            "estimated_mean_dz_signed": float(values.mean() / np.sqrt(len(subjects))),
                            "estimated_peak_abs_dz": float(np.max(np.abs(values)) / np.sqrt(len(subjects))),
                            "start_ms": float(times_ms[time_indices[0]]),
                            "end_ms": float(times_ms[time_indices[-1]]),
                            "n_rois": len(roi_indices),
                            "n_time_roi_points": int(mask.sum()),
                            "roi_names": "; ".join(kept_labels[i] for i in roi_indices),
                        }
                        rows.append(row)
            del selected
        with (args.output / "significant_cluster_summary.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["n_significant_clusters"] = len(rows)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[COMPLETE] {len(rows)} significant clusters; output: {args.output}", flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--masks", nargs="+", choices=sorted(MASKS), default=["manuscript", "alternative"])
    parser.add_argument("--bands", nargs="+", choices=BANDS, default=BANDS)
    parser.add_argument("--contrasts", nargs="+", choices=CONTRASTS, default=CONTRASTS)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    TENSOR_PATH = arguments.tensor
    SUBJECTS_DIR = arguments.subjects_dir
    run(arguments)
