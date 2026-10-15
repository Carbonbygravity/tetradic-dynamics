"""RT-controlled sensitivity of the normative phasic clusters.

Each participant's retained trial power is log1p transformed, regressed on the
matched within-participant rt_z at every ROI x band x time point, then adjusted
to rt_z=0 before averaging into the verified response-event cells. The same
18 within-map cluster tests as the normative primary analysis are applied to
the adjusted cell means. Existing source and cluster outputs are read-only.
"""

import argparse
import csv
import gc
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
from mne import create_info, read_labels_from_annot
from mne.channels import find_ch_adjacency, make_dig_montage
from mne.stats import spatio_temporal_cluster_1samp_test


CELLS = ("LMA", "LMR", "FMA", "FMR")
N_PERMUTATIONS = 5000
SEED = 20261007
CHUNK_TIME = 100


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def subject_files():
    raw = {p.name.removesuffix("_time_resolved_parcellation.pkl"): p
           for p in SOURCE.glob("*_time_resolved_parcellation.pkl")}
    rt = {p.name.removesuffix("_time_resolved_parcellation_rt_resid.pkl"): p
          for p in RT_SOURCE.glob("*_time_resolved_parcellation_rt_resid.pkl")}
    if set(raw) != set(rt) or len(raw) != 30:
        raise ValueError(f"Expected the same 30 subjects: raw={len(raw)}, RT={len(rt)}")
    return [(s, raw[s], rt[s]) for s in sorted(raw)]


def cell_of(meta):
    roles = {"Leader": "L", "Follower": "F"}
    conditions = {"Matching": "MA", "Mirroring": "MR"}
    return roles[meta["role"]] + conditions[meta["condition"]]


def participant_key(value):
    match = re.fullmatch(r"(?:dyad)?0*(\d+)_P([12])", str(value))
    if match is None:
        raise ValueError(f"Unrecognized participant identifier: {value!r}")
    return int(match.group(1)), int(match.group(2))


def load_validate(subject, raw_path, rt_path, expected):
    # Load the RT file first, retain only its small trial table, then release
    # a 1+ GB residual array before opening raw trial power.
    with rt_path.open("rb") as f:
        rd = pickle.load(f)
    table = rd["trial_table"].copy()
    rt_meta = rd["trial_metadata"]
    rt_roi = tuple(rd["roi_labels"])
    rt_band = tuple(rd["band_labels"])
    rt_times = np.asarray(rd["times"], dtype=float)
    if participant_key(rd["participant_id"]) != participant_key(subject):
        raise ValueError(f"{subject}: RT participant ID mismatch")
    del rd
    gc.collect()

    with raw_path.open("rb") as f:
        d = pickle.load(f)
    x = d["data"]
    n = x.shape[0]
    if (participant_key(d["participant_id"]) != participant_key(subject)
            or len(rt_meta) != n or len(table) != n):
        raise ValueError(f"{subject}: raw/RT trial or participant mismatch")
    if x.ndim != 4 or x.shape[1:] != (68, 6, 2001):
        raise ValueError(f"{subject}: unexpected trial tensor {x.shape}")
    roi = tuple(d["roi_labels"])
    band = tuple(d["band_labels"])
    times = np.asarray(d["times"], dtype=float)
    if roi != rt_roi or band != rt_band or not np.array_equal(times, rt_times):
        raise ValueError(f"{subject}: raw/RT feature axes differ")
    if expected is not None and (roi != expected[0] or band != expected[1]
                                 or not np.array_equal(times, expected[2])):
        raise ValueError(f"{subject}: feature axes differ from cohort reference")
    labels = np.asarray([cell_of(m) for m in d["trial_metadata"]])
    if any(a != b for a, b in zip(d["trial_metadata"], rt_meta)):
        raise ValueError(f"{subject}: raw and RT trial metadata differ")
    if not np.array_equal(table["trial_idx_raw"].to_numpy(), np.arange(n)):
        raise ValueError(f"{subject}: RT trial indices not in raw tensor order")
    if not np.array_equal(table["cell"].to_numpy(), labels):
        raise ValueError(f"{subject}: RT cell labels do not match raw trial labels")
    z = table["rt_z"].to_numpy(dtype=np.float64)
    valid_rt = np.isfinite(z)
    n_missing_rt = int(np.sum(~valid_rt))
    x, labels, z = x[valid_rt], labels[valid_rt], z[valid_rt]
    if len(z) < 2 or np.std(z) < 0.1:
        raise ValueError(f"{subject}: invalid RT covariate")
    counts = {c: int(np.sum(labels == c)) for c in CELLS}
    if min(counts.values()) < 2:
        raise ValueError(f"{subject}: insufficient trials in a cell: {counts}")
    return x, labels, z, (roi, band, times), counts, n_missing_rt


def adjusted_cell_means(x, labels, z):
    """Log trial power, subtract each feature's OLS RT slope at observed RT."""
    n, n_roi, n_band, n_time = x.shape
    z0 = z - z.mean()
    denom = np.dot(z0, z0)
    means = np.empty((4, n_roi, n_band, n_time), dtype=np.float32)
    cell_masks = [labels == c for c in CELLS]
    zmeans = [float(np.mean(z[mask])) for mask in cell_masks]
    for start in range(0, n_time, CHUNK_TIME):
        stop = min(start + CHUNK_TIME, n_time)
        raw = x[:, :, :, start:stop]
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError(f"Invalid raw power at time slice {start}:{stop}")
        y = np.log1p(raw.astype(np.float64, copy=False))
        flat = y.reshape(n, -1)
        slope = (z0 @ flat / denom).reshape(n_roi, n_band, stop - start)
        for i, mask in enumerate(cell_masks):
            means[i, :, :, start:stop] = (
                np.mean(y[mask], axis=0) - slope * zmeans[i]
            ).astype(np.float32)
    return means


def adjacency_for(roi_labels):
    labels = read_labels_from_annot(subject="fsaverage", parc="aparc",
                                    subjects_dir=str(FSAVERAGE))
    by_name = {label.name: label.pos.mean(axis=0).tolist() for label in labels}
    if set(roi_labels) - set(by_name):
        raise ValueError(f"Missing ROI centroids: {set(roi_labels) - set(by_name)}")
    info = create_info(ch_names=list(roi_labels), sfreq=1000, ch_types="eeg")
    info.set_montage(make_dig_montage(ch_pos={r: by_name[r] for r in roi_labels},
                                      coord_frame="head"))
    return find_ch_adjacency(info, ch_type="eeg")[0]


def contrast(tensor, band, kind):
    a, b, c, d = [tensor[:, i, :, band, :] for i in range(4)]
    if kind == "Condition":
        difference = (a + c - b - d) / 2  # Matching - Mirroring
    elif kind == "Role":
        difference = (a + b - c - d) / 2  # Leader - Follower
    elif kind == "Interaction":
        difference = (b + c - a - d) / 2  # Follower M-M - Leader M-M
    else:
        raise ValueError(kind)
    return difference.transpose(0, 2, 1)


def summarize_cluster(cluster, t_obs, p, time_ms, roi):
    ti, ri = [np.asarray(v, dtype=int) for v in cluster]
    t_values = t_obs[ti, ri]
    return {
        "p_within_map": float(p),
        "start_ms": float(time_ms[ti.min()]),
        "end_ms": float(time_ms[ti.max()]),
        "n_roi": int(np.unique(ri).size),
        "n_points": int(ti.size),
        "mean_t": float(t_values.mean()),
        "mean_dz": float(t_values.mean() / np.sqrt(30)),
        "direction": "positive" if t_values.mean() > 0 else "negative",
        "roi_labels": [roi[i] for i in np.unique(ri)],
    }


def main():
    global SOURCE, RT_SOURCE, OUTPUT, FSAVERAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--rt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--clusters-only", action="store_true")
    args = parser.parse_args()
    SOURCE = args.raw_dir
    RT_SOURCE = args.rt_dir
    OUTPUT = args.output_dir
    FSAVERAGE = args.subjects_dir
    if args.clusters_only:
        with (OUTPUT / "rt_adjusted_cell_means.pkl").open("rb") as f:
            saved = pickle.load(f)
        with (OUTPUT / "input_audit.json").open("r", encoding="utf-8") as f:
            resumed_audit = json.load(f)
        tensor = saved["tensor"]
        files = [(s, None, None) for s in saved["subjects"]]
        expected = (tuple(saved["roi_labels"]), tuple(saved["band_labels"]),
                    np.asarray(saved["time_ms"], dtype=float))
        if tuple(saved["cell_order"]) != CELLS or tensor.shape != (30, 4, 68, 6, 2001):
            raise ValueError("Saved adjusted tensor does not match the run specification")
        if len(resumed_audit["subjects"]) != 30 or resumed_audit["seed"] != SEED:
            raise ValueError("Saved input audit does not match the run specification")
        resumed_audit["cluster_script_sha256"] = file_sha256(Path(__file__))
        resumed_audit["cluster_jobs"] = 12
        with (OUTPUT / "input_audit.json").open("w", encoding="utf-8") as f:
            json.dump(resumed_audit, f, indent=2)
        print("[RESUMED] Loaded validated 30-participant adjusted tensor", flush=True)
    else:
        files = subject_files()
        expected = None
        tensor = None
        audit = []
        for si, (subject, raw_path, rt_path) in enumerate(files):
            x, labels, z, axes, counts, n_missing_rt = load_validate(
                subject, raw_path, rt_path, expected)
            if expected is None:
                expected = axes
                tensor = np.empty((len(files), 4, 68, 6, 2001), dtype=np.float32)
            audit.append({"subject": subject, "n_trials_total": len(z) + n_missing_rt,
                          "n_trials_with_rt": len(z), "n_missing_rt": n_missing_rt,
                          "cell_counts_with_rt": counts,
                          "rt_z_mean": float(z.mean()), "rt_z_sd": float(z.std())})
            if not args.validate_only:
                tensor[si] = adjusted_cell_means(x, labels, z)
            del x
            gc.collect()
            print(f"[VALIDATED] {si + 1}/30 {subject}: {counts}", flush=True)

        if args.validate_only:
            print("[DONE] Input and trial-to-RT alignment validated for all 30 participants.")
            return

    roi, bands, times = expected
    time_ms = times * 1000 if np.max(np.abs(times)) <= 10 else times
    if not np.isfinite(tensor).all():
        raise ValueError("Adjusted tensor contains nonfinite values")
    adjacency = adjacency_for(roi)
    if not args.clusters_only:
        if OUTPUT.exists():
            raise FileExistsError(f"Separate run already exists: {OUTPUT}")
        OUTPUT.mkdir(parents=False, exist_ok=False)
        with (OUTPUT / "rt_adjusted_cell_means.pkl").open("wb") as f:
            pickle.dump({"tensor": tensor, "subjects": [v[0] for v in files],
                         "cell_order": CELLS, "roi_labels": roi, "band_labels": bands,
                         "time_ms": time_ms, "scale": "mean log1p trial power at rt_z=0"}, f)
        with (OUTPUT / "input_audit.json").open("w", encoding="utf-8") as f:
            json.dump({"subjects": audit, "n_subjects": len(files),
                       "method": "Within-participant OLS: log1p(trial power) ~ 1 + rt_z; adjusted to rt_z=0 before cell averaging",
                       "raw_input_dir": str(SOURCE), "rt_input_dir": str(RT_SOURCE),
                       "script_sha256": file_sha256(Path(__file__)),
                       "n_permutations": N_PERMUTATIONS, "seed": SEED}, f, indent=2)
    rows = []
    for bi, band in enumerate(bands):
        for kind in ("Condition", "Role", "Interaction"):
            out = OUTPUT / f"source_cluster_results_{band}_{kind}.pkl"
            if out.exists():
                with out.open("rb") as f:
                    saved_map = pickle.load(f)
                if (saved_map["band"] != band or saved_map["contrast"] != kind
                        or saved_map["n_subjects"] != len(files)
                        or tuple(saved_map["cell_order"]) != CELLS):
                    raise ValueError(f"Existing map does not match specification: {out}")
                t_obs, clusters, p_values = (saved_map["t_obs"],
                                              saved_map["clusters"],
                                              saved_map["cluster_p_values"])
                print(f"[REUSED] {band} {kind}", flush=True)
            else:
                values = contrast(tensor, bi, kind)
                result = spatio_temporal_cluster_1samp_test(
                    values, adjacency=adjacency, n_permutations=N_PERMUTATIONS,
                    tail=0, threshold=None, n_jobs=12, seed=SEED, verbose=False)
                t_obs, clusters, p_values, h0 = result
                with out.open("wb") as f:
                    pickle.dump({"band": band, "contrast": kind, "t_obs": t_obs,
                                 "clusters": clusters, "cluster_p_values": p_values,
                                 "H0": h0, "roi_labels": roi, "time_vector": time_ms,
                                 "cell_order": CELLS, "n_subjects": len(files),
                                 "scale": "mean log1p trial power at rt_z=0",
                                 "rt_adjustment": "within-participant OLS"}, f)
            sig = [summarize_cluster(cl, t_obs, p, time_ms, roi)
                   for cl, p in zip(clusters, p_values) if p < 0.05]
            rows.append({"band": band, "contrast": kind,
                         "n_significant_clusters": len(sig),
                         "significant_clusters": sig})
            print(f"[SAVED] {band} {kind}: {len(sig)} significant", flush=True)
    with (OUTPUT / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (OUTPUT / "cluster_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["band", "contrast", "cluster", "p_within_map", "start_ms",
                         "end_ms", "n_roi", "n_points", "mean_t", "mean_dz", "direction"])
        for row in rows:
            for ci, cl in enumerate(row["significant_clusters"], start=1):
                writer.writerow([row["band"], row["contrast"], ci, cl["p_within_map"],
                                 cl["start_ms"], cl["end_ms"], cl["n_roi"],
                                 cl["n_points"], cl["mean_t"], cl["mean_dz"], cl["direction"]])
    print("[COMPLETE] All 18 maps processed", flush=True)


if __name__ == "__main__":
    main()
