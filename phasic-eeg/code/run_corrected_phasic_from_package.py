"""Verify or rerun corrected normative phasic cluster maps from package inputs.

The primary mode starts from the original four-cell participant tensor and
applies log1p after cell averaging. The RT-controlled mode starts from the
saved trial-RT-adjusted four-cell means; it does not re-estimate trial slopes.
Only load these pickle files from the trusted local project package.
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("LMA", "LMR", "FMA", "FMR")
CONTRASTS = ("Condition", "Role", "Interaction")
INPUTS = {
    "primary": (ROOT / "data/inputs/phasic_primary/subject_level_time_resolved_tensor.pkl",
                "da7eef3d4ab637f747bb3ec65ca642548854910690c43b8ec4f2279c245c58a1"),
    "rt_controlled": (ROOT / "data/results/phasic_rt_controlled/rt_adjusted_cell_means.pkl",
                      "a43bc46745364e4e82e641133fe0de34c1d7bea21223d61b273e3f379a82f610"),
}
SAVED = {
    "primary": ROOT / "data/results/phasic_primary",
    "rt_controlled": ROOT / "data/results/phasic_rt_controlled",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensor(mode):
    path, expected_hash = INPUTS[mode]
    if sha256(path) != expected_hash:
        raise ValueError(f"Input hash differs from verified package: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    tensor = np.asarray(data["tensor"])
    if tensor.shape != (30, 4, 68, 6, 2001):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    if mode == "primary":
        if "cell_order" in data and tuple(data["cell_order"]) != CELLS:
            raise ValueError("Primary cell order differs from the verified order")
        tensor = np.log1p(tensor)
        time = np.asarray(data["time_vector"], dtype=float)
    else:
        if tuple(data["cell_order"]) != CELLS:
            raise ValueError("RT-adjusted cell order differs from the verified order")
        time = np.asarray(data["time_ms"], dtype=float)
    if np.max(np.abs(time)) <= 10:
        time = time * 1000
    if not np.isfinite(tensor).all():
        raise ValueError("Tensor contains nonfinite values")
    rois = tuple(data["roi_labels"])
    bands = tuple(data["band_labels"])
    if len(rois) != 68 or len(bands) != 6 or time.shape != (2001,):
        raise ValueError("Unexpected ROI, band, or time axis")
    return tensor, rois, bands, time


def contrast_map(tensor, band_index, kind):
    if kind == "Condition":
        left = np.mean(tensor[:, [0, 2]], axis=1)
        right = np.mean(tensor[:, [1, 3]], axis=1)
    elif kind == "Role":
        left = np.mean(tensor[:, [1, 0]], axis=1)
        right = np.mean(tensor[:, [3, 2]], axis=1)
    elif kind == "Interaction":
        left = np.mean(tensor[:, [1, 2]], axis=1)
        right = np.mean(tensor[:, [0, 3]], axis=1)
    else:
        raise ValueError(kind)
    return (left[:, :, band_index, :] - right[:, :, band_index, :]).transpose(0, 2, 1)


def adjacency_for(rois, subjects_dir):
    from mne import create_info, read_labels_from_annot
    from mne.channels import find_ch_adjacency, make_dig_montage

    labels = read_labels_from_annot(subject="fsaverage", parc="aparc",
                                    subjects_dir=str(subjects_dir))
    positions = {label.name: label.pos.mean(axis=0).tolist()
                 for label in labels if label.name in rois}
    missing = [roi for roi in rois if roi not in positions]
    if missing:
        raise ValueError(f"Missing fsaverage ROI labels: {missing}")
    ordered = {roi: positions[roi] for roi in rois}
    info = create_info(ch_names=list(ordered), sfreq=1000, ch_types="eeg")
    info.set_montage(make_dig_montage(ch_pos=ordered, coord_frame="head"))
    adjacency, _ = find_ch_adjacency(info, ch_type="eeg")
    return adjacency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=INPUTS, required=True)
    parser.add_argument("--run-permutations", action="store_true",
                        help="Otherwise verify all 18 observed t maps without writing files")
    parser.add_argument("--subjects-dir", type=Path,
                        help="Directory containing the fsaverage subject; required for permutations")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; required for permutations")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--band", help="Optional single band for a smoke test")
    parser.add_argument("--contrast", choices=CONTRASTS,
                        help="Optional single contrast for a smoke test")
    args = parser.parse_args()
    if args.run_permutations:
        if args.subjects_dir is None or args.output_dir is None:
            parser.error("--subjects-dir and --output-dir are required for permutations")
        if args.output_dir.exists():
            parser.error("output directory already exists; use a new path")

    tensor, rois, bands, time = load_tensor(args.mode)
    adjacency = adjacency_for(rois, args.subjects_dir) if args.run_permutations else None
    if args.run_permutations:
        from mne.stats import spatio_temporal_cluster_1samp_test
        args.output_dir.mkdir(parents=True, exist_ok=False)

    largest_difference = 0.0
    checked = 0
    for band_index, band in enumerate(bands):
        if args.band is not None and band != args.band:
            continue
        for kind in CONTRASTS:
            if args.contrast is not None and kind != args.contrast:
                continue
            values = contrast_map(tensor, band_index, kind)
            saved_path = SAVED[args.mode] / f"source_cluster_results_{band}_{kind}.pkl"
            with saved_path.open("rb") as stream:
                saved = pickle.load(stream)
            observed_t = ttest_1samp(values, 0, axis=0).statistic
            difference = float(np.max(np.abs(observed_t - saved["t_obs"])))
            largest_difference = max(largest_difference, difference)
            if not np.isfinite(difference) or difference > 1e-4:
                raise AssertionError(f"{band} {kind}: saved t-map mismatch {difference}")
            checked += 1
            print(f"{band} {kind}: t-map max |difference| = {difference:.8g}", flush=True)
            if args.run_permutations:
                seed = 20261007 if args.mode == "rt_controlled" else None
                result = spatio_temporal_cluster_1samp_test(
                    values, adjacency=adjacency, n_permutations=args.permutations,
                    tail=0, threshold=None, n_jobs=args.n_jobs, seed=seed,
                    verbose=False)
                target = args.output_dir / saved_path.name
                with target.open("wb") as stream:
                    pickle.dump({"band": band, "contrast": kind,
                                 "t_obs": result[0], "clusters": result[1],
                                 "cluster_p_values": result[2], "H0": result[3],
                                 "roi_labels": rois, "time_vector": time,
                                 "cell_order": CELLS, "n_subjects": len(values)}, stream)

    if checked == 0:
        raise ValueError("No band × contrast map matched the requested filter")
    print(f"Verified {checked} observed maps; largest t-map difference: {largest_difference:.8g}")
    if args.run_permutations and args.mode == "primary":
        print("Primary permutation p values may differ because the archived run did not fix a seed.")


if __name__ == "__main__":
    main()
