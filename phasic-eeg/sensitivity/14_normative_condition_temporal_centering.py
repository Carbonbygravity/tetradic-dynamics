"""Separate sustained-offset sensitivity for normative Condition maps.

For each participant, band, and ROI, subtract the full-epoch mean from the
Matching-minus-Mirroring response-locked difference before cluster inference.
This tests temporal variation beyond a participant-specific sustained offset;
it is not a baseline-normalized analysis or a test of response specificity.
"""

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from mne.stats import spatio_temporal_cluster_1samp_test


RUN_NAME = "Normative_Condition_Temporal_Centering_Run_01"
ORDER = ["LMA", "LMR", "FMA", "FMR"]
SEED = 20261013
BANDS = ["theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "gamma"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_and_validate(source: Path):
    tensor_file = source / "log_transformed_tensor.pkl"
    with tensor_file.open("rb") as stream:
        data = pickle.load(stream)
    tensor = data["tensor"]
    assert tensor.shape == (30, 4, 68, 6, 2001), tensor.shape
    assert list(data["cell_order"]) == ORDER
    assert list(data["band_labels"]) == BANDS
    assert np.isfinite(tensor).all()
    times = np.asarray(data["time_vector"])
    np.testing.assert_allclose(times, np.arange(-1000, 1001), atol=1e-8)

    metadata = {}
    adjacency = None
    for bi, band in enumerate(BANDS):
        file = source / f"source_cluster_results_{band}_Condition.pkl"
        with file.open("rb") as stream:
            old = pickle.load(stream)
        assert old["n_subjects"] == 30
        assert old["cell_order"] == ORDER
        assert old["contrast_definition"]["Condition"] == "(LMA+FMA) - (LMR+FMR)"
        assert list(old["roi_labels"]) == list(data["roi_labels"])
        assert old["t_obs"].shape == (2001, 68)
        current_adj = sparse.csr_matrix(old["adjacency"])
        if adjacency is None:
            adjacency = current_adj
        else:
            assert (adjacency != current_adj).nnz == 0
        field = condition_field(tensor, bi)
        t_rebuilt = stats.ttest_1samp(field, 0, axis=0).statistic
        error = float(np.max(np.abs(t_rebuilt - old["t_obs"])))
        np.testing.assert_allclose(t_rebuilt, old["t_obs"], rtol=2e-6, atol=2e-6)
        metadata[band] = dict(source_result=str(file), source_result_sha256=sha256(file),
                              max_abs_uncentered_t_difference=error)
    assert adjacency.shape == (68, 68)
    assert (adjacency != adjacency.T).nnz == 0
    return data, adjacency, metadata, tensor_file


def condition_field(tensor, band_index):
    """Condition contrast: (LMA + FMA)/2 - (LMR + FMR)/2."""
    matching = tensor[:, [0, 2]].mean(axis=1)[:, :, band_index, :]
    mirroring = tensor[:, [1, 3]].mean(axis=1)[:, :, band_index, :]
    return (matching - mirroring).transpose(0, 2, 1)


def cluster_summary(t_obs, clusters, p_values, times, rois):
    rows = []
    for index, (cluster, p_value) in enumerate(zip(clusters, p_values), start=1):
        time_indices, roi_indices = map(np.asarray, cluster)
        if len(time_indices) == 0:
            continue
        t_vals = t_obs[time_indices, roi_indices]
        rows.append(dict(cluster=index, p_within_map=float(p_value),
                         p_bonferroni_18_maps=min(1.0, float(p_value) * 18),
                         sign="positive" if float(t_vals.sum()) > 0 else "negative",
                         start_ms=float(times[time_indices.min()]),
                         end_ms=float(times[time_indices.max()]),
                         n_time_roi_points=int(len(time_indices)),
                         n_rois=int(len(np.unique(roi_indices))),
                         roi_labels=[str(rois[i]) for i in np.unique(roi_indices)],
                         cluster_t_sum=float(t_vals.sum()),
                         peak_abs_t=float(np.max(np.abs(t_vals)))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--band", choices=BANDS + ["all"], default="all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {args.output}")
    if args.qa.exists():
        raise FileExistsError(f"Refusing to reuse existing QA directory: {args.qa}")
    if args.permutations < 16:
        raise ValueError("At least 16 permutations required")
    data, adjacency, audit, tensor_file = load_and_validate(args.source_dir)
    args.output.mkdir(parents=True)
    args.qa.mkdir(parents=True)
    times = np.asarray(data["time_vector"])
    selected = BANDS if args.band == "all" else [args.band]
    specification = dict(run=RUN_NAME, created_utc=datetime.now(timezone.utc).isoformat(),
                         script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__)),
                         tensor=str(tensor_file), tensor_sha256=sha256(tensor_file),
                         cohort="Normative HC-HC; 30 participant rows, no dyad averaging",
                         cell_order=ORDER, contrast="((LMA+FMA)/2)-((LMR+FMR)/2)",
                         transform="For each participant, ROI and band: subtract the mean of the entire -1000 to +1000 ms Condition difference time course",
                         test="MNE spatio_temporal_cluster_1samp_test; original saved 68-ROI adjacency; two-sided participant sign flips",
                         n_permutations=args.permutations, seed=SEED, tail=0, threshold=None,
                         correction="Within each band x Condition map cluster FWER; Bonferroni x18 shown separately",
                         interpretation="Tests temporal variation beyond an epoch-constant Condition difference. Does not by itself identify a response-specific mechanism or distinguish baseline-relative power.",
                         source_t_map_audit=audit)
    save_json(args.output / "frozen_specification.json", specification)
    rows = []
    validations = []
    for band in selected:
        bi = BANDS.index(band)
        field = condition_field(data["tensor"], bi).astype(np.float64)
        offset = field.mean(axis=1, keepdims=True)
        centered = field - offset
        residual = float(np.max(np.abs(centered.mean(axis=1))))
        assert residual < 1e-10, (band, residual)
        assert np.isfinite(centered).all()
        print(f"Running {band}: {args.permutations} permutations", flush=True)
        t_obs, clusters, p_values, h0 = spatio_temporal_cluster_1samp_test(
            centered, adjacency=adjacency, n_permutations=args.permutations,
            tail=0, threshold=None, n_jobs=1, seed=SEED, out_type="indices",
            buffer_size=1000, verbose=False)
        expected_t = stats.ttest_1samp(centered, 0, axis=0).statistic
        np.testing.assert_allclose(t_obs, expected_t, rtol=2e-6, atol=2e-6)
        summary = cluster_summary(t_obs, clusters, p_values, times, data["roi_labels"])
        significant = [item for item in summary if item["p_within_map"] < .05]
        for item in summary:
            item["band"] = band
        rows.extend(summary)
        np.savez_compressed(args.output / f"{band}_map.npz", t_obs=t_obs,
                            cluster_p_values=p_values, H0=h0,
                            offset=offset[:, 0, :], time_vector_ms=times)
        with (args.output / f"{band}_clusters.pkl").open("xb") as stream:
            pickle.dump(dict(band=band, clusters=clusters, cluster_p_values=p_values,
                             roi_labels=data["roi_labels"], time_vector_ms=times), stream)
        validations.append(dict(band=band, n_clusters=len(summary),
                                n_significant_within_map=len(significant),
                                n_significant_across_18=sum(item["p_bonferroni_18_maps"] < .05 for item in summary),
                                max_abs_temporal_mean_residual=residual,
                                max_abs_t_reproduction_error=float(np.max(np.abs(t_obs - expected_t)))))
        print(f"{band}: {len(significant)} within-map clusters; "
              f"{validations[-1]['n_significant_across_18']} across-18 clusters", flush=True)
    save_json(args.output / "all_clusters.json", rows)
    save_json(args.qa / "validation.json", dict(passed=True, checks=validations,
                                                specification_sha256=sha256(args.output / "frozen_specification.json"),
                                                results_sha256={path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}))
    print("Completed and validated", flush=True)


if __name__ == "__main__":
    main()
