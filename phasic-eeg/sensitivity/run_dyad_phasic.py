"""Dyad-level response-locked phasic EEG cluster sensitivity check.

The 30-participant tensor and saved primary maps must be supplied explicitly.
The analysis uses 15 dyads as sign-flip units, 5,000 permutations per map,
and the same band, contrast, and adjacency definitions as the primary maps.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

CELLS = ("LMA", "LMR", "FMA", "FMR")
SEED = 20261013
N_PERM = 5000


def write_json(output_dir: Path, name: str, obj: dict) -> None:
    (output_dir / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def norm_pid(value: str) -> str:
    match = re.search(r"(?:dyad)?\s*(\d+)_P([12])", str(value), flags=re.I)
    if not match:
        raise ValueError(f"Unrecognized participant ID: {value!r}")
    return f"{int(match.group(1)):02d}_P{match.group(2)}"


def dyad_id(pid: str) -> str:
    return pid.split("_P")[0]


def contrasts(cells: np.ndarray) -> dict[str, np.ndarray]:
    """cells axis 0 must be LMA, LMR, FMA, FMR.

    Interaction is Follower(Matching-Mirroring) - Leader(Matching-Mirroring),
    matching the stated interaction definition and saved primary maps.
    """
    lma, lmr, fma, fmr = cells
    return {
        "Condition": ((lma + fma) - (lmr + fmr)) / 2,
        "Role": ((lma + lmr) - (fma + fmr)) / 2,
        "Interaction": (fma - fmr) - (lma - lmr),
    }


def cluster_extent(cluster, n_time: int, n_roi: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        return np.where(cluster)
    if isinstance(cluster, (tuple, list)) and len(cluster) == 2:
        return np.asarray(cluster[0], dtype=int), np.asarray(cluster[1], dtype=int)
    raise ValueError("Unexpected MNE cluster representation")


def run_phasic(tensor_path: Path, maps_dir: Path, output_dir: Path) -> None:
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)
    if not maps_dir.is_dir():
        raise NotADirectoryError(maps_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    from mne.stats import spatio_temporal_cluster_1samp_test

    with tensor_path.open("rb") as handle:
        record = pickle.load(handle)
    x = np.asarray(record["tensor"])
    assert x.shape == (30, 4, 68, 6, 2001), x.shape
    assert list(record["cell_order"]) == list(CELLS)
    assert np.all(np.isfinite(x))
    subjects = [norm_pid(s) for s in record["subjects"]]
    assert len(subjects) == 30 and len(set(subjects)) == 30
    dyads = sorted({dyad_id(s) for s in subjects})
    assert len(dyads) == 15
    partner_indices = [[i for i, pid in enumerate(subjects) if dyad_id(pid) == dyad] for dyad in dyads]
    assert all(len(indices) == 2 for indices in partner_indices)
    bands = list(record["band_labels"])
    rois = list(record["roi_labels"])
    times = np.asarray(record["time_vector"])
    assert len(bands) == 6 and len(rois) == 68 and len(times) == 2001
    results = []
    map_arrays = {}
    maximum_original_t_difference = 0.0

    for bi, band in enumerate(bands):
        for ci, effect in enumerate(("Condition", "Role", "Interaction")):
            original_path = maps_dir / f"source_cluster_results_{band}_{effect}.pkl"
            with original_path.open("rb") as handle:
                original = pickle.load(handle)
            assert original["n_subjects"] == 30
            assert original["cell_order"] == list(CELLS)
            assert list(original["roi_labels"]) == rois
            assert np.allclose(original["time_vector"], times)
            assert original["adjacency"].shape == (68, 68)
            participant = contrasts(x[:, :, :, bi, :].transpose(1, 0, 2, 3))[effect].transpose(0, 2, 1)
            # The above contracts the first (cell) axis while retaining participant rows.
            assert participant.shape == (30, 2001, 68)
            t_check = ttest_1samp(participant, 0, axis=0).statistic
            discrepancy = float(np.max(np.abs(t_check - original["t_obs"])))
            maximum_original_t_difference = max(maximum_original_t_difference, discrepancy)
            assert discrepancy < 1e-4, (band, effect, discrepancy)
            dyad_maps = np.stack([np.mean(participant[indices], axis=0) for indices in partner_indices])
            assert dyad_maps.shape == (15, 2001, 68)
            print(f"PHASIC {band} {effect}: 15 dyads, 5000 sign flips", flush=True)
            t_obs, clusters, pvals, null = spatio_temporal_cluster_1samp_test(
                dyad_maps, adjacency=original["adjacency"], n_permutations=N_PERM,
                tail=0, threshold=None, out_type="indices", n_jobs=4,
                seed=SEED + bi * 3 + ci,
            )
            key = f"{band}__{effect}"
            map_arrays[f"{key}__t_obs"] = t_obs.astype(np.float32)
            map_arrays[f"{key}__null"] = np.asarray(null, dtype=np.float32)
            for index, (cluster, p) in enumerate(zip(clusters, pvals)):
                time_i, roi_i = cluster_extent(cluster, len(times), len(rois))
                assert len(time_i) == len(roi_i) and len(time_i) > 0
                results.append({
                    "band": band, "effect": effect, "cluster_index": index,
                    "p_within_map": float(p), "significant_within_map": bool(p < 0.05),
                    "n_points": len(time_i), "n_rois": len(set(roi_i.tolist())),
                    "time_start_ms": float(np.min(times[time_i])), "time_end_ms": float(np.max(times[time_i])),
                    "sum_t": float(np.sum(t_obs[time_i, roi_i])),
                    "peak_abs_t": float(np.max(np.abs(t_obs[time_i, roi_i]))),
                })
            print(f"COMPLETE {band} {effect}: {sum(pvals < 0.05)} significant clusters", flush=True)
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "phasic_dyad_maps.npz", **map_arrays)
    pd.DataFrame(results).to_csv(output_dir / "phasic_dyad_clusters.csv", index=False)
    table = pd.DataFrame(results)
    write_json(output_dir, "phasic_verification.json", {
        "participants": len(subjects), "dyads": len(dyads), "shape": list(x.shape),
        "cell_order": list(CELLS), "bands": bands, "rois": len(rois), "time_points": len(times),
        "n_maps": len(bands) * 3, "permutations_per_map": N_PERM,
        "max_original_t_map_difference": maximum_original_t_difference,
        "within_map_significant_clusters": int(table.significant_within_map.sum()) if not table.empty else 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--primary-maps-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_phasic(args.tensor, args.primary_maps_dir, args.output_dir)
