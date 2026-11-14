"""Conservative across-map correction of saved normative phasic clusters.

Each source file already contains cluster-level permutation p values controlling
the familywise error rate within one frequency-band x contrast map. Multiplying
each p value by 18 (capped at 1) controls the familywise error rate across the
18 prespecified maps by the union bound. Primary and RT-controlled analyses
are treated as two separate sensitivity families. No permutations are rerun.
"""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


BANDS = ("theta", "low_alpha", "high_alpha", "low_beta", "high_beta", "gamma")
CONTRASTS = ("Condition", "Role", "Interaction")
CELL_ORDER = ("LMA", "LMR", "FMA", "FMR")
N_MAPS = len(BANDS) * len(CONTRASTS)
ALPHA = 0.05


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_cluster_file(path, band, contrast, reference):
    with path.open("rb") as f:
        d = pickle.load(f)
    if d["band"] != band or d["contrast"] != contrast:
        raise ValueError(f"Map identity mismatch: {path}")
    if d["n_subjects"] != 30 or tuple(d["cell_order"]) != CELL_ORDER:
        raise ValueError(f"Unexpected sample or cell order: {path}")
    t_obs = np.asarray(d["t_obs"], dtype=float)
    roi = tuple(d["roi_labels"])
    times = np.asarray(d["time_vector"], dtype=float)
    p_values = np.asarray(d["cluster_p_values"], dtype=float)
    clusters = d["clusters"]
    h0 = np.asarray(d["H0"], dtype=float)
    if (t_obs.shape != (2001, 68) or len(roi) != 68 or len(times) != 2001
            or len(clusters) != len(p_values) or len(h0) != 5000
            or not np.isfinite(t_obs).all() or not np.isfinite(p_values).all()
            or not np.isfinite(h0).all() or np.any((p_values < 0) | (p_values > 1))):
        raise ValueError(f"Invalid cluster array or permutation count: {path}")
    if reference is not None and (roi != reference[0]
                                  or not np.allclose(times, reference[1], atol=1e-6)):
        raise ValueError(f"ROI or time-axis mismatch: {path}")
    return t_obs, roi, times, p_values, clusters, (roi, times)


def cluster_row(family, band, contrast, index, p, cluster, t_obs, roi, times):
    if not isinstance(cluster, (tuple, list)) or len(cluster) != 2:
        raise ValueError(f"Unexpected cluster representation: {family}/{band}/{contrast}")
    ti = np.asarray(cluster[0], dtype=int)
    ri = np.asarray(cluster[1], dtype=int)
    if (len(ti) == 0 or len(ti) != len(ri) or np.any(ti < 0)
            or np.any(ti >= 2001) or np.any(ri < 0) or np.any(ri >= 68)):
        raise ValueError(f"Invalid paired cluster points: {family}/{band}/{contrast}")
    tvals = t_obs[ti, ri]
    mean_t = float(tvals.mean())
    adjusted = min(1.0, float(p) * N_MAPS)
    return {
        "analysis": family,
        "band": band,
        "contrast": contrast,
        "cluster_index_1based": index,
        "p_within_map": float(p),
        "p_across_18_bonferroni": adjusted,
        "significant_within_map": bool(p < ALPHA),
        "survives_across_18_maps": bool(adjusted < ALPHA),
        "start_ms": int(round(float(times[ti.min()]))),
        "end_ms": int(round(float(times[ti.max()]))),
        "n_roi": int(np.unique(ri).size),
        "n_cluster_points": int(len(ti)),
        "mean_t": mean_t,
        "mean_dz": mean_t / np.sqrt(30),
        "direction": "positive" if mean_t > 0 else "negative",
        "roi_labels": [roi[i] for i in np.unique(ri)],
    }


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Apply across-map Bonferroni correction to saved maps.")
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--rt-controlled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    INPUTS = {
        "primary_results_dir": args.primary_dir,
        "rt_controlled_sensitivity": args.rt_controlled_dir,
    }
    OUTPUT = args.output_dir
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; existing results are protected: {OUTPUT}")
    all_clusters = []
    map_rows = []
    input_hashes = {}
    for family, directory in INPUTS.items():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        expected_names = {f"source_cluster_results_{b}_{c}.pkl"
                          for b in BANDS for c in CONTRASTS}
        actual_names = {p.name for p in directory.glob("source_cluster_results_*.pkl")}
        if actual_names != expected_names:
            raise ValueError(f"Expected exactly 18 maps in {directory}; "
                             f"missing={expected_names - actual_names}, "
                             f"extra={actual_names - expected_names}")
        reference = None
        input_hashes[family] = {}
        for band in BANDS:
            for contrast in CONTRASTS:
                path = directory / f"source_cluster_results_{band}_{contrast}.pkl"
                t_obs, roi, times, p, clusters, axes = read_cluster_file(
                    path, band, contrast, reference)
                if reference is None:
                    reference = axes
                input_hashes[family][path.name] = sha256(path)
                rows = [cluster_row(family, band, contrast, i, value, cluster,
                                    t_obs, roi, times)
                        for i, (value, cluster) in enumerate(zip(p, clusters), 1)]
                all_clusters.extend(rows)
                map_rows.append({
                    "analysis": family, "band": band, "contrast": contrast,
                    "n_clusters": len(rows),
                    "n_within_map_significant": sum(r["significant_within_map"] for r in rows),
                    "n_across_map_significant": sum(r["survives_across_18_maps"] for r in rows),
                    "min_p_within_map": min((r["p_within_map"] for r in rows), default=1.0),
                })

    # Nothing is written until all 36 input files and cluster masks pass QA.
    OUTPUT.mkdir(parents=False, exist_ok=False)
    significant = [r for r in all_clusters if r["significant_within_map"]]
    compact = [{**r, "roi_labels": "; ".join(r["roi_labels"])} for r in significant]
    columns = ["analysis", "band", "contrast", "cluster_index_1based",
               "p_within_map", "p_across_18_bonferroni", "significant_within_map",
               "survives_across_18_maps", "start_ms", "end_ms", "n_roi",
               "n_cluster_points", "mean_t", "mean_dz", "direction", "roi_labels"]
    write_csv(OUTPUT / "within_map_significant_clusters.csv", compact, columns)
    write_csv(OUTPUT / "map_summary.csv", map_rows,
              ["analysis", "band", "contrast", "n_clusters",
               "n_within_map_significant", "n_across_map_significant",
               "min_p_within_map"])
    with (OUTPUT / "audit.json").open("w", encoding="utf-8") as f:
        json.dump({"method": "Bonferroni across 18 band x contrast maps; "
                             "p_across_18_bonferroni = min(1, 18 * p_within_map)",
                   "alpha": ALPHA, "maps_per_family": N_MAPS,
                   "families": list(INPUTS), "input_sha256": input_hashes,
                   "script_sha256": sha256(Path(__file__)),
                   "all_clusters": all_clusters}, f, indent=2)

    lines = ["# Across-map correction of normative phasic clusters", "",
             "Each analysis contains 18 band x contrast maps. The saved cluster p values "
             "already control familywise error within a map. The additional Bonferroni "
             "correction multiplies each within-map p by 18 (capped at 1). The primary "
             "and RT-controlled analyses are separate families; no EEG or permutation "
             "analysis was rerun. This conservative check does not change the saved "
             "within-map p values.", "",
             "| Analysis | Within-map significant clusters | Survive across 18 maps |",
             "| --- | ---: | ---: |"]
    for family in INPUTS:
        subset = [r for r in significant if r["analysis"] == family]
        survivors = [r for r in subset if r["survives_across_18_maps"]]
        lines.append(f"| {family} | {len(subset)} | {len(survivors)} |")
    lines.extend(["", "## Clusters surviving across-map correction", "",
                  "| Analysis | Band | Contrast | Window (ms) | Within-map p | Across-map p |",
                  "| --- | --- | --- | ---: | ---: | ---: |"])
    for r in significant:
        if r["survives_across_18_maps"]:
            lines.append(f"| {r['analysis']} | {r['band']} | {r['contrast']} | "
                         f"{r['start_ms']} to {r['end_ms']} | "
                         f"{r['p_within_map']:.4f} | "
                         f"{r['p_across_18_bonferroni']:.4f} |")
    lines.extend(["", "The full within-map significant-cluster list, including those "
                  "that do not survive the across-map check, is in "
                  "within_map_significant_clusters.csv. The 18-map threshold is "
                  "0.05/18 = 0.002777... for a within-map p value."])
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[COMPLETE] {OUTPUT}")
    for family in INPUTS:
        subset = [r for r in significant if r["analysis"] == family]
        survivors = [r for r in subset if r["survives_across_18_maps"]]
        print(f"{family}: within-map={len(subset)}, across-map={len(survivors)}")


if __name__ == "__main__":
    main()
