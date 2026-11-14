"""RT-adjusted dyad-level tonic sensitivity for the normative study.

Inputs are read-only. Outputs are written only beside this script. The primary
comparison is the same 15-dyad cell-contrast test with and without an RT slope.
This is not a refit of the archived trial-level mixed model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


CELLS = ("LMA", "LMR", "FMA", "FMR")
N_FEATURES = 408


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_pid(value: str) -> str:
    match = re.search(r"(?:dyad)?\s*(\d+)_P([12])", str(value), flags=re.I)
    if not match:
        raise ValueError(f"Unrecognized participant: {value!r}")
    return f"{int(match.group(1)):02d}_P{match.group(2)}"


def cell_of(trial: dict) -> str:
    role = str(trial["role"]).strip().lower()
    condition = str(trial["condition"]).strip().lower()
    cell = ("L" if role.startswith("l") else "F") + ("MA" if condition.startswith("match") else "MR")
    if cell not in CELLS:
        raise ValueError(f"Unrecognized cell: {cell}")
    return cell


def contrast(cells: np.ndarray) -> dict[str, np.ndarray]:
    lma, lmr, fma, fmr = cells
    return {
        "Condition": ((lma + fma) - (lmr + fmr)) / 2,
        "Role": ((lma + lmr) - (fma + fmr)) / 2,
        "Interaction": (fma - fmr) - (lma - lmr),
    }


def stats(matrix: np.ndarray, effect: str, features: list[str], model: str) -> list[dict]:
    assert matrix.shape == (15, N_FEATURES)
    means = matrix.mean(axis=0)
    sds = matrix.std(axis=0, ddof=1)
    assert np.all(sds > 0)
    se = sds / np.sqrt(15)
    ts = means / se
    ps = 2 * t.sf(np.abs(ts), df=14)
    qs = multipletests(ps, method="fdr_bh")[1]
    half = t.ppf(0.975, df=14) * se
    return [
        {
            "model": model, "effect": effect, "feature": feature, "n_dyads": 15,
            "mean": means[i], "sd": sds[i], "t": ts[i], "p": ps[i],
            "q_bh_408": qs[i], "fdr_sig": qs[i] < 0.05,
            "ci_low": means[i] - half[i], "ci_high": means[i] + half[i],
        }
        for i, feature in enumerate(features)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RT-adjusted dyad-level tonic sensitivity.")
    parser.add_argument("--power-dir", type=Path, required=True)
    parser.add_argument("--matched-csv", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    power = args.power_dir
    matched_path = args.matched_csv
    prior_dir = args.prior_dir
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    prior_manifest = json.loads((prior_dir / "source_manifest_tonic_rt.json").read_text(encoding="utf-8"))
    expected_hashes = {Path(item["path"]).name: item["sha256"] for item in prior_manifest["source_files"]}
    assert digest(matched_path) == expected_hashes[matched_path.name]
    matched = pd.read_csv(matched_path, dtype={"participant_id": str, "cell": str})
    matched["participant_id"] = matched["participant_id"].map(norm_pid)
    matched["cell"] = matched["cell"].str.upper().str.strip()
    matched["rt"] = pd.to_numeric(matched["rt"], errors="coerce")
    matched = matched.dropna(subset=["rt"]).copy()
    assert len(matched) == 13145
    assert not matched.duplicated(["participant_id", "cell", "trial_index"]).any()
    grouped = matched.groupby("participant_id")["rt"]
    matched["rt_z"] = (matched["rt"] - grouped.transform("mean")) / grouped.transform("std")
    assert np.isfinite(matched["rt_z"]).all()
    rt_lookup = {
        (row.participant_id, row.cell, row.trial_index): float(row.rt_z)
        for row in matched.itertuples(index=False)
    }
    assert len(rt_lookup) == 13145

    power_files = sorted(power.glob("*_parcellated_power.pkl"))
    assert len(power_files) == 30
    observations: dict[str, list[tuple[int, float, np.ndarray]]] = defaultdict(list)
    used: set[tuple[str, str, int]] = set()
    band_labels = roi_labels = None
    verified_hashes = {}
    for path in power_files:
        file_hash = digest(path)
        assert file_hash == expected_hashes[path.name], f"Input changed: {path}"
        verified_hashes[path.name] = file_hash
        with path.open("rb") as handle:
            trials = pickle.load(handle)
        assert isinstance(trials, list) and trials
        pid = norm_pid(trials[0]["participant_id"])
        ordinal = defaultdict(int)
        for trial in trials:
            assert norm_pid(trial["participant_id"]) == pid
            cell = cell_of(trial)
            key = (pid, cell, ordinal[cell])
            ordinal[cell] += 1
            if key not in rt_lookup:
                continue
            bp = np.asarray(trial["band_power"], dtype=np.float64)
            assert bp.shape == (6, 68) and np.isfinite(bp).all() and (bp >= 0).all()
            if band_labels is None:
                band_labels = list(trial["band_labels"])
                roi_labels = list(trial["roi_labels"])
            assert list(trial["band_labels"]) == band_labels
            assert list(trial["roi_labels"]) == roi_labels
            observations[pid].append((CELLS.index(cell), rt_lookup[key], np.log1p(bp).reshape(-1)))
            assert key not in used
            used.add(key)
    assert used == set(rt_lookup) and len(observations) == 30
    features = [f"{roi} | {band}" for band in band_labels for roi in roi_labels]
    assert len(features) == N_FEATURES

    # Fit one RT slope per participant after removing four cell means. This is
    # the participant-specific-slope version of the RT-adjusted dyad contrast.
    # Also estimate a common within-participant RT slope across all participants
    # as a more pooled alternative. Both use the same matched trials.
    by_pid = {}
    pooled_num = np.zeros(N_FEATURES)
    pooled_den = 0.0
    for pid, rows in sorted(observations.items()):
        cell_idx = np.array([row[0] for row in rows], dtype=int)
        rt = np.array([row[1] for row in rows], dtype=np.float64)
        y = np.stack([row[2] for row in rows])
        counts = np.bincount(cell_idx, minlength=4)
        assert np.all(counts >= 50)
        cell_y = np.stack([y[cell_idx == j].mean(axis=0) for j in range(4)])
        cell_rt = np.array([rt[cell_idx == j].mean() for j in range(4)])
        rt_resid = rt - cell_rt[cell_idx]
        y_resid = y - cell_y[cell_idx]
        den = float(rt_resid @ rt_resid)
        assert den > 0
        num = rt_resid @ y_resid
        own_slope = num / den
        pooled_num += num
        pooled_den += den
        by_pid[pid] = {"cell_y": cell_y, "cell_rt": cell_rt, "own_slope": own_slope, "count": len(rows)}
    pooled_slope = pooled_num / pooled_den

    model_cells = {"unadjusted": {}, "rt_adjusted_pooled_slope": {}, "rt_adjusted_participant_slope": {}}
    for pid, entry in by_pid.items():
        raw = entry["cell_y"]
        mean_rt = entry["cell_rt"][:, None]
        model_cells["unadjusted"][pid] = raw
        model_cells["rt_adjusted_pooled_slope"][pid] = raw - mean_rt * pooled_slope[None, :]
        model_cells["rt_adjusted_participant_slope"][pid] = raw - mean_rt * entry["own_slope"][None, :]

    dyads = sorted({pid.split("_P")[0] for pid in by_pid})
    assert len(dyads) == 15
    result_rows = []
    dyad_rows = []
    summaries = {}
    for model, cells_by_pid in model_cells.items():
        summaries[model] = {}
        for effect in ("Condition", "Role", "Interaction"):
            matrix = np.stack([
                np.mean(np.stack([contrast(cells_by_pid[f"{dyad}_P{partner}"])[effect] for partner in ("1", "2")]), axis=0)
                for dyad in dyads
            ])
            result_rows.extend(stats(matrix, effect, features, model))
            summaries[model][effect] = {
                "positive_features": int((matrix.mean(axis=0) > 0).sum()),
                "raw_p_lt_05": int(sum(row["p"] < 0.05 for row in result_rows[-N_FEATURES:])),
                "bh_fdr_q_lt_05": int(sum(row["fdr_sig"] for row in result_rows[-N_FEATURES:])),
            }
            for dyad, values in zip(dyads, matrix):
                dyad_rows.extend({"model": model, "effect": effect, "dyad": dyad, "feature": feature, "contrast": value}
                                 for feature, value in zip(features, values))

    table = pd.DataFrame(result_rows)
    prior = pd.read_csv(prior_dir / "tonic_dyad_features.csv")
    for effect in ("Condition", "Role", "Interaction"):
        old_effect = prior[prior.effect == effect].set_index("feature")
        new_effect = table[(table.model == "unadjusted") & (table.effect == effect)].set_index("feature")
        assert set(old_effect.index) == set(new_effect.index)
        assert np.allclose(old_effect.loc[new_effect.index, "mean"], new_effect["mean"], atol=1e-12)
        assert np.allclose(old_effect.loc[new_effect.index, "p"], new_effect["p"], atol=1e-12)
    output_dir.mkdir(parents=True)
    table.to_csv(output_dir / "tonic_rt_adjusted_dyad_features.csv", index=False)
    pd.DataFrame(dyad_rows).to_csv(output_dir / "tonic_rt_adjusted_dyad_contrasts.csv", index=False)
    report = {
        "scope": "Normative dyad data",
        "n_trials": len(used), "n_participants": len(by_pid), "n_dyads": len(dyads), "n_features": len(features),
        "rt_z": "z-scored within participant across matched trials, sample SD",
        "outcome": "trial-level log1p power",
        "cell_adjustment": "cell mean log1p power minus RT slope times cell mean rt_z; evaluates each cell at participant mean RT",
        "inference": "two-sided t on 15 equally weighted dyad contrasts, df=14; BH-FDR separately over 408 features/effect",
        "models": {
            "unadjusted": "exact reproduction of 2026-10-13 dyad-level cell-mean test",
            "rt_adjusted_pooled_slope": "one common within-participant RT slope per feature",
            "rt_adjusted_participant_slope": "separate within-cell RT slope per participant and feature",
        },
        "caveat": "Neither RT-adjusted dyad test is the archived trial-level mixed model with participant random effects and LRT; 279-versus-new count is not a like-for-like model comparison.",
        "summary": summaries,
        "input_hashes_verified_against_prior_manifest": {matched_path.name: expected_hashes[matched_path.name], **verified_hashes},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_trials": len(used), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
