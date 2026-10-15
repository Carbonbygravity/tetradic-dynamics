"""Dyad-level tonic EEG sensitivity check using four-cell participant means.

Inputs are the parcellated power files, matched RT index, and the reported
RT-adjusted feature table. Outputs are written only to a new directory.
"""

from __future__ import annotations

import argparse
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
    matching the phasic interaction contrast.
    """
    lma, lmr, fma, fmr = cells
    return {
        "Condition": ((lma + fma) - (lmr + fmr)) / 2,
        "Role": ((lma + lmr) - (fma + fmr)) / 2,
        "Interaction": (fma - fmr) - (lma - lmr),
    }


def one_sample(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)) or len(x) < 3:
        raise ValueError("Non-finite or insufficient dyad contrasts")
    n = len(x)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return {"mean": mean, "sd": sd, "t": float("nan"), "p": float("nan"), "dz": float("nan"), "ci_low": mean, "ci_high": mean}
    se = sd / np.sqrt(n)
    stat = mean / se
    p = float(2 * t.sf(abs(stat), df=n - 1))
    half_width = float(t.ppf(0.975, df=n - 1) * se)
    return {"mean": mean, "sd": sd, "t": float(stat), "p": p, "dz": mean / sd, "ci_low": mean - half_width, "ci_high": mean + half_width}


def run_tonic(power_dir: Path, matched_csv: Path, reference_results_csv: Path, output_dir: Path) -> None:
    if not power_dir.is_dir():
        raise NotADirectoryError(power_dir)
    for required in (matched_csv, reference_results_csv):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    matched = pd.read_csv(matched_csv, dtype={"participant_id": str, "cell": str})
    matched["participant_id"] = matched["participant_id"].map(norm_pid)
    matched["cell"] = matched["cell"].str.upper().str.strip()
    matched["rt"] = pd.to_numeric(matched["rt"], errors="coerce")
    matched = matched.dropna(subset=["rt"])
    assert len(matched) == 13145
    assert not matched.duplicated(["participant_id", "cell", "trial_index"]).any()
    keys = set(zip(matched.participant_id, matched.cell, matched.trial_index))
    assert len(keys) == len(matched)

    files = sorted(power_dir.glob("*_parcellated_power.pkl"))
    assert len(files) == 30, f"Expected 30 tonic power files, got {len(files)}"
    cell_rows: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    band_labels = roi_labels = None
    used: set[tuple[str, str, int]] = set()
    for path in files:
        with path.open("rb") as handle:
            trials = pickle.load(handle)
        assert isinstance(trials, list) and trials
        pid = norm_pid(trials[0]["participant_id"])
        ordinal = defaultdict(int)
        for trial in trials:
            assert norm_pid(trial["participant_id"]) == pid
            role = str(trial["role"]).strip().lower()
            condition = str(trial["condition"]).strip().lower()
            cell = ("L" if role.startswith("l") else "F") + ("MA" if condition.startswith("match") else "MR")
            assert cell in CELLS
            key = (pid, cell, ordinal[cell])
            ordinal[cell] += 1
            if key not in keys:
                continue
            bp = np.asarray(trial["band_power"], dtype=np.float64)
            if band_labels is None:
                band_labels = list(trial["band_labels"])
                roi_labels = list(trial["roi_labels"])
            assert list(trial["band_labels"]) == band_labels
            assert list(trial["roi_labels"]) == roi_labels
            assert bp.shape == (6, 68) and np.all(np.isfinite(bp)) and np.all(bp >= 0)
            cell_rows[(pid, cell)].append(np.log1p(bp).reshape(-1))
            assert key not in used
            used.add(key)
    assert used == keys, f"Matched rows not reproduced: {len(keys-used)}; extra: {len(used-keys)}"
    assert band_labels is not None and roi_labels is not None
    features = [f"{roi} | {band}" for band in band_labels for roi in roi_labels]
    assert len(features) == 408
    participants = sorted({pid for pid, _ in cell_rows})
    assert len(participants) == 30
    assert all(all((pid, cell) in cell_rows for cell in CELLS) for pid in participants)
    cell_mean = {key: np.mean(np.stack(rows), axis=0) for key, rows in cell_rows.items()}
    partner_map = defaultdict(list)
    for pid in participants:
        partner_map[dyad_id(pid)].append(pid)
    assert len(partner_map) == 15 and all(len(v) == 2 for v in partner_map.values())

    reference = pd.read_csv(reference_results_csv)
    reference_condition = reference[reference.effect == "condition"]
    assert len(reference_condition) == 408 and set(reference_condition.feature) == set(features)
    assert int(reference_condition.p_lrt_fdr_sig.astype(str).str.lower().eq("true").sum()) == 279
    reference_coef = dict(zip(reference_condition.feature, reference_condition.coef))
    reference_sig = dict(zip(reference_condition.feature, reference_condition.p_lrt_fdr_sig.astype(str).str.lower().eq("true")))

    rows = []
    dyad_rows = []
    for effect in ("Condition", "Role", "Interaction"):
        matrix = []
        for dyad, partners in sorted(partner_map.items()):
            partner_contrasts = [contrasts(np.stack([cell_mean[(pid, cell)] for cell in CELLS]))[effect] for pid in sorted(partners)]
            dyad_contrast = np.mean(np.stack(partner_contrasts), axis=0)
            matrix.append(dyad_contrast)
            for feature, value in zip(features, dyad_contrast):
                dyad_rows.append({"dyad": dyad, "effect": effect, "feature": feature, "contrast": float(value)})
        matrix = np.stack(matrix)
        assert matrix.shape == (15, 408)
        for index, feature in enumerate(features):
            result = one_sample(matrix[:, index])
            row = {"effect": effect, "feature": feature, "n_dyads": 15, **result}
            if effect == "Condition":
                row["reference_rt_adjusted_coef"] = reference_coef[feature]
                row["reference_lrt_fdr_sig"] = bool(reference_sig[feature])
            rows.append(row)
    table = pd.DataFrame(rows)
    for effect in ("Condition", "Role", "Interaction"):
        mask = table.effect == effect
        p = table.loc[mask, "p"].to_numpy()
        assert np.all(np.isfinite(p))
        reject, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
        table.loc[mask, "q_bh_408"] = q
        table.loc[mask, "fdr_sig"] = reject
    output_dir.mkdir(parents=True)
    table.to_csv(output_dir / "tonic_dyad_features.csv", index=False)
    pd.DataFrame(dyad_rows).to_csv(output_dir / "tonic_dyad_contrasts.csv", index=False)

    verification = {
        "matched_rows_reproduced": len(used),
        "participants": len(participants),
        "dyads_tonic": len(partner_map),
        "tonic_cells_complete": True,
        "tonic_features": len(features),
        "reference_condition_lrt_fdr": 279,
        "tonic_fdr_counts": {
            effect: int(table.loc[table.effect == effect, "fdr_sig"].sum())
            for effect in ("Condition", "Role", "Interaction")
        },
    }
    (output_dir / "tonic_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-dir", type=Path, required=True)
    parser.add_argument("--matched-csv", type=Path, required=True)
    parser.add_argument("--reference-results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_tonic(args.power_dir, args.matched_csv, args.reference_results_csv, args.output_dir)
