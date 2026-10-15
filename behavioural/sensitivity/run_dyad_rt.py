"""Dyad-level behavioural RT sensitivity analysis for four-cell median RT data.

The input CSV has one row per participant, role, and condition, with columns
participant_id, role, condition, and rt. Outputs are written to a new directory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


CELLS = ("LMA", "LMR", "FMA", "FMR")


def norm_pid(value: str) -> str:
    match = re.search(r"(?:dyad)?\s*(\d+)_P([12])", str(value), flags=re.I)
    if not match:
        raise ValueError(f"Unrecognized participant ID: {value!r}")
    return f"{int(match.group(1)):02d}_P{match.group(2)}"


def dyad_id(pid: str) -> str:
    return pid.split("_P")[0]


def contrasts(cells: np.ndarray) -> dict[str, np.ndarray]:
    """Cell order: Leader Matching, Leader Mirroring, Follower Matching, Follower Mirroring."""
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


def run(anova_csv: Path, output_dir: Path) -> None:
    if not anova_csv.is_file():
        raise FileNotFoundError(anova_csv)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    anova = pd.read_csv(anova_csv, dtype={"participant_id": str})
    assert len(anova) == 128
    assert set(anova.role) == {"Leader", "Follower"}
    assert set(anova.condition) == {"MATCHING", "MIRRORING"}
    assert not anova.duplicated(["participant_id", "role", "condition"]).any()
    anova["pid"] = anova.participant_id.map(norm_pid)
    anova["dyad"] = anova.pid.map(dyad_id)
    assert anova.pid.nunique() == 32 and anova.dyad.nunique() == 16
    rt_by_pid = {}
    for pid, group in anova.groupby("pid"):
        lookup = {(r.role, r.condition): float(r.rt) for r in group.itertuples()}
        assert len(lookup) == 4
        rt_by_pid[pid] = contrasts(np.array([lookup[("Leader", "MATCHING")], lookup[("Leader", "MIRRORING")], lookup[("Follower", "MATCHING")], lookup[("Follower", "MIRRORING")]]))
    rt_rows = []
    rt_dyads = []
    for effect in ("Condition", "Role", "Interaction"):
        values = []
        for dyad, group in anova.groupby("dyad"):
            pids = sorted(group.pid.unique())
            assert len(pids) == 2
            value = float(np.mean([rt_by_pid[pid][effect] for pid in pids]))
            values.append(value)
            rt_dyads.append({"dyad": dyad, "effect": effect, "contrast_ms": value})
        result = one_sample(np.array(values))
        rt_rows.append({"effect": effect, "n_dyads": 16, "df": 15, "F_1_15": result["t"] ** 2, **result})

    output_dir.mkdir(parents=True)
    pd.DataFrame(rt_rows).to_csv(output_dir / "behavioural_dyad_rt.csv", index=False)
    pd.DataFrame(rt_dyads).to_csv(output_dir / "behavioural_dyad_contrasts.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anova-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.anova_csv, args.output_dir)
