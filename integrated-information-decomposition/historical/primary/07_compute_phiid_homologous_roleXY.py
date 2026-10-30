"""
Script: 07_compute_phiid_homologous_roleXY.py

Purpose
-------
Compute source-level time-resolved Integrated Information Decomposition (ΦID)
in a role-consistent dyadic frame using homologous (same-ROI) pairing.

Pipeline role
-------------
Primary ΦID analysis.

Frame definition
----------------
- X := Leader
- Y := Follower

Pairing definition
------------------
Homologous (symmetry-preserving):
    X@roi  with  Y@same_roi

This differs from axis geometry, which uses cross-hemisphere pairing.

Inputs
------
dyadic_sourcelevel_time_resolved.pkl containing:
- p1_data / p2_data (trials × rois × bands × time)
- p1_info
- roi_labels
- band_labels

Outputs
-------
dyad_<id>_tensor_roleXY.pkl

Includes:
- phiid_tensor
- atom_labels
- tau
- role-mapping provenance
- block structure
- input transform

Notes
-----
- Uses P1 role labels + complement rule (p2_info ignored)
- Applies log1p(power)
- Gaussian ΦID with MMI redundancy
"""

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from phyid.calculate import calc_PhiID

# =============================================================================
# User configuration
# =============================================================================

INPUT_PATH = Path("PATH_TO_DATA") / "dyadic_sourcelevel_time_resolved.pkl"
OUTPUT_DIR = Path("PATH_TO_OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 1
ATOMS = ["rtr", "xtx", "yty", "sts"]
MIN_VALID_TRIALS = 10

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dyadic source tensor not found: {INPUT_PATH}\n"
        "Edit INPUT_PATH to point to dyadic_sourcelevel_time_resolved.pkl."
    )

# =============================================================================
# Console header
# =============================================================================

print("\n[RUN MODE] Homologous ΦID (role-consistent frame)")
print("[FRAME] X=Leader, Y=Follower")
print("[PAIRING] homologous: X@roi with Y@same_roi")
print(f"[INPUT]  {INPUT_PATH}")
print(f"[OUTPUT] {OUTPUT_DIR}\n")

# =============================================================================
# Helper functions
# =============================================================================

def get_xy_by_role(p1_data, p2_data, p1_info, trial_indices):
    """
    Enforce X=Leader, Y=Follower using P1 role + complement rule.
    """
    if len(trial_indices) == 0:
        return None, None, {"swap_xy": None}

    ti0 = int(trial_indices[0])
    role = (p1_info[ti0].get("role", "") or "").strip()

    if role == "Leader":
        return p1_data, p2_data, {"swap_xy": False}
    elif role == "Follower":
        return p2_data, p1_data, {"swap_xy": True}
    else:
        raise RuntimeError(f"Invalid role label: {role}")


def check_block_homogeneity(p1_info, trial_indices):
    roles = [(p1_info[i].get("role", "") or "").strip() for i in trial_indices]
    conds = [(p1_info[i].get("condition", "") or "").strip() for i in trial_indices]

    if len(set(roles)) != 1 or len(set(conds)) != 1:
        raise RuntimeError(f"Block homogeneity violation: roles={set(roles)}, conds={set(conds)}")

# =============================================================================
# Load data
# =============================================================================

with open(INPUT_PATH, "rb") as f:
    dyads = pickle.load(f)

# =============================================================================
# Main loop
# =============================================================================

for batch in tqdm(dyads, desc="Dyads"):
    dyad_id = batch["dyad"]

    out_path = OUTPUT_DIR / f"dyad_{dyad_id}_tensor_roleXY.pkl"
    if out_path.exists():
        tqdm.write(f"Skipping dyad {dyad_id}")
        continue

    p1_data = np.log1p(batch["p1_data"])
    p2_data = np.log1p(batch["p2_data"])
    p1_info = batch["p1_info"]

    roi_labels = batch["roi_labels"]
    band_labels = batch["band_labels"]

    n_trials, n_rois, n_bands, n_times = p1_data.shape

    # -------------------------------------------------------------------------
    # Define blocks
    # -------------------------------------------------------------------------
    block_trials = {1: [], 2: [], 3: [], 4: []}
    block_desc = {
        1: "Leader+Mirroring",
        2: "Leader+Matching",
        3: "Follower+Matching",
        4: "Follower+Mirroring",
    }

    for i in range(n_trials):
        role = (p1_info[i].get("role", "") or "").strip()
        cond = (p1_info[i].get("condition", "") or "").strip()

        if role == "Leader" and cond == "Mirroring":
            block_trials[1].append(i)
        elif role == "Leader" and cond == "Matching":
            block_trials[2].append(i)
        elif role == "Follower" and cond == "Matching":
            block_trials[3].append(i)
        elif role == "Follower" and cond == "Mirroring":
            block_trials[4].append(i)

    # -------------------------------------------------------------------------
    # Allocate tensor
    # -------------------------------------------------------------------------
    tensor = np.full(
        (4, n_rois, n_bands, n_times - 2 * TAU, len(ATOMS)),
        np.nan,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Compute ΦID
    # -------------------------------------------------------------------------
    for b, trials in block_trials.items():
        tqdm.write(f"Dyad {dyad_id} | Block {b}: {len(trials)} trials")

        if not trials:
            continue

        trials = np.asarray(trials)
        check_block_homogeneity(p1_info, trials)

        x_data, y_data, _ = get_xy_by_role(p1_data, p2_data, p1_info, trials)

        for roi in tqdm(range(n_rois), leave=False):
            for band in range(n_bands):
                for t in range(TAU, n_times - TAU):

                    x_p = x_data[trials, roi, band, t - TAU]
                    y_p = y_data[trials, roi, band, t - TAU]
                    x_f = x_data[trials, roi, band, t + TAU]
                    y_f = y_data[trials, roi, band, t + TAU]

                    valid = (~np.isnan(x_p) & ~np.isnan(y_p) &
                             ~np.isnan(x_f) & ~np.isnan(y_f))

                    if valid.sum() < MIN_VALID_TRIALS:
                        continue

                    src = np.stack([x_p[valid], y_p[valid]], axis=1)
                    trg = np.stack([x_f[valid], y_f[valid]], axis=1)

                    try:
                        atoms, _ = calc_PhiID(
                            src, trg,
                            tau=TAU,
                            kind="gaussian",
                            redundancy="MMI"
                        )

                        for i_atom, atom in enumerate(ATOMS):
                            val = atoms[atom]
                            tensor[b - 1, roi, band, t - TAU, i_atom] = (
                                np.mean(val) if isinstance(val, (list, np.ndarray)) else val
                            )

                    except Exception:
                        continue

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    out = {
        "dyad_id": dyad_id,
        "phiid_tensor": tensor,
        "atom_labels": ATOMS,
        "tau": TAU,
        "xy_frame": "X=Leader, Y=Follower",
        "pairing": "homologous (same ROI)",
        "mapping_rule": "P1 role + complement inference",
        "input_transform": "log1p(power)",
        "roi_labels": roi_labels,
        "band_labels": band_labels,
    }

    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    tqdm.write(f"Saved dyad {dyad_id}")

print("\n[✓] Homologous ΦID complete.")
