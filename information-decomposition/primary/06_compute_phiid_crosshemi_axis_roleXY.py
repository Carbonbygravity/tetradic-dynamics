"""
Script: 06_compute_phiid_crosshemi_axis_roleXY.py

Purpose
-------
Compute source-level time-resolved Integrated Information Decomposition (ΦID)
in a role-consistent dyadic frame using axis (cross-hemisphere) pairing.

Pipeline role
-------------
Primary statistical analysis script for axis-geometry ΦID results.

Frame definition
----------------
- X := Leader
- Y := Follower

Pairing definition
------------------
Axis (cross-hemisphere):
    X@roi  with  Y@opposite_hemisphere(roi)

This is distinct from homologous geometry.

Inputs
------
dyadic_sourcelevel_time_resolved.pkl containing, per dyad:
- p1_data / p2_data: trials × rois × bands × time
- p1_info
- roi_labels
- band_labels

Outputs
-------
One pickle per dyad:
    dyad_<id>_tensor_axis_roleXY.pkl

Stored fields include:
- phiid_tensor
- atom_labels
- tau
- role-mapping provenance
- block homogeneity checks
- opposite-hemisphere index mapping
- roi_labels / band_labels

Notes
-----
- Uses only P1 role labels plus complement inference.
- Ignores p2_info role because dataset metadata are not complementary.
- Applies log1p(power) before ΦID.
- Uses Gaussian ΦID with MMI redundancy.
"""

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from phyid.calculate import calc_PhiID

# =============================================================================
# User settings
# =============================================================================

INPUT_PATH = Path("PATH_TO_DYADIC_SOURCELEVEL_DATA") / "dyadic_sourcelevel_time_resolved.pkl"
OUTPUT_DIR = Path("PATH_TO_PHIID_OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 1
ATOMS_OF_INTEREST = ["rtr", "xtx", "yty", "sts"]
MIN_VALID_TRIALS = 10

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dyadic source tensor not found: {INPUT_PATH}\n"
        "Edit INPUT_PATH to point to dyadic_sourcelevel_time_resolved.pkl."
    )

# =============================================================================
# Console header
# =============================================================================

print("\n[RUN MODE] AXIS ΦID in role-consistent frame: X=Leader, Y=Follower")
print("[PAIRING] axis (cross-hemisphere): X@roi with Y@opp_hemi(roi)")
print(f"[INPUT]   {INPUT_PATH}")
print(f"[OUTPUT]  {OUTPUT_DIR}")
print("[NOTE]    Writing dyad_<id>_tensor_axis_roleXY.pkl")
print("[FIX]     Mapping derived from P1 role + complement rule (p2_info role ignored)\n")

# =============================================================================
# Helper functions
# =============================================================================

def opposite_hemi_label(label: str):
    """
    Return the opposite-hemisphere label for a given ROI label.
    Supports suffixes:
    - '-lh' <-> '-rh'
    - '_lh' <-> '_rh'
    """
    low = label.lower()
    if low.endswith("-lh"):
        return label[:-3] + "-rh"
    if low.endswith("-rh"):
        return label[:-3] + "-lh"
    if low.endswith("_lh"):
        return label[:-3] + "_rh"
    if low.endswith("_rh"):
        return label[:-3] + "_lh"
    return None


def get_xy_by_role_from_p1_only(p1_data, p2_data, p1_info, trial_indices):
    """
    Enforce X=Leader, Y=Follower within a block.

    Uses only P1 role labels and infers P2 role by complement rule.
    """
    if len(trial_indices) == 0:
        return None, None, {
            "swap_xy": None,
            "x_is": None,
            "y_is": None,
            "p1_role": None,
            "p2_role": None,
        }

    ti0 = int(trial_indices[0])
    p1_role = (p1_info[ti0].get("role", "") or "").strip()

    if p1_role == "Leader":
        return p1_data, p2_data, {
            "swap_xy": False,
            "x_is": "P1(Leader)",
            "y_is": "P2(Follower)",
            "p1_role": "Leader",
            "p2_role": "Follower",
        }

    if p1_role == "Follower":
        return p2_data, p1_data, {
            "swap_xy": True,
            "x_is": "P2(Leader)",
            "y_is": "P1(Follower)",
            "p1_role": "Follower",
            "p2_role": "Leader",
        }

    return None, None, {
        "swap_xy": "INVALID_ROLE",
        "x_is": None,
        "y_is": None,
        "p1_role": p1_role,
        "p2_role": None,
    }


def check_block_homogeneity(p1_info, trial_indices):
    """
    Verify that a block contains only one role and one condition label.
    """
    roles = [(p1_info[i].get("role", "") or "").strip() for i in trial_indices]
    conds = [(p1_info[i].get("condition", "") or "").strip() for i in trial_indices]

    uniq_roles = sorted(set(roles))
    uniq_conds = sorted(set(conds))
    ok = (len(uniq_roles) == 1) and (len(uniq_conds) == 1)

    return ok, {
        "unique_roles": uniq_roles,
        "unique_conditions": uniq_conds,
    }

# =============================================================================
# Load dyad data
# =============================================================================

with open(INPUT_PATH, "rb") as f:
    dyads = pickle.load(f)

# =============================================================================
# Main loop over dyads
# =============================================================================

for batch in tqdm(dyads, desc="Total Dyad Progress"):
    dyad_id = batch["dyad"]

    output_path = OUTPUT_DIR / f"dyad_{dyad_id}_tensor_axis_roleXY.pkl"
    if output_path.exists():
        tqdm.write(f"-> Skipping Dyad {dyad_id}: output already exists ({output_path.name})")
        continue

    p1_data = np.log1p(batch["p1_data"])
    p2_data = np.log1p(batch["p2_data"])
    p1_info = batch["p1_info"]

    roi_labels = batch["roi_labels"]
    band_labels = batch["band_labels"]

    n_trials, n_rois, n_bands, n_times = p1_data.shape

    # -------------------------------------------------------------------------
    # Build opposite-hemisphere lookup
    # -------------------------------------------------------------------------
    idx_by_label = {lab: i for i, lab in enumerate(roi_labels)}
    opp_index = [None] * n_rois
    missing_opp = 0

    for i, lab in enumerate(roi_labels):
        opp_lab = opposite_hemi_label(lab)
        if opp_lab is not None and opp_lab in idx_by_label:
            opp_index[i] = idx_by_label[opp_lab]
        else:
            missing_opp += 1

    # -------------------------------------------------------------------------
    # Sort trials into 4 experimental blocks using P1 labels
    # -------------------------------------------------------------------------
    block_trials = {1: [], 2: [], 3: [], 4: []}
    block_desc = {
        1: "Leader+Mirroring",
        2: "Leader+Matching",
        3: "Follower+Matching",
        4: "Follower+Mirroring",
    }

    for trial_idx in range(n_trials):
        role = (p1_info[trial_idx].get("role", "") or "").strip()
        cond = (p1_info[trial_idx].get("condition", "") or "").strip()

        if role == "Leader" and cond == "Mirroring":
            block_trials[1].append(trial_idx)
        elif role == "Leader" and cond == "Matching":
            block_trials[2].append(trial_idx)
        elif role == "Follower" and cond == "Matching":
            block_trials[3].append(trial_idx)
        elif role == "Follower" and cond == "Mirroring":
            block_trials[4].append(trial_idx)

    # -------------------------------------------------------------------------
    # Allocate output tensor
    # Shape: blocks × rois × bands × time × atoms
    # -------------------------------------------------------------------------
    final_tensor = np.full(
        (4, n_rois, n_bands, n_times - 2 * TAU, len(ATOMS_OF_INTEREST)),
        np.nan,
        dtype=np.float32,
    )

    block_xy_mapping = {}
    block_homogeneity = {}

    # -------------------------------------------------------------------------
    # Compute ΦID within each block
    # -------------------------------------------------------------------------
    for block_num, trials_in_block in block_trials.items():
        tqdm.write(f"Dyad {dyad_id}, Block {block_num} ({block_desc[block_num]}): Found {len(trials_in_block)} trials.")

        if not trials_in_block:
            block_xy_mapping[block_num] = {
                "swap_xy": None,
                "x_is": None,
                "y_is": None,
                "note": "EMPTY_BLOCK",
            }
            block_homogeneity[block_num] = {
                "ok": None,
                "details": "EMPTY_BLOCK",
            }
            continue

        block_trial_indices = np.asarray(trials_in_block, dtype=int)

        # ---------------------------------------------------------------------
        # Hard check: block homogeneity
        # ---------------------------------------------------------------------
        ok, details = check_block_homogeneity(p1_info, block_trial_indices)
        block_homogeneity[block_num] = {
            "ok": bool(ok),
            "details": details,
        }

        if not ok:
            raise RuntimeError(
                f"Block homogeneity violation in dyad {dyad_id}, block {block_num}: {details}"
            )

        # ---------------------------------------------------------------------
        # Enforce X=Leader, Y=Follower
        # ---------------------------------------------------------------------
        x_data, y_data, map_info = get_xy_by_role_from_p1_only(
            p1_data, p2_data, p1_info, block_trial_indices
        )

        if x_data is None or y_data is None or map_info.get("swap_xy") == "INVALID_ROLE":
            raise RuntimeError(
                f"Invalid/ambiguous XY mapping in dyad {dyad_id}, block {block_num}: {map_info}"
            )

        block_xy_mapping[block_num] = {
            **map_info,
            "note": "X=Leader, Y=Follower enforced via P1 role + complement rule",
        }

        tqdm.write(
            f"    -> XY mapping: X={map_info['x_is']} | Y={map_info['y_is']} | "
            f"swap={map_info['swap_xy']} | "
            f"(P1={map_info['p1_role']}, P2={map_info['p2_role']})"
        )

        # ---------------------------------------------------------------------
        # ΦID computation: axis pairing
        # X@roi with Y@opposite_hemisphere(roi)
        # ---------------------------------------------------------------------
        for roi in tqdm(range(n_rois), desc=f"  Dyad {dyad_id}, Block {block_num}", leave=False):
            opp = opp_index[roi]
            if opp is None:
                continue

            for band in range(n_bands):
                for t in range(TAU, n_times - TAU):
                    try:
                        x_past = x_data[block_trial_indices, roi, band, t - TAU]
                        y_past = y_data[block_trial_indices, opp, band, t - TAU]
                        x_future = x_data[block_trial_indices, roi, band, t + TAU]
                        y_future = y_data[block_trial_indices, opp, band, t + TAU]

                        valid = (
                            ~np.isnan(x_past) & ~np.isnan(y_past) &
                            ~np.isnan(x_future) & ~np.isnan(y_future)
                        )

                        if valid.sum() < MIN_VALID_TRIALS:
                            continue

                        src = np.stack([x_past[valid], y_past[valid]], axis=1)
                        trg = np.stack([x_future[valid], y_future[valid]], axis=1)

                        atoms, _ = calc_PhiID(
                            src,
                            trg,
                            tau=TAU,
                            kind="gaussian",
                            redundancy="MMI",
                        )

                        for a_idx, atom in enumerate(ATOMS_OF_INTEREST):
                            val = atoms[atom]
                            final_tensor[block_num - 1, roi, band, t - TAU, a_idx] = (
                                np.mean(val) if isinstance(val, (list, np.ndarray)) else val
                            )

                    except Exception as e:
                        tqdm.write(
                            f"[ERROR] Dyad {dyad_id}, B{block_num}, R{roi}, Band{band}, T{t}: {e}"
                        )
                        continue

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    out_obj = {
        "dyad_id": dyad_id,
        "phiid_tensor": final_tensor,
        "atom_labels": ATOMS_OF_INTEREST,
        "tau": TAU,

        "xy_frame": "X=Leader, Y=Follower",
        "pairing_frame": "axis (cross-hemisphere): X@roi with Y@opp_hemi(roi)",
        "mapping_rule": "P1 role + complement rule (p2_info role ignored)",
        "block_desc": block_desc,
        "block_xy_mapping": block_xy_mapping,
        "block_homogeneity": block_homogeneity,
        "min_valid_trials": MIN_VALID_TRIALS,
        "input_transform": "log1p(power)",

        "roi_labels": roi_labels,
        "band_labels": band_labels,
        "opp_index": opp_index,
        "opp_missing_count": int(missing_opp),
    }

    with open(output_path, "wb") as f:
        pickle.dump(out_obj, f)

    tqdm.write(f"-> Saved ROLE-XY AXIS tensor for Dyad {dyad_id} to {output_path.name}")

print("\n[SUCCESS] All axis (role-consistent) source-level dyads have been processed.")
