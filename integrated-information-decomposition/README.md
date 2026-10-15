# Integrated information decomposition analyses

This folder contains the normative study's integrated information decomposition (ΦID) code. Primary analyses and their sensitivity checks are kept together.

## Script order

1. `primary/06_compute_phiid_crosshemi_axis_roleXY.py`: cross-hemisphere ΦID in the role-consistent Leader–Follower frame.
2. `primary/07_compute_phiid_homologous_roleXY.py`: homologous ΦID in that frame.
3. `primary/08_phiid_overlap_homologous_vs_crosshemi.py`: compare geometry-level outputs.
4. `sensitivity/09_phiid_strict_feature_tiering.py`: strict feature tiers.
5. `sensitivity/10_phiid_dyad_recurrence_summary.py`: dyad recurrence summaries.
6. `sensitivity/11_phiid_role_swap_validation.py`: role-swap and block-half checks.

## Inputs and setup

The scripts require approved dyadic source tensors and, for later stages, outputs produced by earlier stages. These participant-level inputs are not stored on GitHub. Access may be requested from the study team, subject to applicable consent, ethics, and institutional requirements. Keep inputs and generated outputs outside Git history. Load pickle files only from a trusted source.

Before running a script, set its `PATH_TO_...` values to the approved input, output, and group-label locations. The ΦID scripts require `phyid`. The Python imports used by this folder are NumPy, pandas, SciPy, tqdm, and phyid, with the standard-library modules shown in each script. The local analysis environment resolved these imports.

All six scripts passed Python syntax checks, and their staged copies match the study source files. A full tensor-level run requires the approved inputs and the computational resources appropriate to the analysis; the code listing alone does not generate the reported results.
