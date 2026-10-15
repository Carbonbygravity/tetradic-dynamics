# Phasic EEG analyses

This folder contains source-level response-locked cluster analyses, sensitivity checks, and a runner for verifying saved band-by-contrast maps in the normative leader–follower coordination study.

The participant tensor cell order is `LMA`, `LMR`, `FMA`, `FMR`. The contrasts are Matching minus Mirroring, Leader-response minus Follower-response, and the Follower Condition difference minus the Leader Condition difference.

## Inputs and data access

Participant-level tensors, trial-level power and RT inputs, and saved cluster maps are not stored on GitHub. Access may be requested from the study team, subject to applicable consent, ethics, and institutional requirements. Place an approved verification input package under `phasic-eeg/data/`; Git excludes that directory. Keep all other approved inputs and generated outputs outside Git history. Load pickle files only from a trusted source.

The cluster generators also require the MNE `fsaverage` subject, including `aparc` labels and white-matter surfaces. Source reconstruction and preprocessing are upstream of these scripts.

## Verify saved maps

Use Python 3.11.11 and install the pinned dependencies:

```powershell
python -m pip install -r environment/phasic_clean_requirements_2026_10_14.txt
python code/run_phasic_clusters.py --mode primary
python code/run_phasic_clusters.py --mode rt_controlled
```

The runner checks all 18 observed t-statistic maps in each mode against the saved maps without writing outputs. To run fresh permutations, provide the MNE subjects directory and a new output path:

```powershell
python code/run_phasic_clusters.py --mode primary --run-permutations --subjects-dir PATH_TO_MNE_SUBJECTS_DIR --output-dir NEW_EMPTY_PATH --permutations 5000
```

Use `--mode rt_controlled` for the corresponding RT-controlled run. The primary saved run did not fix a random seed, so near-threshold cluster p values can vary across fresh runs; the RT-controlled mode uses seed `20261007`.

## Analysis scripts

Run from this folder with approved input paths:

```powershell
python primary/02_spatiotemporal_cluster_source.py --tensor PARTICIPANT_TENSOR_PKL --subjects-dir MNE_SUBJECTS_DIR --output-dir NEW_CLUSTER_DIR
python primary/09_report_normative_clusters.py --results-dir CLUSTER_DIR --report-dir NEW_REPORT_DIR
python primary/normative_space_frequency_cluster.py --tensor PARTICIPANT_TENSOR_PKL --subjects-dir MNE_SUBJECTS_DIR --output-dir NEW_SPACE_FREQUENCY_DIR
python primary/verify_normative_space_frequency_clusters.py --results-dir SPACE_FREQUENCY_DIR --output-dir NEW_SPACE_FREQUENCY_REPORT_DIR
python sensitivity/11_normative_rt_controlled_phasic.py --raw-dir RAW_POWER_DIR --rt-dir TRIAL_RT_DIR --subjects-dir MNE_SUBJECTS_DIR --output-dir NEW_RT_DIR
python sensitivity/12_normative_across_map_bonferroni.py --primary-dir CLUSTER_DIR --rt-controlled-dir RT_CLUSTER_DIR --output-dir NEW_ACROSS_MAP_DIR
python sensitivity/14_normative_condition_temporal_centering.py --source-dir CLUSTER_DIR --qa NEW_QA_DIR --output NEW_CENTERING_DIR
python sensitivity/normative_phasic_emg_recheck.py --tensor PARTICIPANT_TENSOR_PKL --subjects-dir MNE_SUBJECTS_DIR --output NEW_EMG_DIR
python sensitivity/run_dyad_phasic.py --tensor PARTICIPANT_TENSOR_PKL --primary-maps-dir CLUSTER_DIR --output-dir NEW_DYAD_DIR
```

The analysis scripts use NumPy 1.26.4, SciPy 1.15.1, pandas 2.2.2, and MNE 1.7.1 in the local analysis environment. The EMG check also uses Numba 0.60.0. The primary spatiotemporal tests use 5,000 permutations separately for each of 18 band-by-contrast maps. The dyad-level sensitivity uses 15 dyads as independent sign-flip units.

The saved-map runner, cluster reporter, across-map summary, space-frequency verifier, temporal-centering check, and dyad-level check were compared with saved results. Fresh primary spatiotemporal and space-frequency runs reproduced observed statistics, cluster masks, and adjacency. The RT-controlled source runner completed trial-to-RT alignment checks; its full map set and every optional EMG mask/map combination are not part of these local checks.
