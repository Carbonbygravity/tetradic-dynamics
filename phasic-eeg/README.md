# Phasic EEG analyses

This folder contains source-level response-period-onset-aligned cluster analyses and
their sensitivity checks for the normative leader–follower coordination
study. The participant tensor cell order is LMA, LMR, FMA, FMR. Condition
is Matching minus Mirroring, Role is Leader minus Follower, and the
interaction is the Follower Condition difference minus the Leader Condition
difference.

## Inputs and data access

Participant-level tensors, trial-level power files, RT files, and saved
cluster maps are not stored in the repository. Access may be
requested from the corresponding author, subject to applicable consent,
ethics, and institutional requirements. Place the saved-map verification
inputs under `phasic-eeg/data/`; Git excludes that directory. Keep other
participant inputs and generated outputs outside Git history. Load pickle
inputs only from a trusted source. The source-level cluster generators also require the MNE
`fsaverage` subject, including `aparc` labels and white-matter surfaces.
Use a fresh output path for each run.

## Environment

The phasic verification runner in the public repository specifies
Python 3.11.11 and pinned dependencies in
`environment/phasic_clean_requirements_2026_10_14.txt`. The analysis
scripts here use NumPy 1.26.4, SciPy 1.15.1, pandas 2.2.2, and MNE 1.7.1.
Install pandas 2.2.2 and, for the EMG sensitivity script, Numba 0.60.0
in addition to the pinned phasic environment. Source reconstruction
and preprocessing are upstream of this package.

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

## Primary analysis

Run from this folder with the required input paths:

```powershell
python primary/02_spatiotemporal_cluster_source.py --tensor PARTICIPANT_TENSOR_PKL --subjects-dir MNE_SUBJECTS_DIR --output-dir NEW_CLUSTER_DIR
python primary/09_report_normative_clusters.py --results-dir CLUSTER_DIR --report-dir NEW_REPORT_DIR
python primary/normative_space_frequency_cluster.py --tensor PARTICIPANT_TENSOR_PKL --subjects-dir MNE_SUBJECTS_DIR --output-dir NEW_SPACE_FREQUENCY_DIR
python primary/verify_normative_space_frequency_clusters.py --results-dir SPACE_FREQUENCY_DIR --output-dir NEW_SPACE_FREQUENCY_REPORT_DIR
```

The spatiotemporal tests use 5,000 permutations separately for each of
18 band-by-contrast maps, controlling cluster-level familywise error within
each map without an additional correction across maps. The primary run did not fix a random seed;
near-threshold cluster p values can vary between fresh permutation runs.
The saved contrast maps, cluster masks, and ROI adjacency can be checked
independently of that permutation variability.

## Sensitivity analyses

```powershell
python sensitivity/11_normative_rt_controlled_phasic.py --raw-dir RAW_POWER_DIR --rt-dir TRIAL_RT_DIR --subjects-dir MNE_SUBJECTS_DIR --output-dir NEW_RT_DIR
python sensitivity/12_normative_across_map_bonferroni.py --primary-dir CLUSTER_DIR --rt-controlled-dir RT_CLUSTER_DIR --output-dir NEW_ACROSS_MAP_DIR
python sensitivity/14_normative_condition_temporal_centering.py --source-dir CLUSTER_DIR --qa NEW_QA_DIR --output NEW_CENTERING_DIR
python sensitivity/normative_phasic_emg_recheck.py --tensor PARTICIPANT_TENSOR_PKL --subjects-dir MNE_SUBJECTS_DIR --output NEW_EMG_DIR
python sensitivity/run_dyad_phasic.py --tensor PARTICIPANT_TENSOR_PKL --primary-maps-dir CLUSTER_DIR --output-dir NEW_DYAD_DIR
```

The EMG check includes the manuscript ROI mask and a separately labelled
alternative mask. The dyad-level analysis treats 15 dyads, rather than 30
participants, as independent sign-flip units. Sensitivity analyses do not
replace the primary cluster tests. Its interaction contrast is written without
the primary script's factor of 1/2; this positive scaling leaves its one-sample
t map unchanged, but raw contrast amplitudes should not be compared directly.

## Verification limits

The cluster reporter, across-map summary, space-frequency verifier,
temporal-centering outputs, and dyad-level run were compared with saved
results. Fresh primary spatiotemporal and space-frequency runs reproduced
observed statistics, cluster masks, and adjacency, with expected
permutation-p variation. The RT-controlled runner passed trial-to-RT
alignment checks but was not independently rerun through all 18 maps.
The EMG check reproduced one selected high-beta Condition map; every
optional mask and map was not independently checked.
