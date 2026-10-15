# Phasic EEG contrast verification

This folder contains code for checking the response-locked phasic EEG band-by-contrast maps from the leader–follower coordination study. It covers the primary and RT-controlled phasic analyses; it does not reproduce preprocessing, source reconstruction, trial-level RT adjustment, or the paper's behavioural, tonic, and information-decomposition analyses.

The analysis uses four participant cells in this order: `LMA`, `LMR`, `FMA`, `FMR`. The reported contrasts are Matching minus Mirroring, Leader-response minus Follower-response, and `(LMR + FMA) - (LMA + FMR)` for the interaction.

## Data access

Derived participant-level tensors and archived cluster maps are **not stored in this public repository**. They are maintained separately with access controlled by the study team. After obtaining access, place the supplied `data/` directory under `phasic-eeg/`. Do not commit or redistribute that directory. The runner checks the input tensors against their verified SHA-256 hashes before loading them. Python pickle files should only be opened from a trusted source.

## Verify the saved maps

Use Python 3.11.11 and install the pinned dependencies:

```powershell
python -m pip install -r environment/phasic_clean_requirements_2026_10_14.txt
python code/run_corrected_phasic_from_package.py --mode primary
python code/run_corrected_phasic_from_package.py --mode rt_controlled
```

Each command checks all 18 observed t-statistic maps against the archived results without writing outputs. A fresh 5,000-permutation run additionally needs the MNE `fsaverage` subject with `aparc` annotations and white-matter surfaces:

```powershell
python code/run_corrected_phasic_from_package.py --mode primary --run-permutations --subjects-dir PATH_TO_MNE_SUBJECTS_DIR --output-dir NEW_EMPTY_PATH --permutations 5000
```

Use `--mode rt_controlled` for the corresponding RT-controlled rerun. The primary archived run did not fix a random seed, so cluster p values close to a threshold may vary across reruns; the RT-controlled mode uses seed `20261007`.

