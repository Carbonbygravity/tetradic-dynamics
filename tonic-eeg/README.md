# Tonic EEG analyses

This folder contains the trial-level, RT-adjusted tonic source-power model
and its role-invariance and dyad-level sensitivity checks for the normative
leader–follower coordination study.

## Inputs and data access

The scripts use approved participant-level parcellated-power files and a
matched RT index. These inputs are not public. Access may be requested from
the corresponding author, subject to applicable consent, ethics, and
institutional requirements. Do not commit participant files or pickle
inputs to GitHub; load pickle files only from a trusted source.

The dyad-level check also takes the reported feature table as
`--reference-results-csv`. The RT-adjusted dyad check takes the primary
model's output directory as `--prior-dir`. Each run writes to a new output
directory and refuses to overwrite an existing one.

## Environment

The local reruns used Python 3.11.11, NumPy 1.26.4, pandas 2.2.2,
SciPy 1.15.1, and statsmodels 0.13.5. The primary model processes 30
participants, 13,145 matched trials, and 408 ROI-by-band features.

## Commands

Run from this folder with approved input paths:

```powershell
python primary/01_trial_level_mixedlm_rt_covariate.py --eeg-dir POWER_DIR --matched-csv MATCHED_RT_CSV --output-dir NEW_PRIMARY_DIR
python sensitivity/01_jaccard_role_invariance_sensitivity.py --eeg-dir POWER_DIR --matched-csv MATCHED_RT_CSV --output-dir NEW_JACCARD_DIR
python sensitivity/run_dyad_tonic.py --power-dir POWER_DIR --matched-csv MATCHED_RT_CSV --reference-results-csv REPORTED_FEATURE_CSV --output-dir NEW_DYAD_DIR
python sensitivity/run_rt_adjusted_dyad_tonic.py --power-dir POWER_DIR --matched-csv MATCHED_RT_CSV --prior-dir PRIMARY_RESULTS_DIR --output-dir NEW_RT_DYAD_DIR
```

The reported feature table is also published at
[`results/mixedlm_rt_cov_results_fdr.csv`](results/mixedlm_rt_cov_results_fdr.csv).
The dyad-level scripts are sensitivity checks; they do not replace the
primary trial-level model.

## Verification limits

The primary staged script reproduced the 1,632-row feature tables
byte-for-byte against saved results. The dyad-level contrast tables and
RT-adjusted dyad outputs were independently compared with saved results.
The role-invariance Jaccard summary matched its saved summary; its two
optional detailed ROI tables had no saved counterpart for direct comparison.
Unseeded or floating-point-dependent outputs may differ at negligible
rounding precision across environments.
