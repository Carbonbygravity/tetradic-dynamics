# Behavioural reaction-time analyses

This folder contains reaction-time (RT) analysis code for the normative leader–follower coordination study.

## Included scripts

- `primary/01_rt_qa_outlier_screening.py`: screen trial-level RTs and write cleaned trials and exclusion summaries.
- `primary/02_trial_level_lmm_rt.py`: fit the trial-level mixed model.
- `primary/03_intra_individual_triallevel_welch.py`: summarize within-participant trial-level contrasts.
- `sensitivity/run_dyad_rt.py`: test the planned contrasts with dyads as independent units.

Each script accepts explicit input and a new output directory; see `--help` for arguments. Existing output directories are not overwritten. The supplementary ANOVA, equivalence, and Bayesian RT scripts are not included in this release.

## Inputs and environment

Participant-level RT inputs are available on request, subject to applicable consent, ethics, and institutional requirements. The RT screening script takes the trial-level input. Downstream scripts take the cleaned trials, participant-lock table, or four-cell table indicated by their command-line arguments. Use an approved input package and keep it outside Git history.

The deterministic RT scripts were run locally with Python 3.11.11, NumPy 1.26.4, pandas 2.2.2, SciPy 1.15.1, and statsmodels 0.13.5. On Windows, run the screening script with `python -X utf8` so its console output prints correctly.

## Commands

```powershell
python -X utf8 primary/01_rt_qa_outlier_screening.py --input-csv INPUT_TRIAL_CSV --output-dir NEW_QA_DIR
python primary/02_trial_level_lmm_rt.py --input-csv CLEANED_TRIAL_CSV --output-dir NEW_LMM_DIR
python primary/03_intra_individual_triallevel_welch.py --participant-lock-csv PARTICIPANT_LOCK_CSV --trials-csv CLEANED_TRIAL_CSV --output-dir NEW_WELCH_DIR
python sensitivity/run_dyad_rt.py --anova-csv FOUR_CELL_CSV --output-dir NEW_DYAD_DIR
```

The screening, within-participant, and dyad-level outputs were compared with saved study results. The mixed-model script is part of the paper's primary analysis; its input requirements and model specification are defined in the script.
