# Tetradic Dynamics

Analysis code for a two-phase study of leader–follower sensorimotor coordination. The normative study characterized coordination in healthy dyads. The clinical study extended the tetradic framework to patient–control dyads and compared their coordination with the normative findings.

## Normative study

Code is organized by analysis, with sensitivity procedures alongside the corresponding primary analyses.

- [`behavioural/`](behavioural/): reaction-time screening, trial-level modelling, within-participant contrasts, and dyad-level sensitivity analysis.
- [`tonic-eeg/`](tonic-eeg/): source-power modelling, sensitivity analyses, and feature-level reporting.
- [`phasic-eeg/`](phasic-eeg/): response-period-onset-aligned source-power cluster analyses, sensitivity analyses, and saved-map verification.
- [`integrated-information-decomposition/`](integrated-information-decomposition/): revised normative ΦID estimation, inference, controls, and sensitivity analyses.

## Clinical study

The [clinical study code and reproduction instructions](clinical-dyads/README.md) cover behavioural, tonic EEG, phasic EEG and ΦID analyses, sensitivity checks, and clinical associations. Two companion CSVs provide the complete clinical association results:

- [Main clinical associations](clinical-dyads/results/Complete_main_associations_309.csv) — 309 tests, including the behavioural, tonic and main ΦID findings reported in Supplementary Tables ST19a–b.
- [Response-period × condition interaction associations](clinical-dyads/results/Complete_genuine_interaction_associations_35.csv) — 35 tests in the separate interaction family, including the findings reported in Supplementary Table ST20.

## Data access

Each analysis folder describes its required inputs and dependencies. Participant-level EEG analysis inputs and behavioural response-time data may be requested from the corresponding author, subject to applicable consent, ethics, and institutional data-sharing requirements.
