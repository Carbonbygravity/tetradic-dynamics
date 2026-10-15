# Tetradic Dynamics Repository

Analysis code for the normative leader–follower sensorimotor coordination paper. The repository is organized by analysis, with sensitivity checks alongside the corresponding primary analysis.

## Available code

- [`behavioural/`](behavioural/): reaction-time screening, the trial-level mixed model, within-participant contrasts, and the dyad-level sensitivity check.
- [`tonic-eeg/`](tonic-eeg/): the RT-adjusted tonic power model, sensitivity checks, and the feature-level results table.
- [`phasic-eeg/`](phasic-eeg/): response-locked cluster analyses, their sensitivity checks, and the saved-map verification runner.
- [`integrated-information-decomposition/`](integrated-information-decomposition/): integrated information decomposition (ΦID) analyses and their sensitivity checks.

Each folder describes its inputs, setup, and commands. The code covers the analyses currently included here; it is not a substitute for the restricted participant-level inputs.

## Data access

Participant-level EEG tensors, behavioural response-time data, and saved cluster maps are not hosted on GitHub. Access may be requested from the study team, subject to applicable consent, ethics, and institutional data-sharing requirements. Store approved inputs outside Git history. The repository's `.gitignore` excludes common participant-data and generated-output formats.

## Scope

This repository concerns the normative paper only. It does not contain code or data from the separate clinical-dyad study.
