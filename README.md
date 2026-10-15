# Tetradic Dynamics Repository

Analysis code for the normative leader–follower sensorimotor coordination paper. Code is organized by analysis, with sensitivity checks alongside the corresponding primary analysis.

## Available code

- [`phasic-eeg/`](phasic-eeg/): code to verify the response-locked phasic EEG contrast maps and run the primary and RT-controlled cluster analyses.

The `behavioural/`, `tonic-eeg/`, and `information-decomposition/` folders are currently placeholders; they contain no analysis code.

## Data access

Derived participant-level EEG tensors and saved cluster maps are not hosted on GitHub. Access may be requested from the study team, subject to applicable ethics and institutional requirements. These inputs are required to verify or rerun the maps and belong under `phasic-eeg/data/`, which Git excludes.

See [`phasic-eeg/README.md`](phasic-eeg/README.md) for dependencies, commands, and the limits of the verification code.

## Scope

This repository covers the normative paper, not the separate clinical-dyad study.
