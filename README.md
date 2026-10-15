# Tetradic Dynamics Repository

Analysis code for the normative leader-follower sensorimotor coordination paper. The repository is organized by analysis so that each analysis and its sensitivity checks remain together.

## Current contents

- [`phasic-eeg/`](phasic-eeg/): corrected response-locked phasic EEG contrast verification, including the primary and RT-controlled sensitivity modes. The four files in this folder are unchanged copies of [the original phasic code repository](https://github.com/Carbonbygravity/tetradic-phasic-eeg-analysis) at commit `5fc24f8c2eeb58a2ef267c942f99b1adaee3b91b`.

Behavioural, tonic EEG, and information-decomposition analysis code is not included yet. No claim of reproducibility for those analyses is made by this release.

## Data access

Participant-level tensors and archived cluster maps are not stored on GitHub. Access to coded participant-level EEG tensors may be considered upon reasonable request, subject to applicable ethics and institutional data-sharing requirements. Until access is authorized, the phasic code can be inspected but cannot reproduce the saved maps.

The phasic runner expects any approved input package under `phasic-eeg/data/`. This directory is excluded from Git. Do not commit or redistribute participant data. See [`phasic-eeg/README.md`](phasic-eeg/README.md) for dependencies, commands, and the limits of the verification code.

## Scope

This repository is for the normative paper only. It does not contain code or data from the separate clinical-dyad analysis.
