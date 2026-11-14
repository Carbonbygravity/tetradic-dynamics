# Clinical comparison

The `clinical-dyads/` directory contains the analysis code for the patient–control study. The behavioural model uses measured-only normative trials. The clinical association workflow reports the main analyses and the separate response-period × condition interaction. The scripts use prepared inputs for analyses including the participant-level phasic comparisons and the theta sensitivity analysis omitting one dyad. Reproduction from raw data also requires the upstream preprocessing steps.

The ΦID code includes the tile calculation and functions for contrast construction, surrogate generation, embedding and direct cohort comparison. The bundled phyid backend retains its BSD 3-Clause licence.

See WORKFLOWS.md for covariance sensitivity, ΦID selection and controls, and the scripts behind Supplementary Tables ST12–ST16.
