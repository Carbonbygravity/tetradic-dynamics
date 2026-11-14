# Reproduction from prepared inputs

## Inputs by workflow

| Workflow | Inputs required beyond the public release |
|---|---|
| Behavioural response times | Measured normative trials and screened clinical trials, including the participant and dyad grouping variables. |
| Tonic EEG and covariance sensitivity | Prepared event-linked EEG features and model inputs. The covariance sensitivity also requires the corresponding EEG, forward and inverse model inputs, parcellation and upstream fits or checkpoints for the relevant stages. |
| Phasic EEG, paired contrasts and theta follow-ups | Participant-level power tensors, mappings and adjacency inputs. Historical comparisons additionally require the saved participant-run specification and reference records described below. |
| ΦID production and matched per-map inference | Paired clinical HDF5 inputs and event-to-actor mappings for production. Assembly uses clinical production tiles and completion records. The historical matched per-map preflight also reads an observed normative dyad-level contrast array. |
| Main ΦID selection and surrogate controls | Clinical overlap results, the clinical manifest, paired participant inputs, production tiles and verification records. Tier assignment uses observed dyad estimates and surrogate checkpoints. Clinical targets are selected from clinical results. |
| Separate ΦID interaction selection and controls | Clinical interaction clusters, their verification records and the clinical inputs for the surrogate calculations. The archived selection and finalization checks also hash the private normative interaction tier file, which contains dyad-level values. |
| ΦID sensitivity and supplementary tables | Freezing targets requires completed clinical main and interaction tiers, specifications, receipts and the input manifest. Numerical follow-ups use the corresponding dyad inputs. The ST12–ST16 table builder instead reads saved analysis results and writes separate main and interaction tables. |
| Direct between-cohort ΦID comparisons | Prepared cohort arrays and frozen dyad draws. The public bootstrap function evaluates one map block. Whole-study inference requires the corresponding complete execution records. |
| Clinical associations | The clinical score workbook, frozen association inputs and behavioural components. Main and interaction tests are adjusted separately. |

## Normative references

From the repository root, point the clinical drivers to the bundled references:

```sh
export TETRADIC_NORMATIVE_REFERENCE_ROOT="$PWD/clinical-dyads/normative-references"
```

This directory supplies the eight references hashed by the main control launcher and the normative direction-summary source compared by its preflight. Sensitivity preparation reads the two normative sensitivity scripts as source references. Matched per-map setup reads six settings from the public specification and uses the bundled map ordering and permutation signs.

For the historical matched per-map preflight or the separate interaction preservation check, use an authorized private reference tree with the same directory layout and the additional files described in [the workflow guide](WORKFLOWS.md). Public specifications and path-adjusted scripts have their own file hashes. Historical receipts refer to archived originals. See the workflow guide before reusing checkpoints or preparing new run records.

## Commands from prepared inputs

Run from the repository root, in the environment for the relevant workflow. These shell examples assume fresh work directories and authorized prepared inputs as described in INPUTS.md. Set the behavioural compiler and dynamic library paths for the local installation. Prepare the matching PyTensor overlay or establish its equivalence before numerical comparison.

```sh
python clinical-dyads/behaviour/11_bayesian_combined_cohort.py --hchc-input inputs/normative_measured_rt.csv --hcsz-input inputs/clinical_screened_rt.csv --output-dir work/rt --qc-dir work/rt_qc --draws 2000 --tune 2000 --chains 4 --cores 4 --target-accept .99 --max-treedepth 12 --seed 20260922 --ppc-thin 20
python clinical-dyads/tonic/83_fit_clinical_combined_tonic_models.py --inputs work/tonic/01_Event_Linked_Inputs_82 --output work/tonic/03_Model_Fits_83
python clinical-dyads/tonic/85_clinical_dyad_jackknife_robustness.py --model-run work/tonic --output work/tonic_robustness
export TETRADIC_PHASIC_ROOT="$PWD/work/phasic"
for graph in anatomical historical; do
  python clinical-dyads/phasic/06_anatomical_and_paired_clusters.py --stage pilot --mode paired --adjacency "$graph"
  python clinical-dyads/phasic/06_anatomical_and_paired_clusters.py --stage full --mode paired --adjacency "$graph"
done
export TETRADIC_FIGURE_REFERENCE="$PWD/inputs/historical_panel_manifest.json"
export TETRADIC_HISTORICAL_NOTEBOOK="$PWD/inputs/historical_clustering.ipynb"
for stage in prepare pilot full; do
  python clinical-dyads/phasic/11_historical_participant_rerun.py --stage "$stage"
done
export TETRADIC_THETA_OUTPUT="$PWD/work/theta_omit"
python clinical-dyads/phasic/theta_omit_dyad1.py
export TETRADIC_FROZEN_INPUTS="$PWD/inputs/association_preflight"
export TETRADIC_COMPONENT_INPUTS="$PWD/inputs/behavioural_components"
export TETRADIC_SCORE_WORKBOOK="$PWD/inputs/clinical_scores.xlsx"
export TETRADIC_ASSOC_OUTPUT="$PWD/work/associations_new"
python clinical-dyads/associations/run_clinical_associations.py
```

Compare population estimates and planned contrasts against the measured-only behavioural reference. For paired phasic tests, use PT minus the healthy partner within each dyad. For associations compare by test_id: 305 original main tests, four component tests added to the 309-test adjustment, and 35 interaction tests adjusted separately. Outputs include private scores and row-level diagnostics; keep the work directory private.

For a single ΦID tile, open the two authorized HDF5 inputs for a dyad and use the verified block-to-actor mapping. For one ROI, the following Python call returns two geometries × four cells × six bands × 1,999 centres × four persistence atoms. The ROI index refers to the validated order in both files.

```python
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path("clinical-dyads").resolve()))
from phiid.clinical_tile import compute_tile
from phiid.tetradic_phiid.core import verify_backend

verify_backend()
values, metadata = compute_tile(p1_h5, p2_h5, roi_index, np.arange(1, 2000), actor_mapping)
```

Use the functions in main_contrasts and interaction for their respective contrasts. The lag and history functions expose the observation construction and historical scoring. The cohort_bootstrap.calculate function takes separately centred cohort arrays and frozen dyad draws for one map block; it does not construct a corrected null distribution across all blocks. Use the corresponding complete execution records for whole-study inference.

The participant phasic workflow also requires the trusted historical log-transformed tensor, its cluster results, panel manifest and notebook under the layout expected by the preparation stage. Point the dyad-omission follow-up at the verified inputs and adjacencies. The three-group descriptive script (13_theta_group_comparison.py) requires the original pinned participant-run specification and the inputs at its recorded locations. A freshly generated portable specification has a different hash. Use an isolated copy of the historical tree for that script. Portability across locations has not been demonstrated.
