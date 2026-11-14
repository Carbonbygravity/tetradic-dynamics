# Reproducing the revised normative ΦID analysis

This code requires the 30 source-ROI HDF5 inputs listed in the private frozen
manifest; participant data and reported outputs are not included in the
repository. Check the environment and input hashes before computation. Use
new output locations and verify each stage's completion records before
starting a dependent stage. Some stages are computationally intensive.

## 1. Prepare the environment and inputs

From `integrated-information-decomposition/`, create the Conda environment
and install the recorded `phyid` revision as documented in `ENVIRONMENT.md`.
Run `check_environment.py`. Place the obtained inputs under a data root
as `normative/dyad<ID>_P<PARTNER>_roi.h5` for dyads 1–4 and 6–16, both
partners. Keep this data root outside the public repository.

Run `workflow/PhiID_Production_Preflight_2026-10-25/prepare_inputs.py` with
`--input-dir <DATA_ROOT>` and `--output-manifest` pointing to a new local
`workflow/PhiID_Production_Preflight_2026-10-25/input_manifest.json`.
Set `PRIVATE_INPUT_MANIFEST` to the path of the authorized original frozen manifest.
From `integrated-information-decomposition/`, run:

```sh
python workflow/PhiID_Production_Preflight_2026-10-25/verify_inputs.py \
  --runtime-manifest workflow/PhiID_Production_Preflight_2026-10-25/input_manifest.json \
  --frozen-manifest "$PRIVATE_INPUT_MANIFEST"
```

The verifier compares all 30 dyad–partner keys, relative filenames, byte sizes and
SHA-256 hashes. The public example illustrates the schema with one synthetic
entry. Keep both verification manifests in the private analysis environment.

## 2. Generate the primary tensor and temporal maps

Run `workflow/PhiID_Production_Preflight_2026-10-25/runner.py --run` with
`--input-dir <DATA_ROOT>`, `--input-manifest` pointing to the verified runtime
manifest, and `--output-dir` pointing to that stage's `production/` folder.
The expected full output is 15 dyads × 68 ROI tiles. Verify its saved
completion markers before proceeding.

Run `workflow/PhiID_Normative_Inference_2026-10-26/pipeline.py` in the order
`prepare`, `preflight`, `run`. This makes the original pooled condition and
pooled response-period maps plus the separately identified genuine
interaction. Then run `workflow/PhiID_Matched_PerMap_2026-10-26/runner.py`
in the same `prepare`, `preflight`, `run` order. Run that folder's
`verify_and_overlap.py` after completion. The matched stage supplies the two
within-condition contrasts and preserves the interaction as a separate
branch. Geometry overlap is identified before surrogate controls.

## 3. Generate and finalize controls

For the four main contrasts, use
`workflow/PhiID_Normative_Controls_2026-10-26/runner.py` for `prepare` and
`preflight`, then run its `trial_generation.py`. After that process completes,
run `finish_controls.py` for the fixed-parent-direction trial
summary, eligible time shifts and final geometry-specific tiers. **Do not
use the initial controls runner's historical upper-tail summary as the
reported result.** The finalizer's hash is recorded in
`direction_decision.json` and should match before execution.

Only after the four-contrast finalizer exits successfully, use
`workflow/PhiID_Normative_Interaction_Controls_2026-10-26/manage.py` with
`prepare`, `preflight`, then `run-ready`. This branch performs the separate
genuine-interaction controls. Do not run both control branches concurrently.
The saved finalization rules keep homologous and cross-hemispheric target
tiers distinct, even for overlapping pairings.

## 4. Generate sensitivity analyses separately

After both final tier files exist, set `PRIVATE_LAG_INVENTORY` to the path of
the original 19-target inventory. From a private working copy of
`integrated-information-decomposition/`, run:

```sh
python workflow/PhiID_Matched_Followup_Preflight_2026-10-27/prepare_frozen_targets.py \
  --frozen-targets "$PRIVATE_LAG_INVENTORY"
```

The script checks the original file hash and compares every field of the 16
main and three interaction Tier A/B records, including the dyad-level values.
It copies the verified private bytes to `frozen_targets.json` only if that
destination is absent. The public target inventory provides group-level records
for inspection.

Run that folder's
`check_baseline.py` before the one-dyad tau benchmark, which requires its
`tau_benchmark.json`. The tau benchmark's planned-stop/resume behavior is
part of its own verification; the subsequent tau sweep requires its completed
specification and verification files.

History preflight determines which targets are eligible for history-specific
controls. The history production stage, chronological task-half comparison
and signal-timescale diagnostics use their own saved specifications and
separate outputs. Their results do not replace the primary estimates or
tiers. Consult the scripts' preflight and completion records at each stage.

### Reusing historical checkpoints

The lag sweep includes the hash of the private `frozen_targets.json` handoff in
its run specification. The chronological task-half workflow reads and hashes
`workflow/PhiID_Normative_RoleSwap_Plan_2026-10-27/target_inventory.json`.
For historical checkpoint verification, use the original five-target file at
that path in a private working tree. The public copy omits per-dyad values and
has a different hash.

Both workflows bind checkpoints to their full specifications, including other
input and code hashes and recorded paths. Retain the archived specifications
and completion records. If the bindings differ, use a fresh output location for
a new run and preserve the historical checkpoints.

## Verification scope

The release checks compared a fresh pinned environment, all 30 available
input copies with recorded hashes and sizes, a five-centre production tensor
sample with saved values, a 19-target handoff, and four selected saved-control
targets spanning both signs and both control branches. These bounded checks
do not constitute a fresh end-to-end rerun or establish upstream source
validity or interpretive validity.
