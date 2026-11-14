"""One bounded clinical dyad-cluster jackknife; no input or source changes.

Fixed primary and existing sensitivity inputs, unchanged Gaussian ML estimator.
Jackknife uncertainty and influence are diagnostics, not a robust location model
or a claim of exact inference with 15/12 independent dyads. No normative/joint fits.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import argparse, contextlib, datetime, hashlib, importlib.util, json, time, traceback
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_module('clinical83', Path(__file__).with_name('83_fit_clinical_combined_tonic_models.py'))
engine_mod = mod.engine_module
write = engine_mod.write
BRANCHES = ['primary', 'sensitivity_omit_dyads_1_4_14']
PILOT_FEATURES = [0, 68, 136, 340]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correction(frame, family):
    frame['p_jackknife_fdr'] = np.nan
    for _, ix in frame.groupby(family, sort=False).groups.items():
        p = frame.loc[ix, 'p_jackknife']
        q = multipletests(p.fillna(1), method='fdr_bh')[1]
        q[p.isna()] = np.nan
        frame.loc[ix, 'p_jackknife_fdr'] = q
    return frame


def interval_row(beta, covariance, vector, df, usable):
    estimate = float(vector @ beta)
    se = float(np.sqrt(max(0., vector @ covariance @ vector))) if usable else np.nan
    valid = bool(usable and np.isfinite(se) and se > 0)
    critical = float(student_t.ppf(.975, df))
    return dict(estimate=estimate, jackknife_se=se, df=df,
                ci_lo_jackknife=estimate-critical*se,
                ci_hi_jackknife=estimate+critical*se,
                p_jackknife=float(2*student_t.sf(abs(estimate/se), df)) if valid else np.nan,
                jackknife_valid=valid)


def run(args):
    root, out = args.model_run, args.output
    assert not out.exists(), 'Preserve any existing robustness run'
    inputs = root/'01_Event_Linked_Inputs_82'
    models = root/'03_Model_Fits_83'
    complete = json.loads((models/'model_run_complete.json').read_text())
    assert complete['status'] == 'clinical_combined_model_fits_complete'
    frozen = json.loads((models/'frozen_model_and_contrast_specification.json').read_text())
    assert sha(Path(__file__).with_name('80_fit_shared_inverse_normative_models.py')) == frozen['engine_sha256']
    fingerprints = {str(models/'frozen_model_and_contrast_specification.json'): sha(models/'frozen_model_and_contrast_specification.json')}
    for branch in BRANCHES:
        for name, item in [('model_metadata.csv', 'input_metadata_sha256'), ('model_power.npz', 'input_power_sha256')]:
            path = inputs/branch/name
            assert sha(path) == frozen[branch][item], 'Frozen input changed'
            fingerprints[str(path)] = sha(path)
    out.mkdir(parents=True)
    specification = dict(
        scope='Clinical primary and existing diagnostic sensitivity only; all 408 features',
        pilot=args.pilot, method='Delete-one-whole-dyad cluster jackknife of unchanged nested-intercept Gaussian ML estimator',
        estimator='Original full-sample coefficients retained, not jackknife-bias-corrected',
        covariance='(G-1)/G * sum((beta_minus_g - mean_minus) outer (beta_minus_g - mean_minus))',
        inference='Approximate two-sided Student t, df=G-1; not exact small-sample coverage',
        primary_dyads=15, sensitivity_dyads=12, features=408,
        frozen_RT_scaling=True, source_covariance_inverse_recomputed=False,
        exclusion='Diagnostic leave-one-dyad-out fits only; no permanent omissions added',
        multiplicity=frozen['multiplicity'], same_saved_direct_vectors=True,
        stopping_rule='One fixed run; invalid delete-one fits leave feature inference NA, no fallback/optimizer sweep',
        original_fingerprints=fingerprints,
        engine_sha256=frozen['engine_sha256'], script_sha256=sha(Path(__file__)),
        references=['https://journal.r-project.org/articles/RJ-2012-011/',
                    'https://pmc.ncbi.nlm.nih.gov/articles/PMC12041842/'],
        limits=['Does not make residuals Gaussian or robustify point estimates.',
                'Assumes independence between dyads; only 15/12 dyads.',
                'Source inverse is held fixed; reconstruction uncertainty is not propagated.',
                'Existing sensitivity changes sample AND covariance/inverse normalization.',
                'No new clinical-versus-normative conclusion is validated by this clinical-only check.',
                'Boundary variance estimates make jackknife inference approximate.'])
    write(out/'frozen_robustness_specification.json', specification)
    start = time.monotonic()
    def status(**state):
        state.update(worker_pid=os.getpid(), worker_running=state.get('worker_running', True),
                     updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-start)
        write(out/'run_status.json', state)
        print(json.dumps(state), flush=True)
    summaries = {}
    try:
        for branch in BRANCHES:
            meta = pd.read_csv(inputs/branch/'model_metadata.csv', low_memory=False)
            X, names, vector, groups, omnibus = mod.design(meta, False)
            assert names == frozen[branch]['fixed_columns']
            assert all(len(ix) == 1 for ix in omnibus.values())
            dyads = sorted(meta.dyad.unique().tolist())
            G = len(dyads)
            assert G == (15 if branch == 'primary' else 12)
            with np.load(inputs/branch/'model_power.npz', allow_pickle=False) as saved:
                Y = saved['raw_power'].astype(np.float64)
                features = saved['features'].tolist()
            assert Y.shape == (len(meta), 408)
            contrasts = frozen[branch]['contrasts']
            folder = out/branch
            (folder/'checkpoints').mkdir(parents=True)
            masks = [meta.dyad.ne(g).to_numpy() for g in dyads]
            submeta = [meta.loc[m].reset_index(drop=True) for m in masks]
            subX = [X[m] for m in masks]
            for m in submeta:
                assert m.groupby('participant_id').cell.nunique().eq(4).all()
            tests, direct, influence, residual_energy = [], [], [], []
            selected = PILOT_FEATURES if args.pilot else range(408)
            accepted_replicates = accepted_features = 0
            for count, idx in enumerate(selected, 1):
                feature = features[idx]
                path = models/branch/'checkpoints'/f'{idx:03d}.json'
                fingerprints[str(path)] = sha(path)
                base = json.loads(path.read_text())
                assert base['feature'] == feature and base['fit']['valid']
                beta = np.array(base['fit']['coef'])
                model_cov = np.array(base['fit']['cov_beta'])
                y = np.log1p(Y[:, idx]); unit = float(y.std()); mean = float(y.mean())
                standardized = (y-mean)/unit
                engine = engine_mod.NestedProfile(X, standardized, meta)
                d = engine.evaluate(np.log1p([base['fit']['relative_participant_variance'],
                                              base['fit']['relative_dyad_variance']]), True)
                check_beta = d['beta']*unit; check_beta[0] += mean
                assert np.max(np.abs(check_beta-beta))/unit < 1e-7, 'Full-sample coefficients do not reproduce'
                if args.pilot:
                    refit, _ = engine.fit(unit, mean, verify=True)
                    assert refit['valid'] and np.max(np.abs(np.array(refit['coef'])-beta))/unit < 1e-6
                # Conditional residual energy locates influence, not artifact diagnoses.
                sums = engine.zy-np.einsum('dik,k->di', engine.zx, d['beta'])
                intercepts = np.einsum('dij,dj->di', d['W'], sums)
                fitted = X@d['beta']
                for positions, values in zip(engine.slices, intercepts):
                    for rows, value in zip(positions, values): fitted[rows] += value
                residual = standardized-fitted
                energy = residual@residual
                extreme = np.abs(residual) >= np.quantile(np.abs(residual), .99)
                for g in dyads:
                    ix = meta.dyad.eq(g).to_numpy()
                    residual_energy.append(dict(dataset=branch, feature=feature, dyad=g,
                        observations=int(ix.sum()), observation_share=float(ix.mean()),
                        residual_energy_share=float(residual[ix]@residual[ix]/energy),
                        share_top_1pct_residuals=float(extreme[ix].sum()/extreme.sum())))
                replicas = []
                for j, g in enumerate(dyads):
                    sub = engine_mod.NestedProfile(subX[j], standardized[masks[j]], submeta[j])
                    fit, _ = sub.fit(unit, mean, verify=args.pilot and j == 0)
                    replicas.append(dict(omitted_dyad=g, **fit))
                    accepted_replicates += int(fit['valid'])
                usable = all(r['valid'] for r in replicas)
                accepted_features += int(usable)
                B = np.array([r['coef'] for r in replicas])
                center = B.mean(axis=0)
                covariance = (G-1)/G*((B-center).T@(B-center)) if usable else np.full(model_cov.shape, np.nan)
                # Independent algebra check of jackknife diagonal.
                if usable:
                    np.testing.assert_allclose(np.diag(covariance), (G-1)*B.var(axis=0, ddof=0), rtol=1e-10, atol=1e-25)
                for effect, positions in omnibus.items():
                    v = np.eye(len(beta))[positions[0]]
                    row = interval_row(beta, covariance, v, G-1, usable)
                    tests.append(dict(dataset=branch, feature=feature, effect=effect,
                        model_se=float(np.sqrt(v@model_cov@v)),
                        leave_one_out_min=float((B@v).min()), leave_one_out_max=float((B@v).max()), **row))
                for c in contrasts:
                    v = np.array(c['vector']); est = float(v@beta); vals = B@v
                    row = interval_row(beta, covariance, v, G-1, usable)
                    model_se = float(np.sqrt(v@model_cov@v))
                    flips = np.sign(vals) != np.sign(est)
                    max_j = int(np.argmax(np.abs(vals-est)))
                    direct.append(dict(dataset=branch, feature=feature, contrast=c['name'], family=c['family'],
                        model_se=model_se, se_ratio_jackknife_model=row['jackknife_se']/model_se,
                        leave_one_out_min=float(vals.min()), leave_one_out_max=float(vals.max()),
                        any_delete_one_direction_change=bool(flips.any()), direction_change_count=int(flips.sum()),
                        most_influential_dyad=dyads[max_j],
                        max_absolute_change=float(abs(vals[max_j]-est)), **row))
                    for j, g in enumerate(dyads):
                        influence.append(dict(dataset=branch, feature=feature, contrast=c['name'], family=c['family'],
                            omitted_dyad=g, original_estimate=est, delete_one_estimate=float(vals[j]),
                            coefficient_change=float(vals[j]-est), change_over_original_model_se=float((vals[j]-est)/model_se),
                            direction_changed=bool(flips[j]), fit_valid=bool(replicas[j]['valid'])))
                write(folder/'checkpoints'/f'{idx:03d}.json', dict(feature=feature,
                    original_beta=beta.tolist(), original_covariance=model_cov.tolist(),
                    jackknife_covariance=covariance.tolist(), valid=usable, replicates=replicas))
                if args.pilot or count % 17 == 0:
                    status(stage='jackknife_fitting', dataset=branch, completed_features=count,
                           total_features=len(selected), accepted_replicates=accepted_replicates,
                           expected_replicates=len(selected)*G)
            test_frame, direct_frame = pd.DataFrame(tests), pd.DataFrame(direct)
            if not args.pilot:
                assert len(test_frame) == 408*len(omnibus)
                for family, size in frozen[branch]['contrast_family_sizes'].items():
                    assert direct_frame.family.eq(family).sum() == size
                correction(test_frame, 'effect'); correction(direct_frame, 'family')
                original_tests = pd.read_csv(models/branch/'omnibus_wald_results.csv')
                original_direct = pd.read_csv(models/branch/'direct_contrast_results.csv')
                test_frame = test_frame.merge(original_tests[['feature','effect','p_wald_fdr']], on=['feature','effect'], validate='one_to_one')
                direct_frame = direct_frame.merge(original_direct[['feature','contrast','p_wald_fdr']], on=['feature','contrast'], validate='one_to_one')
            test_frame.to_csv(folder/'omnibus_jackknife_comparison.csv', index=False)
            direct_frame.to_csv(folder/'direct_contrast_jackknife_comparison.csv', index=False)
            influence_frame = pd.DataFrame(influence)
            influence_frame.to_csv(folder/'all_delete_one_contrast_influence.csv', index=False)
            residual_frame = pd.DataFrame(residual_energy)
            residual_frame.to_csv(folder/'dyad_residual_energy.csv', index=False)
            influence_summary = influence_frame.groupby(['omitted_dyad','contrast'], sort=False).agg(
                median_absolute_change_model_SE=('change_over_original_model_se', lambda s: float(np.median(np.abs(s)))),
                maximum_absolute_change_model_SE=('change_over_original_model_se', lambda s: float(np.max(np.abs(s)))),
                direction_changes=('direction_changed','sum')).reset_index()
            influence_summary.to_csv(folder/'dyad_influence_summary.csv', index=False)
            summary = dict(observations=len(meta), dyads=G, participants=meta.participant_id.nunique(),
                           features=len(selected), valid_jackknife_features=accepted_features,
                           expected_delete_one_fits=G*len(selected), valid_delete_one_fits=accepted_replicates)
            if not args.pilot:
                summary['effects'] = {effect: dict(model_FDR=int(s.p_wald_fdr.lt(.05).sum()),
                    jackknife_FDR=int(s.p_jackknife_fdr.lt(.05).sum()),
                    shared_FDR=int((s.p_wald_fdr.lt(.05)&s.p_jackknife_fdr.lt(.05)).sum()),
                    unresolved=int(s.p_jackknife.isna().sum()),
                    median_SE_ratio=float(np.nanmedian(s.jackknife_se/s.model_se)))
                    for effect,s in test_frame.groupby('effect', sort=False)}
                summary['direct'] = {name: dict(model_FDR=int(s.p_wald_fdr.lt(.05).sum()),
                    jackknife_FDR=int(s.p_jackknife_fdr.lt(.05).sum()),
                    shared_FDR=int((s.p_wald_fdr.lt(.05)&s.p_jackknife_fdr.lt(.05)).sum()),
                    original_FDR_direction_changes=int((s.p_wald_fdr.lt(.05)&s.any_delete_one_direction_change).sum()),
                    unresolved=int(s.p_jackknife.isna().sum()),
                    median_SE_ratio=float(np.nanmedian(s.se_ratio_jackknife_model)))
                    for name,s in direct_frame.groupby('contrast', sort=False)}
                summary['residual_energy'] = {str(g): dict(
                    observation_share=float(s.observation_share.iloc[0]),
                    median_energy_share=float(s.residual_energy_share.median()),
                    median_top_1pct_share=float(s.share_top_1pct_residuals.median()))
                    for g,s in residual_frame.groupby('dyad')}
            summaries[branch] = summary
            write(folder/'jackknife_complete.json', summary)
        for path, expected in fingerprints.items():
            assert sha(Path(path)) == expected, 'Original/frozen input changed during run'
        write(out/'original_input_hashes_verified.json', fingerprints)
        result = dict(status='pilot_complete' if args.pilot else 'bounded_clinical_robustness_complete',
                      datasets=summaries, original_files_changed=False, manuscript_changed=False,
                      worker_running=False, elapsed_seconds=time.monotonic()-start,
                      completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      caveats=specification['limits'])
        write(out/('pilot_complete.json' if args.pilot else 'robustness_complete.json'), result)
        status(stage=result['status'], worker_running=False)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as e:
        status(stage='failed_attention_required', worker_running=False, error=repr(e))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open('x', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log): run(args)
    else:
        run(args)
