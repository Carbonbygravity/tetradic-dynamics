"""Equal-response-event-role tonic contrasts from unchanged separate fits.

No model refits or reconstruction. The cohorts comprise separate participants. Clinical HC/PT covariance is retained. Primary normal-Wald
inference only, not a robustness validation. CSV files are analysis outputs.
"""
from pathlib import Path
import os
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(os.environ['TETRADIC_TONIC_ROOT']).resolve()
NORM = ROOT / '03_Results/Normative_Source_Repair_Run_01/Shared_Inverse_Two_Branch_Run_01/06_Models/Nested_Intercept_Models_80/complete_epochs'
CLIN = ROOT / '03_Results/Clinical_Combined_Tonic_Model_Run_01/03_Model_Fits_83/primary'
GROUPS = ['normative_HC', 'mixed_dyad_HC', 'PT']
BANDS = ['theta','low_alpha','high_alpha','low_beta','high_beta','gamma']
PAIRS = [('mixed_dyad_HC','normative_HC'),('PT','normative_HC'),('PT','mixed_dyad_HC')]
NNAMES = ['Intercept','role_code','cond_code','role_code:cond_code','rt_z']
CNAMES = ['Intercept','group_code','role_code','cond_code','group:role','group:condition','role:condition','group:role:condition','rt_z']

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dot(a,b):
    assert len(a)==len(b)
    return sum(x*y for x,y in zip(a,b))

def bilinear(a,c,b):
    assert len(a)==len(b)==len(c) and all(len(row)==len(a) for row in c)
    return sum(a[i]*c[i][j]*b[j] for i in range(len(a)) for j in range(len(b)))

def bh(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i])
    q=[0.]*len(order); current=1.
    for rank in range(len(order),0,-1):
        idx=order[rank-1];current=min(current,pvalues[idx]*len(order)/rank);q[idx]=current
    return q

def result(estimate,variance,**fields):
    assert math.isfinite(estimate) and math.isfinite(variance) and variance>0
    se=math.sqrt(variance);z=estimate/se
    return dict(**fields,estimate=estimate,se=se,ci95_lo=estimate-1.959963984540054*se,
                ci95_hi=estimate+1.959963984540054*se,z=z,p_wald=math.erfc(abs(z)/math.sqrt(2)))

def correct(rows):
    q=bh([r['p_wald'] for r in rows])
    for r,val in zip(rows,q):r.update(q_bh=val,significant_q05=val<.05)

def write_csv(path,rows):
    assert rows
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main(out):
    assert not out.exists(),'Preserve prior comparison'
    source_files=sorted((NORM/'checkpoints').glob('*.json'))+sorted((CLIN/'checkpoints').glob('*.json'))
    assert len(source_files)==816
    extra=[NORM/'model_results.csv',CLIN/'fixed_effect_coefficients.csv',CLIN/'direct_contrast_results.csv']
    fingerprints={str(p):digest(p) for p in source_files+extra}
    specification={
        'status':'specification_frozen_before_new_contrast_calculation',
        'datasets':{'normative':'stage80 complete_epochs, 23364 observations, 30 participants, 15 dyads',
                    'clinical':'stage83 primary, 24234 observations, 30 participants, 15 dyads'},
        'cohort_independence':'The cohorts comprise separate participants',
        'contrast':'Mirroring-minus-Matching, equal average across Leader/Follower RESPONSE-EVENT roles',
        'response_scale':'difference in fitted mean log1p(dSPM-derived mean-ROI Welch band-bin power), NOT percent raw power or calibrated absolute cortical power',
        'normative_contrast_vector':[0,0,-1,0,0],
        'clinical_HC_contrast_vector':[0,0,0,-1,0,.5,0,0,0],
        'clinical_PT_contrast_vector':[0,0,0,-1,0,-.5,0,0,0],
        'direct_pairs':PAIRS,
        'independent_cohort_variance':'Var(delta_clinical - delta_normative)=Var(delta_clinical)+Var(delta_normative)',
        'paired_clinical_variance':'Var(delta_PT-delta_HC)=(v_PT-v_HC) Cov(beta_clinical) (v_PT-v_HC); includes HC/PT covariance',
        'inference':'Approximate two-sided normal Wald tests and pointwise 95% CIs from existing coefficient covariance; not cluster-jackknife inference',
        'multiplicity':{'within_context':'one BH family of 3 contexts x 408 features =1224 tests',
                        'direct_difference':'one SEPARATE BH family of 3 pairs x408 features =1224 tests'},
        'timing':'Post-hoc comparisons; not claimed preregistered; rules fixed before extracting new results',
        'clinical_robustness':'Previously documented influential dyads and heavy tails remain; no claim that primary findings are robustness-validated',
        'source_scale_caveat':'Cohort-specific covariance/inverse noise normalization remains; within-condition contrasts do not automatically eliminate it',
        'no_changes':['source inputs','saved fits','historical replication','combined model','manuscript','figures'],
        'sources_sha256':fingerprints}
    out.mkdir(parents=True)
    (out/'frozen_comparison_specification.json').write_text(json.dumps(specification,indent=2),encoding='utf-8')
    within=[];between=[];maps=[];max_pair_variance_error=0.;max_existing_direct_error=0.
    with (CLIN/'fixed_effect_coefficients.csv').open(newline='',encoding='utf-8-sig') as stream:
        names={}
        for row in csv.DictReader(stream):names.setdefault(row['feature'],[]).append(row['param'])
    assert len(names)==408 and all(v==CNAMES for v in names.values())
    with (CLIN/'direct_contrast_results.csv').open(newline='',encoding='utf-8-sig') as stream:
        old={(r['feature'],r['contrast']):r for r in csv.DictReader(stream)}
    nv=[0,0,-1,0,0];hv=[0,0,0,-1,0,.5,0,0,0];pv=[0,0,0,-1,0,-.5,0,0,0]
    for idx in range(408):
        n=load(NORM/'checkpoints'/f'{idx:03d}.json');c=load(CLIN/'checkpoints'/f'{idx:03d}.json')
        assert n['feature']==c['feature']
        feature=n['feature'];roi,band=feature.split(' | ')
        assert band in BANDS
        nf=n['full'];cf=c['fit'];assert nf['valid'] and cf['valid']
        nb,nc=nf['coef'],nf['cov_beta'];cb,cc=cf['coef'],cf['cov_beta']
        assert len(nb)==5 and len(cb)==9
        assert all(math.isfinite(x) for x in nb+cb)
        assert abs(nf['se'][2]**2-nc[2][2])<1e-12
        est={'normative_HC':dot(nv,nb),'mixed_dyad_HC':dot(hv,cb),'PT':dot(pv,cb)}
        var={'normative_HC':bilinear(nv,nc,nv),'mixed_dyad_HC':bilinear(hv,cc,hv),'PT':bilinear(pv,cc,pv)}
        clinical_cov=bilinear(hv,cc,pv)
        for group in GROUPS:
            within.append(result(est[group],var[group],feature=feature,roi=roi,band=band,context=group,
                                 matching_minus_mirroring=-est[group],higher_condition='Mirroring' if est[group]>0 else 'Matching'))
        for a,b in PAIRS:
            vdiff=[p-h for p,h in zip(pv,hv)]
            variance=bilinear(vdiff,cc,vdiff) if b=='mixed_dyad_HC' else var[a]+var[b]
            if b=='mixed_dyad_HC':
                error=abs(variance-(var['PT']+var['mixed_dyad_HC']-2*clinical_cov))
                max_pair_variance_error=max(error,max_pair_variance_error);assert error<1e-12
                row=old[feature,'PT-minus-HC: averaged-role Mirroring-minus-Matching']
                error=max(abs(est[a]-est[b]-float(row['estimate'])),abs(math.sqrt(variance)-float(row['se'])))
                max_existing_direct_error=max(error,max_existing_direct_error);assert error<1e-12
            between.append(result(est[a]-est[b],variance,feature=feature,roi=roi,band=band,
                                  comparison=f'{a}-minus-{b}',context_a=a,context_b=b,
                                  delta_a=est[a],delta_b=est[b],delta_a_variance=var[a],delta_b_variance=var[b],
                                  cross_context_covariance=clinical_cov if b=='mixed_dyad_HC' else 0.,
                                  estimate_definition='delta_a-minus-delta_b; each delta=Mirroring-minus-Matching'))
    assert len(within)==len(between)==1224
    correct(within);correct(between)
    wi={(r['feature'],r['context']):r for r in within};bi={(r['feature'],r['comparison']):r for r in between}
    for idx in range(408):
        feature=load(NORM/'checkpoints'/f'{idx:03d}.json')['feature'];roi,band=feature.split(' | ')
        row=dict(feature=feature,roi=roi,band=band)
        for group in GROUPS:
            r=wi[feature,group]
            row.update({f'{group}_delta':r['estimate'],f'{group}_q':r['q_bh'],f'{group}_pattern':r['higher_condition'] if r['significant_q05'] else 'Not FDR significant'})
        for a,b in PAIRS:
            name=f'{a}-minus-{b}';r=bi[feature,name]
            row.update({f'{name}_difference':r['estimate'],f'{name}_q':r['q_bh'],f'{name}_difference_significant':r['significant_q05']})
        row['same_point_direction_all_three']=len({wi[feature,g]['higher_condition'] for g in GROUPS})==1
        row['matching_significant_all_three']=all(wi[feature,g]['significant_q05'] and wi[feature,g]['estimate']<0 for g in GROUPS)
        row['mirroring_significant_all_three']=all(wi[feature,g]['significant_q05'] and wi[feature,g]['estimate']>0 for g in GROUPS)
        maps.append(row)
    counts=[];band_counts=[]
    for group in GROUPS:
        for band in ['ALL']+BANDS:
            rows=[r for r in within if r['context']==group and (band=='ALL' or r['band']==band)]
            row=dict(context=group,band=band,tested=len(rows),matching_higher=sum(r['significant_q05'] and r['estimate']<0 for r in rows),
                     mirroring_higher=sum(r['significant_q05'] and r['estimate']>0 for r in rows),significant_total=sum(r['significant_q05'] for r in rows))
            (counts if band=='ALL' else band_counts).append(row)
    direct_counts=[]
    for a,b in PAIRS:
        name=f'{a}-minus-{b}';rows=[r for r in between if r['comparison']==name]
        direct_counts.append(dict(comparison=name,tested=408,significant_difference=sum(r['significant_q05'] for r in rows),
                                  delta_a_greater=sum(r['significant_q05'] and r['estimate']>0 for r in rows),delta_a_lower=sum(r['significant_q05'] and r['estimate']<0 for r in rows)))
    for filename,rows in [('within_context_condition_effects.csv',within),('direct_condition_differences.csv',between),('ROI_band_comparison_matrix.csv',maps),
                          ('within_context_counts.csv',counts),('within_context_band_counts.csv',band_counts),('direct_difference_counts.csv',direct_counts)]:
        write_csv(out/filename,rows)
    assert all(digest(Path(p))==h for p,h in fingerprints.items()),'Source changed during extraction'
    summary=dict(status='separate_fit_primary_comparison_complete',models_refitted=0,reconstructions=0,source_hashes_unchanged=True,
                 independence_user_confirmed=True,within_counts=counts,direct_counts=direct_counts,
                 matching_significant_all_three=sum(r['matching_significant_all_three'] for r in maps),
                 mirroring_significant_all_three=sum(r['mirroring_significant_all_three'] for r in maps),
                 same_point_direction_all_three=sum(r['same_point_direction_all_three'] for r in maps),
                 max_clinical_pair_variance_identity_error=max_pair_variance_error,max_existing_clinical_contrast_error=max_existing_direct_error,
                 primary_only_not_robustness_validated=True)
    (out/'comparison_complete.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=['# Separate-fit primary tonic condition comparison — step 90','',
            'Updated normative complete-epoch fit versus clinical primary fit. Entirely separate cohort participants as reported in the study. No reconstruction, refitting, exclusions, manuscript changes or archived-result replacement.',
            '', 'Each condition effect is Mirroring minus Matching, equally averaged across response-event roles. Negative means higher in Matching; positive means higher in Mirroring. Role remains in both models. The outcome is log1p of dSPM-derived power, not a raw-power percentage.',
            '', '## Within-context condition effects','', '| Context | Matching higher | Mirroring higher | Total FDR significant |','| --- | ---: | ---: | ---: |']
    for r in counts:report.append(f"| {r['context']} | {r['matching_higher']} | {r['mirroring_higher']} | {r['significant_total']} |")
    report.extend(['','One BH family of 1,224 within-context tests. These counts are not the historical 279 discoveries, not combined-model counts, and not a test of between-context differences.',
                   '', '## Direct differences in condition effects','', '| Comparison | FDR significant differences | Delta A greater | Delta A lower |','| --- | ---: | ---: | ---: |'])
    for r in direct_counts:report.append(f"| {r['comparison']} | {r['significant_difference']} | {r['delta_a_greater']} | {r['delta_a_lower']} |")
    report.extend(['','A separate BH family of 1,224 direct tests. Delta is Mirroring-minus-Matching; greater/lower refers to that signed condition contrast, not overall power.',
                   '', 'Between-cohort variance is the sum of the two separately estimated contrast variances. Clinical PT-minus-HC variance retains their fitted covariance and agrees with the original saved clinical contrast.',
                   '', '## Interpretation limits','',
                   'These are conventional primary Gaussian normal-Wald results. Previous influential-dyad and heavy-tail concerns remain. Significance does not establish robustness; absence of a difference does not establish equivalence or preservation. Cohort-specific dSPM normalization remains, so effects concern this derived feature scale, not calibrated absolute cortical power.',
                   '', 'Clinical HC/PT retain shared variance components within the clinical fit. Only normative-versus-clinical variance pooling has been avoided. Pointwise 95% CIs are not simultaneous intervals.',
                   '', '## Verification','',
                   'All 408 features match across checkpoint files. Both saved fits were accepted for every feature. All 816 checkpoint hashes and source table hashes remained unchanged. Clinical pair covariance identities and saved contrast estimates/SEs reproduced within 1e-12.',
                   '', 'Detailed values and differences are in ROI_band_comparison_matrix.csv, within_context_condition_effects.csv and direct_condition_differences.csv. Sources, contrast vectors and correction rules are recorded in frozen_comparison_specification.json.'])
    (out/'comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
