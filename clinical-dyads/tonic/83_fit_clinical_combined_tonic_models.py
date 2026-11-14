"""Clinical and three-context Gaussian ML tonic fits with direct contrasts.

Imports unchanged verified step80 nested-intercept likelihood engine.
Freezes contrasts and multiplicity before viewing clinical model results.
No feature-dependent model simplification, added exclusions or manuscript edits.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse, contextlib, datetime, hashlib, importlib.util, json, time, traceback
import numpy as np
import pandas as pd
from scipy.stats import norm,chi2,skew,kurtosis
from statsmodels.stats.multitest import multipletests

ENGINE_PATH=Path(__file__).with_name('80_fit_shared_inverse_normative_models.py')
module_spec=importlib.util.spec_from_file_location('verified_step80',ENGINE_PATH)
engine_module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(engine_module)
NestedProfile=engine_module.NestedProfile;write=engine_module.write

MODELS=['primary','sensitivity_omit_dyads_1_4_14','combined_primary','combined_sensitivity_omit_dyads_1_4_14']

def design(meta,combined):
    r=meta.role_code.to_numpy();c=meta.cond_code.to_numpy();z=meta.rt_z.to_numpy();one=np.ones(len(meta))
    if combined:
        h=meta.actor_context.eq('mixed_dyad_HC').astype(float).to_numpy();p=meta.actor_context.eq('PT').astype(float).to_numpy()
        names=['Intercept','mixed_HC','PT','role_code','cond_code','role:condition','mixed_HC:role','PT:role','mixed_HC:condition','PT:condition','mixed_HC:role:condition','PT:role:condition','rt_z']
        X=np.column_stack([one,h,p,r,c,r*c,h*r,p*r,h*c,p*c,h*r*c,p*r*c,z])
        def vector(group,role,condition):
            hh=float(group=='mixed_dyad_HC');pp=float(group=='PT')
            return np.array([1,hh,pp,role,condition,role*condition,hh*role,pp*role,hh*condition,pp*condition,hh*role*condition,pp*role*condition,0])
        groups=['normative_HC','mixed_dyad_HC','PT']
        omnibus={'ActorContext':[1,2],'ActorContext:Role':[6,7],'ActorContext:Condition':[8,9],'ActorContext:Role:Condition':[10,11],
                  'Role':[3],'Condition':[4],'Role:Condition':[5],'RT':[12]}
    else:
        g=meta.group_code.to_numpy()
        names=['Intercept','group_code','role_code','cond_code','group:role','group:condition','role:condition','group:role:condition','rt_z']
        X=np.column_stack([one,g,r,c,g*r,g*c,r*c,g*r*c,z])
        def vector(group,role,condition):
            gg=.5 if group=='PT' else -.5
            return np.array([1,gg,role,condition,gg*role,gg*condition,role*condition,gg*role*condition,0])
        groups=['HC','PT']
        omnibus={'Group':[1],'Group:Role':[4],'Group:Condition':[5],'Group:Role:Condition':[7],'Role':[2],'Condition':[3],'Role:Condition':[6],'RT':[8]}
    assert np.linalg.matrix_rank(X)==X.shape[1]
    return X,names,vector,groups,omnibus

def define_contrasts(vector,groups):
    pairs=[(groups[1],groups[0]),(groups[2],groups[0]),(groups[2],groups[1])] if len(groups)==3 else [('PT','HC')]
    contrasts=[]
    def add(name,family,v):contrasts.append(dict(name=name,family=family,vector=v.tolist()))
    def cond(g,r):return vector(g,r,-.5)-vector(g,r,.5)
    def role(g,c):return vector(g,.5,c)-vector(g,-.5,c)
    for g in groups:
        for lab,r in [('Leader',.5),('Follower',-.5)]:add(f'{g}: {lab}: Mirroring-minus-Matching','within_context_condition',cond(g,r))
        for lab,c in [('Matching',.5),('Mirroring',-.5)]:add(f'{g}: {lab}: Leader-event-minus-Follower-event','within_context_role',role(g,c))
    for a,b in pairs:
        pair=f'{a}-minus-{b}'
        add(f'{pair}: averaged-role-condition','context_average',vector(a,0,0)-vector(b,0,0))
        add(f'{pair}: averaged-role Mirroring-minus-Matching','context_condition_average',cond(a,0)-cond(b,0))
        add(f'{pair}: averaged-condition Leader-event-minus-Follower-event','context_role_average',role(a,0)-role(b,0))
        add(f'{pair}: condition difference Leader-minus-Follower','context_threeway',cond(a,.5)-cond(b,.5)-cond(a,-.5)+cond(b,-.5))
        for lab,r in [('Leader',.5),('Follower',-.5)]:
            add(f'{pair}: {lab}: difference in Mirroring-minus-Matching','context_condition_by_role',cond(a,r)-cond(b,r))
            for condition,c in [('Matching',.5),('Mirroring',-.5)]:add(f'{pair}: {lab} {condition}','context_cells',vector(a,r,c)-vector(b,r,c))
    return contrasts

def correct(frame,pcol,familycol):
    frame[pcol+'_fdr']=np.nan
    for _,ids in frame.groupby(familycol,sort=False).groups.items():
        vals=frame.loc[ids,pcol];q=multipletests(vals.fillna(1),method='fdr_bh')[1];q[vals.isna()]=np.nan
        frame.loc[ids,pcol+'_fdr']=q
    return frame

def run(args):
    out=args.output;inputs=args.inputs
    report=json.loads((inputs/'input_validation_complete.json').read_text());assert report['status']=='clinical_combined_inputs_verified'
    assert not out.exists(),'Preserve previous statistical run';out.mkdir(parents=True)
    specification={**engine_module.SPEC,'scope':'clinical primary/sensitivity and combined primary/sensitivity; direct Wald contrasts and 2-df context omnibus tests',
        'primary_normative_sample':'complete_epochs; historical first480 comparison separately preserved',
        'clinical_formula':'log1p(power) ~ Group * Role * Condition + rt_z + (1|dyad) + (1|dyad:participant)',
        'combined_formula':'log1p(power) ~ ActorContext * Role * Condition + rt_z + (1|cohort:dyad) + (1|cohort:dyad:participant)',
        'RT_random_slope':False,'group_assignment':'recorded brain P1=HC/P2=PT, NOT response actor',
        'LRT':'Not used for new factorial inferences; direct linear Wald and appropriate-df omnibus Wald tests',
        'omnibus_interpretation':'Clinical effect coding gives averaged effects. Combined Role/Condition terms at normative reference; context tests compare reference coding coefficients at averaged interacting predictors.',
        'multiplicity':'Each omnibus effect has408-feature BH family. Direct contrast BH pools408features times all contrasts in named family, separately by dataset; no correction chosen after results.',
        'combined_contexts':['normative_HC','mixed_dyad_HC','PT'],'engine_sha256':hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest(),
        'input_validation_sha256':hashlib.sha256((inputs/'input_validation_complete.json').read_bytes()).hexdigest(),'pilot':args.pilot}
    for model in MODELS:
        meta=pd.read_csv(inputs/model/'model_metadata.csv',low_memory=False);X,names,v,groups,omnibus=design(meta,model.startswith('combined'))
        contrasts=define_contrasts(v,groups)
        specification[model]=dict(fixed_columns=names,groups=groups,omnibus=omnibus,contrasts=contrasts,
            input_metadata_sha256=hashlib.sha256((inputs/model/'model_metadata.csv').read_bytes()).hexdigest(),
            input_power_sha256=hashlib.sha256((inputs/model/'model_power.npz').read_bytes()).hexdigest(),
            contrast_family_sizes={family:408*sum(c['family']==family for c in contrasts) for family in {c['family'] for c in contrasts}})
    write(out/'frozen_model_and_contrast_specification.json',specification)
    start=time.monotonic();totals={}
    def status(**state):
        state.update(worker_pid=os.getpid(),worker_running=state.get('worker_running',True),updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        write(out/'run_status.json',state);print(json.dumps(state),flush=True)
    try:
        for model in MODELS:
            meta=pd.read_csv(inputs/model/'model_metadata.csv',low_memory=False);combined=model.startswith('combined');X,names,vector,groups,omnibus=design(meta,combined)
            contrasts=specification[model]['contrasts']
            with np.load(inputs/model/'model_power.npz',allow_pickle=False) as saved:Y=saved['raw_power'].astype(np.float64);features=saved['features'].tolist()
            assert Y.shape==(len(meta),408)
            folder=out/model;(folder/'checkpoints').mkdir(parents=True)
            tests=[];direct=[];cells=[];coefficients=[];diagnostics=[]
            selected=[0,68,136,340] if args.pilot else range(408)
            for idx in selected:
                y=np.log1p(Y[:,idx]);unit=float(y.std());mean=float(y.mean());assert unit>0
                engine=NestedProfile(X,(y-mean)/unit,meta);fit,d=engine.fit(unit,mean,verify=idx in [0,68,136,340])
                beta=np.array(fit['coef']);cov=np.array(fit['cov_beta']);feature=features[idx]
                for j,name in enumerate(names):coefficients.append(dict(dataset=model,feature=feature,param=name,coef=beta[j],se=fit['se'][j],valid=fit['valid']))
                for effect,positions in omnibus.items():
                    estimate=beta[positions];sub=cov[np.ix_(positions,positions)]
                    wald=float(estimate@np.linalg.solve(sub,estimate)) if fit['valid'] else np.nan
                    tests.append(dict(dataset=model,feature=feature,effect=effect,wald_chi2=wald,df=len(positions),p_wald=float(chi2.sf(wald,len(positions))),full_valid=fit['valid']))
                for contrast in contrasts:
                    v=np.array(contrast['vector']);estimate=float(v@beta);se=float(np.sqrt(v@cov@v)) if fit['valid'] else np.nan
                    direct.append(dict(dataset=model,feature=feature,contrast=contrast['name'],family=contrast['family'],
                        estimate=estimate if fit['valid'] else np.nan,se=se,ci_lo=estimate-1.96*se,ci_hi=estimate+1.96*se,
                        z=estimate/se,p_wald=float(2*norm.sf(abs(estimate/se))),full_valid=fit['valid']))
                for group in groups:
                    for role,r in [('Leader',.5),('Follower',-.5)]:
                        for condition,c in [('Matching',.5),('Mirroring',-.5)]:
                            v=vector(group,r,c);pred=float(v@beta);se=float(np.sqrt(v@cov@v)) if fit['valid'] else np.nan
                            cells.append(dict(dataset=model,feature=feature,context=group,role=role,condition=condition,
                                adjusted_log1p_power=pred if fit['valid'] else np.nan,se=se,ci_lo_log1p=pred-1.96*se,ci_hi_log1p=pred+1.96*se,
                                conditional_typical_power=np.expm1(pred) if fit['valid'] else np.nan,full_valid=fit['valid']))
                rsum=engine.zy-np.einsum('dik,k->di',engine.zx,d['beta']);intercepts=np.einsum('dij,dj->di',d['W'],rsum);fitted=X@d['beta']
                for positions,values in zip(engine.slices,intercepts):
                    for rows,val in zip(positions,values):fitted[rows]+=val
                residual=(y-mean)/unit-fitted
                context=meta.actor_context if combined else meta.group
                cellstats=[]
                for group in groups:
                    for cell in ['LMA','LMR','FMA','FMR']:
                        r=residual[context.eq(group)&meta.cell.eq(cell)]
                        cellstats.append(dict(context=group,cell=cell,n=len(r),sd=float(r.std(ddof=1)*unit),mean=float(r.mean()*unit)))
                diag=dict(dataset=model,feature=feature,valid=fit['valid'],projected_score=fit['normalized_projected_score'],
                    participant_variance=fit['participant_variance'],dyad_variance=fit['dyad_variance'],residual_variance=fit['residual_variance'],
                    participant_boundary=fit['participant_boundary'],dyad_boundary=fit['dyad_boundary'],
                    residual_skew=float(skew(residual)),residual_excess_kurtosis=float(kurtosis(residual)),
                    cell_sd_max_min=max(row['sd'] for row in cellstats)/min(row['sd'] for row in cellstats),cell_residuals=cellstats)
                diagnostics.append(diag);write(folder/'checkpoints'/f'{idx:03d}.json',dict(feature=feature,fit=fit,diagnostics=diag))
                if args.pilot or (idx+1)%34==0:status(stage='fitting',dataset=model,completed_features=len(diagnostics),total_features=len(selected))
            test_frame=pd.DataFrame(tests);contrast_frame=pd.DataFrame(direct)
            if not args.pilot:
                for effect in omnibus:assert test_frame.effect.eq(effect).sum()==408
                for family,size in specification[model]['contrast_family_sizes'].items():assert contrast_frame.family.eq(family).sum()==size
                correct(test_frame,'p_wald','effect');correct(contrast_frame,'p_wald','family')
            test_frame.to_csv(folder/'omnibus_wald_results.csv',index=False);contrast_frame.to_csv(folder/'direct_contrast_results.csv',index=False)
            pd.DataFrame(cells).to_csv(folder/'adjusted_cell_predictions.csv',index=False);pd.DataFrame(coefficients).to_csv(folder/'fixed_effect_coefficients.csv',index=False)
            pd.DataFrame([{k:v for k,v in row.items() if k!='cell_residuals'} for row in diagnostics]).to_csv(folder/'fit_residual_diagnostics.csv',index=False)
            pd.DataFrame([dict(dataset=model,feature=row['feature'],**cell) for row in diagnostics for cell in row['cell_residuals']]).to_csv(folder/'cell_residual_diagnostics.csv',index=False)
            totals[model]=dict(features=len(diagnostics),valid_full=sum(row['valid'] for row in diagnostics),observations=len(meta),participants=meta.participant_id.nunique(),dyads=meta.dyad.nunique())
            write(folder/'models_complete.json',totals[model])
        complete=dict(status='pilot_complete' if args.pilot else 'clinical_combined_model_fits_complete',datasets=totals,worker_running=False,elapsed_seconds=time.monotonic()-start,originals_changed=False,manuscript_changed=False)
        write(out/('pilot_complete.json' if args.pilot else 'model_run_complete.json'),complete);status(stage=complete['status'],worker_running=False);print(json.dumps(complete),flush=True)
    except Exception as error:
        status(stage='failed_attention_required',worker_running=False,error=repr(error));traceback.print_exc();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot',action='store_true');parser.add_argument('--log',type=Path)
    args=parser.parse_args()
    if args.log:
        args.log.parent.mkdir(parents=True,exist_ok=True)
        with args.log.open('x',encoding='utf-8',buffering=1) as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):run(args)
    else:run(args)
