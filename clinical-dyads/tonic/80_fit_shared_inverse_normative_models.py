"""Uniform Gaussian ML nested-intercept fits for the two approved branches.

Woodbury sufficient statistics avoid repeatedly factoring large trial matrices.
Boundary variance candidates are fitted explicitly under the SAME specification.
No random-slope, feature-dependent model selection or significance-based fallback.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import argparse, contextlib, datetime, hashlib, json, time, traceback
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, chi2, skew, kurtosis
from statsmodels.stats.multitest import multipletests

NAMES=['Intercept','role_code','cond_code','role_code:cond_code','rt_z']
EFFECTS=['role','condition','role:condition','rt_z']
SPEC=dict(formula='log1p(power) ~ role_code * cond_code + rt_z + (1|dyad) + (1|dyad:participant)',
    coding='Leader +0.5 / Follower -0.5; Matching +0.5 / Mirroring -0.5',
    role='response-event role, NOT recorded participant task role',estimation='Gaussian ML',
    response='float64 log1p of unchanged saved float32 raw mean-ROI Welch power',
    RT='verified measured original-export interval; existing primary screening membership; common full-eligible within-recorded-participant ddof1 scaling',
    optimizer='profile Gaussian ML, three fixed L-BFGS-B starts; explicit each-zero variance faces and all-zero corner; no per-feature structure substitution',
    numerical_bounds='log1p(relative variance) in [0,20]; upper-bound solutions invalid',
    fixed_effect_inference='observed joint information for positive variance components; zero-boundary components fixed at their ML zero for information, boundary clearly flagged',
    boundary_caveat='Boundary Wald intervals are approximate, conditional on zero nuisance component. They are not variance-component significance tests.',
    validity='finite likelihood and SE; normalized projected score <=1e-5; positive information; no upper guard; independently checked profile optimizer',
    LRT='Historical-style 1-df single fixed-column removal, other columns retained (including interaction); NOT Type III or omnibus factorial tests',
    multiplicity='BH fixed family of 408 features separately per effect, test and branch; unresolved p treated as 1 for correction then q restored to NA',
    historical_status='Not historical 279-effect replication; both new branches share repaired dyad6, new inverse and revised masks',
    source_changes=False,packages={p:version(p) for p in ['numpy','pandas','scipy','statsmodels']})

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)

class NestedProfile:
    def __init__(self,X,y,meta):
        self.X=X; self.y=y; self.n=len(y); self.p=X.shape[1]
        self.dyads=sorted(meta.dyad.unique()); self.ids=[]; self.slices=[]
        self.zx=[]; self.zy=[]; self.ns=[]
        for dyad in self.dyads:
            people=sorted(meta.loc[meta.dyad.eq(dyad),'participant_id'].unique())
            assert len(people)==2
            indices=[np.flatnonzero(meta.participant_id.eq(person)) for person in people]
            self.ids.append(people);self.slices.append(indices)
            self.zx.append([X[i].sum(axis=0) for i in indices]);self.zy.append([y[i].sum() for i in indices]);self.ns.append([len(i) for i in indices])
        self.zx=np.array(self.zx);self.zy=np.array(self.zy);self.ns=np.array(self.ns,float)
        self.xx=X.T@X;self.xy=X.T@y;self.yy=y@y
        self.eye=np.eye(2);self.ones=np.ones((2,2))

    def quantities(self,t):
        g=np.expm1(t); B=g[0]*self.eye+g[1]*self.ones
        # Determinant and inverse in participant-sized two-by-two blocks.
        M=self.eye+B[None,:,:]*self.ns[:,None,:]
        sign,logdet=np.linalg.slogdet(M)
        assert (sign>0).all()
        W=np.linalg.solve(M,np.broadcast_to(B,M.shape))
        U=self.eye-W*self.ns[:,None,:]
        derivatives=np.array([np.einsum('dij,jk,dlk->dil',U,(1+g[k])*basis,U)
                              for k,basis in enumerate([self.eye,self.ones])])
        A=self.xx-np.einsum('dij,dik,djl->kl',W,self.zx,self.zx)
        b=self.xy-np.einsum('dij,dik,dj->k',W,self.zx,self.zy)
        c=self.yy-np.einsum('di,dij,dj->',self.zy,W,self.zy)
        ld=logdet.sum()
        ldp=np.array([np.einsum('dij,ji,di->',U,(1+g[k])*basis,self.ns)
                      for k,basis in enumerate([self.eye,self.ones])])
        return g,W,derivatives,A,b,c,ld,ldp

    def evaluate(self,t,details=False):
        try:
            g,W,D,A,b,c,ld,ldp=self.quantities(t)
            beta=np.linalg.solve(A,b);rss=c-b@beta;s=rss/self.n
            if s<=0 or not np.isfinite(s): raise ValueError('Nonpositive residual variance')
            residual_sums=self.zy-np.einsum('dik,k->di',self.zx,beta)
            rp=-np.einsum('di,kdij,dj->k',residual_sums,D,residual_sums)
            nll=.5*(self.n*(np.log(2*np.pi)+1+np.log(s))+ld)
            grad=.5*(ldp+rp/s)
            if details:return dict(g=g,W=W,D=D,A=A,b=b,c=c,ld=ld,ldp=ldp,beta=beta,s=s,nll=nll,grad=grad)
            return nll,grad
        except (AssertionError,ValueError,np.linalg.LinAlgError):
            if details:raise
            return 1e100,np.zeros(2)

    def joint_score(self,v):
        beta=v[:self.p];t=v[self.p:self.p+2];s=np.exp(v[-1])
        _,_,D,A,b,c,_,ldp=self.quantities(t)
        rsum=self.zy-np.einsum('dik,k->di',self.zx,beta)
        rp=-np.einsum('di,kdij,dj->k',rsum,D,rsum)
        rss=c-2*beta@b+beta@A@beta
        return np.r_[(A@beta-b)/s,.5*(ldp+rp/s),.5*(self.n-rss/s)]

    def fit(self,unit,mean,verify=False):
        attempts=[]
        for gstart in [[.001,.001],[.05,.2],[.5,.05]]:
            result=minimize(self.evaluate,np.log1p(gstart),jac=True,method='L-BFGS-B',bounds=[(0,20)]*2,
                options=dict(maxiter=400,ftol=1e-13,gtol=1e-7,maxls=40))
            attempts.append(dict(method='L-BFGS-B',start=gstart,t=result.x.tolist(),nll=float(result.fun),success=bool(result.success)))
        # Explicit faces/corner retain the same nested-intercept specification.
        for active in [0,1]:
            def face(t):
                v=np.zeros(2);v[active]=t;return self.evaluate(v)[0]
            result=minimize_scalar(face,bounds=(0,20),method='bounded',options=dict(maxiter=300,xatol=1e-9))
            t=np.zeros(2);t[active]=result.x
            attempts.append(dict(method=f'zero_face_{1-active}',t=t.tolist(),nll=float(result.fun),success=bool(result.success)))
        attempts.append(dict(method='zero_corner',t=[0.,0.],nll=float(self.evaluate(np.zeros(2))[0]),success=True))
        chosen=min(attempts,key=lambda v:v['nll']);t=np.array(chosen['t']);d=self.evaluate(t,True)
        boundary=t<1e-7;guard=bool((t>19.9).any())
        v=np.r_[d['beta'],t,np.log(d['s'])];score=self.joint_score(v)
        projected=score.copy()
        for k in [0,1]:
            if boundary[k]:projected[self.p+k]=min(0,score[self.p+k])
        grad=float(np.linalg.norm(projected)/self.n)
        active=np.r_[np.arange(self.p),self.p+np.flatnonzero(~boundary),self.p+2]
        h=1e-5;information=np.column_stack([(self.joint_score(v+np.eye(len(v))[j]*h)-self.joint_score(v-np.eye(len(v))[j]*h))/(2*h) for j in active])[active]
        information=(information+information.T)/2
        info_pd=False;se=np.full(self.p,np.nan);cov=np.full((self.p,self.p),np.nan)
        try:
            scales=np.sqrt(np.diag(information));scaled=information/scales[:,None]/scales[None,:]
            np.linalg.cholesky(scaled)
            covariance=np.linalg.inv(scaled)/scales[:,None]/scales[None,:]
            cov=covariance[:self.p,:self.p]*unit**2;se=np.sqrt(np.diag(cov))
            info_pd=bool(np.isfinite(se).all() and (se>0).all())
        except (ValueError,np.linalg.LinAlgError):pass
        beta=d['beta']*unit;beta[0]+=mean
        valid=bool(info_pd and not guard and grad<=1e-5 and np.isfinite(d['nll']))
        record=dict(llf=float(-d['nll']-self.n*np.log(unit)),coef=beta.tolist(),se=se.tolist(),cov_beta=cov.tolist(),
            relative_participant_variance=float(d['g'][0]),relative_dyad_variance=float(d['g'][1]),
            participant_variance=float(d['g'][0]*d['s']*unit**2),dyad_variance=float(d['g'][1]*d['s']*unit**2),
            residual_variance=float(d['s']*unit**2),participant_boundary=bool(boundary[0]),dyad_boundary=bool(boundary[1]),
            upper_guard=guard,information_pd=info_pd,normalized_projected_score=grad,valid=valid,attempts=attempts,
            optimizer_best_nll_gap=float(sorted(a['nll'] for a in attempts)[1]-chosen['nll']))
        if verify:
            # Same nested structure through statsmodels' independent implementation.
            import statsmodels.api as sm
            from statsmodels.regression.mixed_linear_model import MixedLMParams
            meta=pd.DataFrame(dict(y=self.y,x0=self.X[:,0],pid=np.empty(self.n,dtype=object),dyad=np.zeros(self.n,int)))
            for dyad,(people,indices) in enumerate(zip(self.ids,self.slices)):
                for person,idx in zip(people,indices):meta.loc[idx,'pid']=person;meta.loc[idx,'dyad']=dyad
            model=sm.MixedLM.from_formula('y ~ 1',meta,groups='dyad',re_formula='1',vc_formula={'participant':'0 + C(pid)'})
            # Substitute the identical fixed design while keeping independent variance-component construction.
            model=sm.MixedLM(self.y,self.X,groups=meta.dyad,exog_re=np.ones((self.n,1)),exog_vc=model.exog_vc)
            model.reml=False;model.cov_pen=None;model.fe_pen=None;model._freepat=None;model._cov_sing=0
            test=np.log1p([.3,.2]); direct=-self.evaluate(test)[0]
            reference=model.loglike(MixedLMParams.from_components(cov_re=np.array([[.2]]),vcomp=np.array([.3])),profile_fe=True)
            assert abs(direct-reference)<1e-5,(direct,reference)
            test_d=self.evaluate(test,True);step=1e-5
            numeric=np.array([(self.evaluate(test+np.eye(2)[j]*step)[0]-self.evaluate(test-np.eye(2)[j]*step)[0])/(2*step) for j in [0,1]])
            gradient_error=float(np.max(np.abs(numeric-test_d['grad']))/self.n)
            assert gradient_error<1e-7,gradient_error
            independent=minimize(lambda a:self.evaluate(a)[0],t,method='Powell',bounds=[(0,20)]*2,options=dict(maxiter=500,ftol=1e-12,xtol=1e-8))
            # Powell can miss an exact boundary; retain explicit verified faces in its comparison.
            independent_best=min(float(independent.fun),*(a['nll'] for a in attempts if a['method'].startswith('zero')))
            ll_error=float(abs(independent_best-d['nll']))
            assert ll_error<1e-4,ll_error
            record['verification']=dict(statsmodels_loglik_error=float(direct-reference),normalized_analytic_gradient_error=gradient_error,independent_optimizer_nll_error=ll_error)
        return record,d

def run(args):
    inputs=args.inputs;out=args.output
    validation=json.loads((inputs/'input_validation_complete.json').read_text())
    assert validation['status']=='verified_event_linked_inputs_ready'
    assert not (out/'model_run_complete.json').exists(),'Preserve completed models'
    out.mkdir(parents=True,exist_ok=True)
    frozen=out/'frozen_model_specification.json'
    spec={**SPEC,'input_validation_sha256':hashlib.sha256((inputs/'input_validation_complete.json').read_bytes()).hexdigest(),'pilot':args.pilot}
    if frozen.exists():assert json.loads(frozen.read_text())==spec
    else:write(frozen,spec)
    start=time.monotonic();allrows=[];all_diag=[]
    def status(**state):
        state.update(worker_pid=os.getpid(),updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        write(out/'model_run_status.json',state);print(json.dumps(state),flush=True)
    try:
        for branch in ['complete_epochs','historical_cap']:
            meta=pd.read_csv(inputs/branch/'model_metadata.csv')
            with np.load(inputs/branch/'model_power.npz',allow_pickle=False) as saved:Y=saved['raw_power'].astype(np.float64);features=saved['features'].tolist()
            assert len(meta)==validation['branches'][branch]['model_observations'] and Y.shape==(len(meta),408)
            X=np.column_stack([np.ones(len(meta)),meta.role_code,meta.cond_code,meta.role_code*meta.cond_code,meta.rt_z])
            folder=out/branch;(folder/'checkpoints').mkdir(parents=True,exist_ok=True)
            rows=[];diagnostics=[]
            selected=[0,68,136,340] if args.pilot else range(408)
            for idx in selected:
                feature=features[idx];checkpoint=folder/'checkpoints'/f'{idx:03d}.json'
                if checkpoint.exists():
                    saved=json.loads(checkpoint.read_text());rows.extend(saved['rows']);diagnostics.append(saved['diagnostics']);continue
                y=np.log1p(Y[:,idx]);mean=float(y.mean());unit=float(y.std());assert unit>0
                ys=(y-mean)/unit
                engine=NestedProfile(X,ys,meta);full,d=engine.fit(unit,mean,verify=idx in [0,68,136,340])
                results=[];reduced={}
                for effect,drop in zip(EFFECTS,range(1,5)):
                    red,_=NestedProfile(np.delete(X,drop,axis=1),ys,meta).fit(unit,mean)
                    reduced[effect]=red;delta=full['llf']-red['llf'];lrvalid=full['valid'] and red['valid'] and delta>=-1e-5
                    coefficient=full['coef'][drop] if full['valid'] else np.nan;se=full['se'][drop] if full['valid'] else np.nan
                    z=coefficient/se
                    results.append(dict(branch=branch,feature=feature,effect=effect,param=NAMES[drop],coef=coefficient,se=se,
                        ci_lo=coefficient-1.96*se,ci_hi=coefficient+1.96*se,z=z,p_wald=float(2*norm.sf(abs(z))),
                        lrt_chi2=2*max(delta,0) if lrvalid else np.nan,p_lrt=float(chi2.sf(2*max(delta,0),1)) if lrvalid else np.nan,
                        full_valid=full['valid'],reduced_valid=red['valid'],lrt_valid=bool(lrvalid),full_llf=full['llf'],reduced_llf=red['llf'],
                        participant_boundary=full['participant_boundary'],dyad_boundary=full['dyad_boundary'],n=len(meta),participants=30,dyads=15))
                # Conditional residuals use the SAME fitted random-intercept covariance.
                residual_sums=engine.zy-np.einsum('dik,k->di',engine.zx,d['beta'])
                intercepts=np.einsum('dij,dj->di',d['W'],residual_sums)
                fitted=X@d['beta']
                for indices,values in zip(engine.slices,intercepts):
                    for positions,value in zip(indices,values):fitted[positions]+=value
                residual=ys-fitted
                cell_stats=[]
                for cell in ['LMA','LMR','FMA','FMR']:
                    r=residual[meta.cell.eq(cell)];cell_stats.append(dict(cell=cell,n=len(r),residual_mean=float(r.mean()*unit),residual_sd=float(r.std(ddof=1)*unit)))
                diagnostic=dict(branch=branch,feature=feature,full_valid=full['valid'],information_pd=full['information_pd'],
                    normalized_projected_score=full['normalized_projected_score'],participant_variance=full['participant_variance'],
                    dyad_variance=full['dyad_variance'],residual_variance=full['residual_variance'],
                    participant_boundary=full['participant_boundary'],dyad_boundary=full['dyad_boundary'],
                    residual_skew=float(skew(residual)),residual_excess_kurtosis=float(kurtosis(residual)),
                    cell_residual_sd_max_min=float(max(c['residual_sd'] for c in cell_stats)/min(c['residual_sd'] for c in cell_stats)),cell_residuals=cell_stats)
                write(checkpoint,dict(feature=feature,full=full,reduced=reduced,rows=results,diagnostics=diagnostic))
                rows.extend(results);diagnostics.append(diagnostic)
                if args.pilot or (idx+1)%34==0:status(stage='fitting',branch=branch,completed_features=len(diagnostics),total_features=len(selected))
            frame=pd.DataFrame(rows)
            if not args.pilot:
                for effect in EFFECTS:
                    mask=frame.effect.eq(effect)
                    for col in ['p_wald','p_lrt']:
                        values=frame.loc[mask,col];assert len(values)==408
                        q=multipletests(values.fillna(1),method='fdr_bh')[1];q[values.isna()]=np.nan
                        frame.loc[mask,col+'_fdr']=q
            frame.to_csv(folder/'model_results.csv',index=False)
            pd.DataFrame([{k:v for k,v in row.items() if k!='cell_residuals'} for row in diagnostics]).to_csv(folder/'fit_and_residual_diagnostics.csv',index=False)
            pd.DataFrame([dict(branch=branch,feature=row['feature'],**cell) for row in diagnostics for cell in row['cell_residuals']]).to_csv(folder/'cell_residual_diagnostics.csv',index=False)
            allrows.append(frame);all_diag.extend(diagnostics)
        pd.concat(allrows,ignore_index=True).to_csv(out/'both_branch_model_results.csv',index=False)
        summary=dict(status='pilot_complete' if args.pilot else 'model_fitting_complete_comparison_pending',full_models=len(all_diag),
            valid_full_models=sum(row['full_valid'] for row in all_diag),elapsed_seconds=time.monotonic()-start,worker_running=False,
            manuscript_changed=False,originals_changed=False)
        write(out/('pilot_complete.json' if args.pilot else 'model_run_complete.json'),summary)
        status(stage=summary['status'],worker_running=False);print(json.dumps(summary),flush=True)
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
