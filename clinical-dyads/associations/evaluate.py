"""Numerical evaluator for the clinical main and interaction families."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json, hashlib, time, sys
import numpy as np
import pandas as pd
import scipy
import association_core as c

O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(v): return float(v) if np.isfinite(v) else None

def evaluate(x,y,features,outcomes,inventory,nperm,nboot,progress=None):
    """Called with synthetic inputs in preflight; study inputs only after launch approval."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    assert x.shape==(len(y),len(features)) and y.shape[1]==len(outcomes)
    assert np.isfinite(x).all() and len({r['test_id'] for r in inventory})==len(inventory)
    fi={v:i for i,v in enumerate(features)};oi={v:i for i,v in enumerate(outcomes)}
    groups={}
    for j in range(y.shape[1]):
        mask=np.isfinite(y[:,j]);key=sum(int(ok)<<i for i,ok in enumerate(mask))
        groups.setdefault(key,(mask,[]))[1].append(j)
    observations={};pvalues={};extremes={}
    for key,(mask,jjs) in sorted(groups.items()):
        obs,p,counts=c.permutation_counts(x[mask],y[mask][:,jjs],nperm,c.generator(1,key))
        for jj,j in enumerate(jjs):
            for i in range(len(features)):
                observations[i,j]=float(obs[i,jj]);pvalues[i,j]=float(p[i,jj]);extremes[i,j]=int(counts[i,jj])
    rows=[]
    for k,test in enumerate(inventory):
        i,j=fi[test['feature']],oi[test['outcome']];mask=np.isfinite(y[:,j]);xx=x[mask,i];yy=y[mask,j]
        assert len(xx)==int(test['n'])
        ci=c.bca_interval(xx,yy,nboot,c.generator(2,k));loo=c.leave_one_out(xx,yy);r=observations[i,j]
        row=dict(test,rho=r,permutation_p=pvalues[i,j],permutation_extreme_count=extremes[i,j],permutations=nperm,ci_low=ci['low'],ci_high=ci['high'],ci_status=ci['status'],bootstrap_invalid=ci['invalid_resamples'],bootstrap_warnings='; '.join(ci.get('warnings',[])),bootstrap_draws=nboot,bootstrap_stream=k,loo_min=safe(np.min(loo)),loo_max=safe(np.max(loo)),loo_max_absolute_change=safe(np.max(abs(loo-r))),loo_opposite_sign=bool(np.any(loo*r<0)),loo_undefined=int((~np.isfinite(loo)).sum()))
        rows.append(row)
        if progress:progress(k+1,len(inventory))
    for family in sorted({r['family'] for r in rows}):
        idx=[i for i,r in enumerate(rows) if r['family']==family]
        assert all(int(rows[i]['family_size'])==len(idx) for i in idx)
        adjusted=c.by_adjust([rows[i]['permutation_p'] for i in idx])
        for i,q in zip(idx,adjusted):rows[i].update(by_adjusted_p=float(q),passes_by_05=bool(q<=.05))
    return rows

