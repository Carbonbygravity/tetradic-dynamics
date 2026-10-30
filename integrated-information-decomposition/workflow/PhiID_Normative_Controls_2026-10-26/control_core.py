"""Batched mean Gaussian/MMI atoms with production-compatible trial observations."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,importlib.util
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent.parent
P=W/'PhiID_Production_Preflight_2026-10-25'
sys.path.insert(0,str(P))
sp=importlib.util.spec_from_file_location('production_reference',P/'runner.py');prod=importlib.util.module_from_spec(sp);sp.loader.exec_module(prod)
sys.path.insert(0,str(W/'PhiID_Inference_Control_Plan_2026-10-26'))
import inference_core as common

def estimate_batch(v):
    # v: centres x four variables x trials. Normalization matches production float precision.
    if v.ndim!=3 or v.shape[1]!=4 or v.shape[2]<10 or not np.isfinite(v).all():raise ValueError('Invalid observations')
    sd=np.std(v,axis=2,ddof=1,keepdims=True)
    if np.any(sd<=0):raise ValueError('Zero variance')
    z=(v/sd).astype(np.float64);n=z.shape[2];z-=z.mean(axis=2,keepdims=True)
    cov=np.einsum('tin,tjn->tij',z,z,optimize=False)/(n-1)
    subsets={'h_p1':[0],'h_p2':[1],'h_t1':[2],'h_t2':[3],'h_p1p2':[0,1],'h_t1t2':[2,3],'h_p1t1':[0,2],'h_p1t2':[0,3],'h_p2t1':[1,2],'h_p2t2':[1,3],'h_p1p2t1':[0,1,2],'h_p1p2t2':[0,1,3],'h_p1t1t2':[0,2,3],'h_p2t1t2':[1,2,3],'h_p1p2t1t2':[0,1,2,3]}
    h={}
    for key,idx in subsets.items():
        c=cov[:,idx][:,:,idx];ev=np.linalg.eigvalsh(c)
        if np.any(ev[:,0]<=1e6*np.finfo(float).eps*ev[:,-1]):raise ValueError('Singular covariance; no regularization')
        sign,ld=np.linalg.slogdet(c);assert np.all(sign>0);d=len(idx)
        h[key]=.5*(d*np.log(2*np.pi)+ld+d*(n-1)/n)
    i=prod.calc._get_coinfo_four_vec(h)
    r={k:np.minimum(i[a],i[b]) for k,a,b in [('R_xyta','I_xta','I_yta'),('R_xytb','I_xtb','I_ytb'),('R_xytab','I_xtab','I_ytab'),('R_abtx','I_xta','I_xtb'),('R_abty','I_yta','I_ytb'),('R_abtxy','I_xyta','I_xytb')]}
    a=prod.calc._get_atoms_four_vec(dict(I_res=i,R_res=r,rtr=np.minimum.reduce([i[k] for k in ['I_xta','I_xtb','I_yta','I_ytb']])))
    result=np.stack([a[k] for k in prod.ATOMS],axis=1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite atoms')
    return result

def blocks_periods(events):
    blocks=np.array([1 if e[-1]=='1' and e[2]=='R' else 2 if e[-1]=='1' else 3 if e[2]=='A' else 4 for e in events])
    return blocks,np.array([e[0] for e in events])

def perturbation(events,dyad,rep,kind):
    rng=common.stable_rng(1337,'normative',kind,int(dyad),int(rep))
    if kind=='trial':
        blocks,periods=blocks_periods(events);return common.p2_trial_permutation(blocks,periods,rng),0
    if kind=='time':return np.arange(len(events)),int(rng.integers(20,2001//2))
    return np.arange(len(events)),0

def cell_atoms(p1,p2,events,cell,geometry,band,centres,perm,lag):
    idx=np.flatnonzero([e[:3]==cell for e in events]);mask=np.array([events[j][-1]=='1' for j in idx])
    p2idx=perm[idx];past=centres-1;future=centres+1
    def take(p,rows,roi,t):return p[rows,roi,band][:,t]
    # Circular shifting physical P2 precedes per-epoch actor alignment.
    xpast=np.where(mask[:,None],take(p1,idx,0,past),take(p2,p2idx,0,(past-lag)%2001))
    xfuture=np.where(mask[:,None],take(p1,idx,0,future),take(p2,p2idx,0,(future-lag)%2001))
    ypast=np.where(mask[:,None],take(p2,p2idx,geometry,(past-lag)%2001),take(p1,idx,geometry,past))
    yfuture=np.where(mask[:,None],take(p2,p2idx,geometry,(future-lag)%2001),take(p1,idx,geometry,future))
    v=np.stack([xpast,ypast,xfuture,yfuture],axis=1).transpose(2,1,0)
    # Bounded allocation per covariance batch.
    return np.concatenate([estimate_batch(v[k:k+64]) for k in range(0,len(v),64)]),v
