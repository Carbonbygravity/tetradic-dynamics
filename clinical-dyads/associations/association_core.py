"""Statistical primitives only. Importing this module never reads study data."""
import numpy as np
from scipy import stats
import warnings

SEED=20261104
TOL=100*np.finfo(float).eps
def generator(kind,key):
    return np.random.default_rng(np.random.SeedSequence([SEED,int(kind),int(key)]))

def rho(x,y,axis=-1):
    x=stats.rankdata(np.asarray(x,dtype=float),method='average',axis=axis)
    y=stats.rankdata(np.asarray(y,dtype=float),method='average',axis=axis)
    x=x-x.mean(axis=axis,keepdims=True);y=y-y.mean(axis=axis,keepdims=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        v=np.sum(x*y,axis=axis)/np.sqrt(np.sum(x*x,axis=axis)*np.sum(y*y,axis=axis))
    return np.clip(v,-1,1)

def normalized_ranks(a):
    a=stats.rankdata(a,axis=0,method='average');a=a-a.mean(axis=0)
    den=np.linalg.norm(a,axis=0)
    if np.any(den==0): raise ValueError('Constant input column')
    return a/den

def permutation_counts(x,y,resamples,rng,batch=2000):
    """x=n x features; y=n x scores. One shared patient permutation per draw."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    assert x.ndim==y.ndim==2 and len(x)==len(y)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    xr=normalized_ranks(x);yr=normalized_ranks(y)
    observed=np.clip(xr.T@yr,-1,1);count=np.zeros(observed.shape,dtype=np.int64)
    for start in range(0,resamples,batch):
        k=min(batch,resamples-start)
        # Repeated rng.permutation calls make the stream independent of batch size.
        indices=np.array([rng.permutation(len(x)) for _ in range(k)])
        null=np.einsum('ni,bnj->bij',xr,yr[indices],optimize=True)
        count+=(np.abs(null)>=np.abs(observed)[None,:,:]-TOL).sum(axis=0)
    return observed,(count+1)/(resamples+1),count

def by_adjust(p):
    return stats.false_discovery_control(np.asarray(p,float),method='by')

def leave_one_out(x,y):
    return np.array([rho(np.delete(x,i),np.delete(y,i)) for i in range(len(x))])

def bca_interval(x,y,resamples,rng,batch=1000):
    x=np.asarray(x,float);y=np.asarray(y,float)
    jack=leave_one_out(x,y)
    if not np.isfinite(jack).all() or np.sum((jack.mean()-jack)**2)==0:
        return dict(low=None,high=None,status='unavailable: undefined jackknife acceleration',invalid_resamples=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r=stats.bootstrap((x,y),rho,paired=True,vectorized=True,axis=-1,n_resamples=resamples,batch=batch,confidence_level=.95,method='BCa',rng=rng)
    bad=int((~np.isfinite(r.bootstrap_distribution)).sum())
    lo,hi=map(float,r.confidence_interval)
    if bad or not np.isfinite([lo,hi]).all():
        return dict(low=None,high=None,status='unavailable: undefined bootstrap correlation or BCa interval',invalid_resamples=bad,warnings=sorted({str(w.message) for w in caught}))
    return dict(low=lo,high=hi,status='available',invalid_resamples=0,warnings=sorted({str(w.message) for w in caught}))
