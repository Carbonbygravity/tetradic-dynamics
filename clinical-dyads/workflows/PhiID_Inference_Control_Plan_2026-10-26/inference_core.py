"""Temporal-cluster one-sample tests and physical-P2 surrogate operators.

No filesystem mutations or study-data analysis on import. Broader correction
uses a common sign assignment across all maps and their maximum cluster mass.
Spatial/frequency/geometry coordinates do not create cluster adjacency.
"""
import hashlib,json
import numpy as np
from scipy import ndimage,stats

CELL_ORDER=('LMR','LMA','FMA','FMR')
CONTRASTS={
    'Matching_minus_Mirroring':np.array([-.5,.5,.5,-.5]),
    'Leader_period_minus_Follower_period':np.array([.5,.5,-.5,-.5]),
    'Role_by_Condition':np.array([-1.,1.,-1.,1.]),
    'Leader_minus_Follower_Mirroring':np.array([1.,0.,0.,-1.]),
    'Leader_minus_Follower_Matching':np.array([0.,1.,-1.,0.]),
}
PRIMARY=tuple(CONTRASTS)[:3]
STRUCTURE=np.array([[0,0,0],[1,1,1],[0,0,0]])

def stable_rng(seed,*keys):
    digest=hashlib.sha256(json.dumps([seed,*keys],separators=(',',':')).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16],'little'))

def sign_assignments(n,count=5000,seed=7):
    if not 2<=n<=20:raise ValueError('Supported sign-enumeration range is 2..20 dyads')
    total=2**n
    if count>=total:codes=np.arange(total-1,-1,-1,dtype=np.uint64)
    else:
        if count<2:raise ValueError('At least two assignments required')
        codes=np.r_[total-1,stable_rng(seed,'signs',n,count).choice(total-1,count-1,replace=False)].astype(np.uint64)
    return (((codes[:,None] >> np.arange(n,dtype=np.uint64))&1)*2).astype(float)-1

def t_from_sums(sums,sumsq,n):
    numerator=sums/n
    variance=(sumsq-sums*sums/n)/(n-1)
    tolerance=1e-12*np.maximum(sumsq,1.)
    if np.any(variance < -tolerance):raise ValueError('Invalid negative variance')
    variance=np.maximum(variance,0.)
    if np.any((variance==0)&(numerator!=0)):raise ValueError('Nonzero constant dyad contrast; review required')
    return np.divide(numerator,np.sqrt(variance/n),out=np.zeros_like(numerator),where=variance>0)

def cluster_masses(tmap,threshold):
    """Separate positive/negative contiguous time runs; maps remain disconnected."""
    if tmap.ndim!=2 or not np.isfinite(tmap).all():raise ValueError('Expected finite maps x time')
    result=[]
    for sign in [1,-1]:
        labels,n=ndimage.label(sign*tmap>threshold,structure=STRUCTURE)
        if not n:continue
        masses=np.bincount(labels.ravel(),weights=(sign*tmap).ravel(),minlength=n+1)[1:]
        for k,slc in enumerate(ndimage.find_objects(labels),1):
            row=slc[0].start
            result.append(dict(map=row,start=slc[1].start,stop=slc[1].stop,sign=sign,mass=float(masses[k-1])))
    return result

def test_family(values,signs,forming_p=.05):
    """values: dyads x maps x time. Returns observed clusters and null maxima.

    Identity must be the first assignment; its inclusion supplies the +1 in
    Monte Carlo p values. Full enumeration is exact under sign exchangeability.
    Per-map and broad results share the same permutations and observed clusters.
    """
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=3 or not np.isfinite(x).all():raise ValueError('Expected finite dyads x maps x time')
    n,m,_=x.shape
    if signs.ndim!=2 or signs.shape[1]!=n or not np.all(signs[0]==1) or not np.isin(signs,[-1,1]).all():raise ValueError('Invalid sign assignments')
    threshold=float(stats.t.ppf(1-forming_p/2,n-1));sumsq=(x*x).sum(axis=0)
    observed=cluster_masses(t_from_sums(x.sum(axis=0),sumsq,n),threshold)
    null_map=np.zeros((len(signs),m))
    for i,assignment in enumerate(signs):
        tmap=t_from_sums(np.einsum('n,nmt->mt',assignment,x,optimize=False),sumsq,n)
        for c in cluster_masses(tmap,threshold):null_map[i,c['map']]=max(null_map[i,c['map']],c['mass'])
    null_family=null_map.max(axis=1)
    for c in observed:
        # Tolerance prevents identity summation roundoff from excluding itself.
        cutoff=c['mass']-1e-12*max(1.,abs(c['mass']))
        c['p_primary_family']=float(np.mean(null_family>=cutoff))
        c['p_historical_per_map']=float(np.mean(null_map[:,c['map']]>=cutoff))
    return dict(clusters=observed,threshold=threshold,null_family=null_family,null_map=null_map)

def p2_trial_permutation(blocks,periods,rng):
    blocks=np.asarray(blocks);periods=np.asarray(periods)
    if blocks.ndim!=1 or blocks.shape!=periods.shape:raise ValueError('Misaligned strata')
    permutation=np.arange(len(blocks))
    for block,period in sorted(set(zip(blocks.tolist(),periods.tolist()))):
        idx=np.flatnonzero((blocks==block)&(periods==period))
        permutation[idx]=rng.permutation(idx)
    return permutation

def shuffle_physical_p2(p1,p2,blocks,periods,rng):
    if p1.shape!=p2.shape or len(p2)!=len(blocks):raise ValueError('Unpaired data')
    permutation=p2_trial_permutation(blocks,periods,rng)
    return p1,p2[permutation],permutation

def shift_physical_p2(p1,p2,lag):
    if p1.shape!=p2.shape or not 0<lag<p2.shape[-1]:raise ValueError('Invalid shift')
    return p1,np.roll(p2,shift=lag,axis=-1)
