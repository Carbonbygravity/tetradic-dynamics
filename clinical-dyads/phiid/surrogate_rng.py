import hashlib,json
import numpy as np

def stable_rng(seed,*keys):
    digest=hashlib.sha256(json.dumps([seed,*keys],separators=(',',':')).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16],'little'))

def p2_trial_permutation(blocks,periods,rng):
    blocks=np.asarray(blocks);periods=np.asarray(periods)
    if blocks.ndim!=1 or blocks.shape!=periods.shape:raise ValueError('Misaligned strata')
    permutation=np.arange(len(blocks))
    for block,period in sorted(set(zip(blocks.tolist(),periods.tolist()))):
        idx=np.flatnonzero((blocks==block)&(periods==period))
        permutation[idx]=rng.permutation(idx)
    return permutation
