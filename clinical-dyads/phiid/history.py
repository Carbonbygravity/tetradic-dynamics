import numpy as np
from scipy.spatial import cKDTree
from . import controls as c

def design(ev,cell,centres,d):
 idx=np.flatnonzero([e[:3]==cell for e in ev]);fold=np.empty(len(idx),int)
 rng=c.common.stable_rng(2026,'history_folds',d,cell)
 for suffix in ['1','2']:
  ix=np.flatnonzero([ev[j][-1]==suffix for j in idx]);assert len(ix)>=5
  shuffled=rng.permutation(ix);fold[shuffled]=np.arange(len(ix))%5
 # Equal samples per trial, cap <=25,000 per cell across all folds.
 ntime=min(len(centres),25000//len(idx));assert ntime>=1
 positions=np.sort(rng.choice(len(centres),ntime,replace=False))
 return idx,fold,positions

def cv(v,fold,positions):
 # v centres x variables x paired trials. Each physical trial appears in one fold.
 z=v[positions].transpose(2,0,1).reshape(-1,4).astype('float64');ids=np.repeat(np.arange(len(fold)),len(positions));foldpoints=fold[ids]
 errors=np.empty((len(fold),2));scored=np.zeros(len(fold),bool)
 for f in range(5):
  tr=foldpoints!=f;te=~tr
  assert not set(ids[tr])&set(ids[te]);assert tr.sum()>=10 and te.any()
  mean=z[tr,:2].mean(0);sd=z[tr,:2].std(0,ddof=1);scale=z[tr,2:].var(0,ddof=1);assert (sd>0).all() and (scale>0).all()
  tree=cKDTree((z[tr,:2]-mean)/sd);_,nn=tree.query((z[te,:2]-mean)/sd,k=10,workers=1)
  pred=z[tr,2:][nn].mean(1);loss=(pred-z[te,2:])**2/scale
  for j in np.unique(ids[te]):errors[j]=loss[ids[te]==j].mean(0);scored[j]=True
 assert scored.all();return float(errors.mean()),len(z)
