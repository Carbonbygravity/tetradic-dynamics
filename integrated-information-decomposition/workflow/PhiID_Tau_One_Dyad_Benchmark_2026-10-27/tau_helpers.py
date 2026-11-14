"""Original tau/history observation selection, isolated from preflight code."""

import numpy as np


def variables(p1,p2,ev,cell,g,b,centres,tau,perm,lag,H=1):
 idx=np.flatnonzero([e[:3]==cell for e in ev]);mask=np.array([ev[j][-1]=='1' for j in idx])[:,None];j=perm[idx]
 def take(p,rows,roi,t,shift=0):return p[rows,roi,b][:,(t-shift)%2001]
 def actor(t,x):
  return np.where(mask,take(p1,idx,0,t) if x else take(p2,j,g,t,lag),take(p2,j,0,t,lag) if x else take(p1,idx,g,t))
 xp=actor(centres-tau,True);yp=actor(centres-tau,False)
 if H>1:
  xp=np.mean(np.stack([actor(centres-k*tau,True) for k in range(1,H+1)]),axis=0)
  yp=np.mean(np.stack([actor(centres-k*tau,False) for k in range(1,H+1)]),axis=0)
 return np.stack([xp,yp,actor(centres+tau,True),actor(centres+tau,False)],axis=1).transpose(2,1,0)
