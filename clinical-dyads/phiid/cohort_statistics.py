import numpy as np
from scipy import stats,ndimage
TH=float(stats.t.ppf(.975,14))
STRUCT=np.array([[0,0,0],[1,1,1],[0,0,0]])

def welch(c,n):
 v=c.var(0,ddof=1)/len(c)+n.var(0,ddof=1)/len(n)
 if not np.isfinite(c).all() or not np.isfinite(n).all() or np.any(v<=0):raise ValueError('Undefined statistic')
 t=(c.mean(0)-n.mean(0))/np.sqrt(v)
 if not np.isfinite(t).all():raise ValueError('Nonfinite statistic')
 return t

def maxima(t):
 out=np.zeros(t.shape[0])
 for sign in (1,-1):
  lab,num=ndimage.label(sign*t>TH,structure=STRUCT)
  if num:
   masses=np.bincount(lab.ravel(),weights=(sign*t).ravel())[1:]
   for k,s in enumerate(ndimage.find_objects(lab)):out[s[0].start]=max(out[s[0].start],masses[k])
 return out

def pv(null,m):return float((1+np.count_nonzero(null>=m-1e-12*max(1.,m)))/(len(null)+1))
