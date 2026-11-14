import numpy as np
from . import cohort_statistics as b

def calculate(c,n,draws):
 return np.stack([b.maxima(b.welch(c[ia],n[ib])) for ia,ib in draws])
