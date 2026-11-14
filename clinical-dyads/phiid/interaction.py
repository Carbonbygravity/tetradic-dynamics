import numpy as np
COEF=np.array([-1.,1.,-1.,1.])

def contrast(v):return np.einsum('c,gcbta->gbat',COEF,v,optimize=False).reshape(48,1999)

def direct(v):return ((v[:,1]-v[:,2])-(v[:,0]-v[:,3])).transpose(0,1,3,2).reshape(48,1999)
