NAMES=('Matching_minus_Mirroring','Leader_minus_Follower_Mirroring','Leader_minus_Follower_Matching','Leader_period_minus_Follower_period')

def direct(v,name):
 if name==NAMES[0]:return ((v[:,1]+v[:,2])-(v[:,0]+v[:,3]))/2
 if name==NAMES[1]:return v[:,0]-v[:,3]
 if name==NAMES[2]:return v[:,1]-v[:,2]
 if name==NAMES[3]:return ((v[:,0]+v[:,1])-(v[:,2]+v[:,3]))/2
 raise ValueError(name)
