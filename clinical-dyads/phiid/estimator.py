import numpy as np
import mne
from .tetradic_phiid.vendor.phyid import calculate as calc
CELLS=['LMR','LMA','FMA','FMR'];ATOMS=['rtr','xtx','yty','sts']
FREQS=np.arange(4.,46.);BANDS=[(4,7),(8,9),(10,12),(13,19),(20,29),(30,45)]

def estimate(v):
    if v.shape[1]<10 or not np.isfinite(v).all():raise ValueError('Insufficient/nonfinite paired observations')
    # Preserve the benchmark's observation-selection layout and reduction order.
    v=v[:,np.isfinite(v).all(axis=0)]
    sd=np.std(v,axis=1,ddof=1,keepdims=True)
    if (sd<=0).any():raise ValueError('Zero variance; no automatic regularization')
    v=v/sd;h=calc._get_entropy_four_vec(v,'gaussian');i=calc._get_coinfo_four_vec(h)
    res=dict(h_res=h,I_res=i,R_res=calc._get_redundancy_four_vec('MMI',i))
    res['rtr']=calc._get_double_redundancy_four_vec('MMI',res)
    a=calc._get_atoms_four_vec(res)
    result=np.array([np.mean(a[k]) for k in ATOMS])
    if not np.isfinite(result).all():raise ValueError('Nonfinite estimator result')
    return result

def load_power(f,rois):
    n=len(f['event_labels']);out=np.empty((n,len(rois),6,2001),dtype='float32')
    for ri,r in enumerate(rois):
        for start in range(0,n,8):
            ts=f['roi_time_series'][start:start+8,r:r+1,:]
            if not np.isfinite(ts).all():raise ValueError('Nonfinite source signal')
            tf=mne.time_frequency.tfr_array_morlet(ts,sfreq=1000,freqs=FREQS,n_cycles=FREQS/2,zero_mean=False,output='power',use_fft=True,n_jobs=1,verbose=False)
            for b,(lo,hi) in enumerate(BANDS):
                pw=tf[:,:,(FREQS>=lo)&(FREQS<=hi),:].mean(axis=2).astype('float32')
                out[start:start+len(ts),ri,b]=np.log1p(pw[:,0])
    return out
