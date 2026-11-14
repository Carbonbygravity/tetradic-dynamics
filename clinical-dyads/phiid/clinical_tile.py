import numpy as np
from .estimator import estimate, load_power
from .tetradic_phiid.core import actor_rows

def aligned_rows(a,b,mapping):
    for k in ['events','selection','event_labels','roi_labels','band_labels','times_seconds']:
        np.testing.assert_array_equal(a[k][:],b[k][:])
    assert a.attrs['complete'] and b.attrs['complete']
    n=len(a['event_labels']);assert len(set(a['selection'][:].tolist()))==n
    assert a['roi_time_series'].shape==b['roi_time_series'].shape==(n,68,2001)
    np.testing.assert_allclose(a['times_seconds'][:],np.arange(-1000,1001)/1000,rtol=0,atol=1e-12)
    labels=a['event_labels'].asstr()[:]
    blocks=np.array([mapping['events'][e]['block'] for e in labels])
    assert np.all(np.diff(blocks.astype(int))>=0)
    return actor_rows(mapping,dict(epoch_id=a['selection'][:].astype(str),block_id=blocks,event_label=labels))

def compute_tile(a,b,roi,centres,mapping):
    rows=aligned_rows(a,b,mapping)
    labels=a['roi_labels'].asstr()[:].tolist();lab=labels[roi]
    opposite=labels.index(lab[:-3]+('-rh' if lab.endswith('-lh') else '-lh'))
    p1,p2=load_power(a,[roi,opposite]),load_power(b,[roi,opposite])
    result=np.empty((2,4,6,len(centres),4),dtype='float64');counts=[]
    cells=[('Leader','Mirroring'),('Leader','Matching'),('Follower','Matching'),('Follower','Mirroring')]
    for ci,cell in enumerate(cells):
        idx=np.array([j for j,r in enumerate(rows) if (r['period'],r['condition'])==cell])
        mask=np.array([rows[j]['X']=='P1' for j in idx]);counts.append(len(idx))
        assert len(idx)>=10
        for geometry in range(2):
            x=np.where(mask[:,None,None],p1[idx,0],p2[idx,0])
            y=np.where(mask[:,None,None],p2[idx,geometry],p1[idx,geometry])
            for band in range(6):
                for ti,t in enumerate(centres):
                    v=np.stack([x[:,band,t-1],y[:,band,t-1],x[:,band,t+1],y[:,band,t+1]])
                    result[geometry,ci,band,ti]=estimate(v)
    return result,dict(roi=lab,opposite_roi=labels[opposite],counts=counts)
