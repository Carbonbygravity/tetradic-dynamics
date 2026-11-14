def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))

def signature(x):return hashlib.sha256(canonical(x).encode()).hexdigest()

def atomic_json(p,obj):
    temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
    with temp.open('x') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def checkpoint_ok(path,spec,shape):
    marker=path.with_suffix('.complete.json')
    if not path.exists() and not marker.exists():return False
    if not path.exists() or not marker.exists():raise RuntimeError(f'Incomplete checkpoint requires review: {path}')
    record=json.loads(marker.read_text())
    if record.get('signature')!=spec or record.get('sha256')!=digest(path):raise RuntimeError(f'Checkpoint hash/specification mismatch: {path}')
    with h5py.File(path,'r') as f:
        if f.attrs.get('signature')!=spec or f['values'].shape!=tuple(shape):raise RuntimeError(f'Checkpoint metadata mismatch: {path}')
        for block in range(shape[0]):
            if not np.isfinite(f['values'][block]).all():raise RuntimeError(f'Nonfinite checkpoint: {path}')
    return True

def publish(path,values,spec,metadata):
    if path.exists() or path.with_suffix('.complete.json').exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with h5py.File(temp,'w') as f:
        f['values']=values;f.attrs['signature']=spec;f.attrs['metadata']=canonical(metadata);f.flush()
    with h5py.File(temp,'r') as f:
        np.testing.assert_array_equal(f['values'][:],values)
        assert np.isfinite(f['values'][:]).all()
    checksum=digest(temp);temp.rename(path)
    atomic_json(path.with_suffix('.complete.json'),dict(signature=spec,sha256=checksum,shape=list(values.shape)))
    assert checkpoint_ok(path,spec,values.shape)

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
