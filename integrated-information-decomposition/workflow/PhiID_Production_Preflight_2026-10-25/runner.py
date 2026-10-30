"""Versioned normative PhiID runner. --smoke is bounded; --run requires approval."""
import os,sys,json,time,hashlib,argparse,uuid,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'normative_phiid_cache'))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,h5py,mne,scipy
from phyid import calculate as calc
from threadpoolctl import threadpool_limits
from prepare_inputs import digest
threadpool_limits(1)
CELLS=['LMR','LMA','FMA','FMR'];ATOMS=['rtr','xtx','yty','sts']
FREQS=np.arange(4.,46.);BANDS=[(4,7),(8,9),(10,12),(13,19),(20,29),(30,45)]
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
def compute_tile(a,b,roi,centres):
    labels=a['roi_labels'].asstr()[:].tolist();lab=labels[roi]
    opposite=labels.index(lab[:-3]+('-rh' if lab.endswith('-lh') else '-lh'))
    chosen=[roi,opposite];ev=a['event_labels'].asstr()[:]
    p1,p2=load_power(a,chosen),load_power(b,chosen)
    result=np.empty((2,4,6,len(centres),4),dtype='float64')
    counts=[]
    for ci,cell in enumerate(CELLS):
        idx=np.flatnonzero([e[:3]==cell for e in ev]);mask=np.array([ev[j][-1]=='1' for j in idx]);counts.append(len(idx))
        for geometry in range(2):
            # X ROI stays fixed in Leader coordinates; Y is same or opposite ROI.
            x=np.where(mask[:,None,None],p1[idx,0],p2[idx,0])
            y=np.where(mask[:,None,None],p2[idx,geometry],p1[idx,geometry])
            for band in range(6):
                for ti,t in enumerate(centres):
                    v=np.stack([x[:,band,t-1],y[:,band,t-1],x[:,band,t+1],y[:,band,t+1]])
                    result[geometry,ci,band,ti]=estimate(v)
    return result,dict(roi=lab,opposite_roi=labels[opposite],counts=counts)
def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke',action='store_true');g.add_argument('--run',action='store_true')
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--input-manifest',type=Path,required=True)
    parser.add_argument('--source-specification',type=Path,help='Required when the manifest records its SHA-256')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    input_root=args.input_dir.resolve();manifest_path=args.input_manifest.resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('source_specification_sha256'):
        if not args.source_specification:parser.error('--source-specification is required by this manifest')
        if digest(args.source_specification.resolve())!=manifest['source_specification_sha256']:
            raise RuntimeError('Source specification hash differs from manifest')
    def source_path(record):
        candidate=Path(record['path'])
        if candidate.is_absolute():
            candidate=candidate.resolve()
            if not candidate.is_relative_to(input_root):raise ValueError('Manifest source path is outside input directory')
            return candidate
        if '..' in candidate.parts:raise ValueError('Manifest source path must stay within input directory')
        return input_root/candidate
    centres=[250,750,1000,1250,1750] if args.smoke else list(range(1,2000))
    dyads=[1] if args.smoke else [1,2,3,4,*range(6,17)]
    rois=[0] if args.smoke else list(range(68));out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    spec=dict(mode='smoke' if args.smoke else 'full',dyads=dyads,roi_indices=rois,centres=centres,cells=CELLS,bands=BANDS,atoms=ATOMS,geometries=['homologous','cross_hemisphere'],epoch_batch=8,workers=1,min_trials=10,offset_samples=1,past_future_gap_ms=2,transform='float32 mean-frequency raw power then log1p',mapping='suffix1 X=P1; suffix2 X=P2, per epoch',final_valid_trial='included',regularization='none; fail on estimator error',manifest_sha256=digest(manifest_path),runner_sha256=digest(Path(__file__)),hashing_module_sha256=digest(HERE/'prepare_inputs.py'),package_hashes={p.name:digest(p) for p in Path(calc.__file__).parent.glob('*.py')},versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mne=mne.__version__,h5py=h5py.__version__))
    sig=signature(spec);specpath=out/'specification.json'
    if specpath.exists():
        if canonical(json.loads(specpath.read_text()))!=canonical(spec):raise RuntimeError('Existing output specification differs')
    else:atomic_json(specpath,spec)
    actions=[]
    for d in dyads:
        sources=sorted([r for r in manifest['inputs'] if r['dyad']==d],key=lambda x:x['partner'])
        for r in sources:
            if digest(source_path(r))!=r['sha256']:raise RuntimeError('Source content changed: '+r['path'])
        with h5py.File(source_path(sources[0]),'r') as a,h5py.File(source_path(sources[1]),'r') as b:
            for k in ['events','selection','event_labels','roi_labels','times_seconds']:np.testing.assert_array_equal(a[k][:],b[k][:])
            assert np.allclose(a['times_seconds'][:],np.arange(-1000,1001)/1000)
            assert all(e in ['LMR1','FMR1','LMA1','FMA1','LMA2','FMA2','LMR2','FMR2'] for e in a['event_labels'].asstr()[:])
            for roi in rois:
                path=out/f'dyad{d}_roi{roi:02d}.h5';shape=(2,4,6,len(centres),4)
                if checkpoint_ok(path,sig,shape):actions.append(dict(dyad=d,roi=roi,action='verified_resume'));continue
                start=time.perf_counter();values,meta=compute_tile(a,b,roi,centres)
                publish(path,values,sig,meta);actions.append(dict(dyad=d,roi=roi,action='computed',seconds=time.perf_counter()-start))
                print(actions[-1],flush=True)
        # Per-dyad completion indexes independently validated ROI tiles, avoiding duplication.
        for roi in rois:assert checkpoint_ok(out/f'dyad{d}_roi{roi:02d}.h5',sig,(2,4,6,len(centres),4))
        atomic_json(out/f'dyad{d}_complete.json',dict(signature=sig,roi_tiles=rois,complete=True))
    atomic_json(out/'last_execution.json',dict(actions=actions,complete=True,signature=sig))
    print('COMPLETE',args.smoke,flush=True)
if __name__=='__main__':main()
