"""Clinical production adapter. Explicit actor manifest; no output overwrites."""
import os,sys,ast,json,time,hashlib,uuid,argparse,concurrent.futures,multiprocessing
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'cache')
sys.path.insert(0,str(WORK.parent/'phiid'))
import numpy as np,h5py,mne,scipy
from threadpoolctl import threadpool_limits
from tetradic_phiid.core import actor_rows,verify_backend,digest
from tetradic_phiid.vendor.phyid import calculate as calc
threadpool_limits(1)
SOURCE=HERE/'source_functions.py'
SOURCE_HASH='a463555fe6affdf6e0b6aeabcb6cc4428bef4a066f2a16e642b25b92c808b2cb'
if digest(SOURCE)!=SOURCE_HASH:raise RuntimeError('Frozen numerical source changed')
CELLS=['LMR','LMA','FMA','FMR'];ATOMS=['rtr','xtx','yty','sts']
FREQS=np.arange(4.,46.);BANDS=[(4,7),(8,9),(10,12),(13,19),(20,29),(30,45)]
names=['canonical','signature','atomic_json','checkpoint_ok','publish','estimate','load_power']
functions=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(functions)==len(names)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),globals())

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

def verify(spec):
    assert digest(Path(__file__))==spec['runner_sha256']
    assert digest(HERE/'input_manifest.json')==spec['input_manifest_sha256']
    assert verify_backend()==spec['backend_hashes']
    for path,checksum in spec['dependency_hashes'].items():assert digest(path)==checksum,path
    for name,module in [('numpy',np),('scipy',scipy),('mne',mne),('h5py',h5py)]:assert module.__version__==spec['versions'][name]

def worker(spec,out,d,roi,sources):
    verify(spec);out=Path(out);sig=signature(spec);shape=(2,4,6,len(spec['centres']),4)
    path=out/f'dyad{d}_roi{roi:02d}.h5'
    if checkpoint_ok(path,sig,shape):return dict(dyad=d,roi=roi,action='verified_resume')
    claim=path.with_suffix('.claim');claim.open('x').close();start=time.perf_counter()
    try:
        with h5py.File(sources[0]['path'],'r') as a,h5py.File(sources[1]['path'],'r') as b:
            values,meta=compute_tile(a,b,roi,spec['centres'],spec['mapping'])
        publish(path,values,sig,meta)
        return dict(dyad=d,roi=roi,action='computed',seconds=time.perf_counter()-start)
    finally:claim.unlink()

def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke',action='store_true');g.add_argument('--run',action='store_true');args=parser.parse_args()
    spec=json.loads((HERE/('smoke_specification.json' if args.smoke else 'specification.json')).read_text());verify(spec)
    out=HERE/('smoke' if args.smoke else 'production');out.mkdir(exist_ok=True)
    if (out/'specification.json').exists():assert json.loads((out/'specification.json').read_text())==spec
    else:atomic_json(out/'specification.json',spec)
    manifest=json.loads((HERE/'input_manifest.json').read_text());sig=signature(spec);shape=(2,4,6,len(spec['centres']),4)
    actions=[];start=time.perf_counter();lock=out/'scheduler.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            for d in spec['dyads']:
                sources=sorted([r for r in manifest if r['dyad']==d],key=lambda r:r['participant'])
                for r in sources:assert digest(r['path'])==r['sha256'],'Input changed: '+r['path']
                pending=[]
                for roi in spec['roi_indices']:
                    path=out/f'dyad{d}_roi{roi:02d}.h5'
                    if checkpoint_ok(path,sig,shape):actions.append(dict(dyad=d,roi=roi,action='verified_resume'))
                    else:pending.append(pool.submit(worker,spec,str(out),d,roi,sources))
                for future in concurrent.futures.as_completed(pending):
                    result=future.result();actions.append(result);print(result,flush=True)
                    atomic_json(out/'parallel_status.json',dict(state='running',actions=actions,elapsed_seconds=time.perf_counter()-start))
                for roi in spec['roi_indices']:assert checkpoint_ok(out/f'dyad{d}_roi{roi:02d}.h5',sig,shape)
                index=dict(signature=sig,roi_tiles=spec['roi_indices'],complete=True)
                indexpath=out/f'dyad{d}_complete.json'
                if indexpath.exists():assert json.loads(indexpath.read_text())==index
                else:atomic_json(indexpath,index)
        record=dict(state='complete',actions=actions,elapsed_seconds=time.perf_counter()-start,signature=sig)
        atomic_json(out/'parallel_status.json',record)
        atomic_json(out/f'execution_{time.time_ns()}.json',record)
        print('COMPLETE',flush=True)
    except BaseException as e:
        atomic_json(out/'parallel_status.json',dict(state='failed',error=repr(e),actions=actions));raise
    finally:lock.unlink()
if __name__=='__main__':main()
