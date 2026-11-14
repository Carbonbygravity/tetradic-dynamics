"""Normative PhiID inference: explicit full-family correction, resumable null batches."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time,argparse,uuid,platform
from pathlib import Path
import numpy as np,h5py,scipy
from scipy import stats
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(WORK/'PhiID_Inference_Control_Plan_2026-10-26'))
import inference_core as core
SOURCE=WORK/'PhiID_Production_Preflight_2026-10-25/production'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def save(path,obj):
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temp.open('x') as f:json.dump(obj,f,indent=2)
    temp.rename(path)
def status(obj):
    temp=ROOT/'status.partial.json';temp.write_text(json.dumps(obj,indent=2));os.replace(temp,ROOT/'status.json')

def prepare():
    spec=json.loads((SOURCE/'specification.json').read_text());sig=hashlib.sha256(canonical(spec).encode()).hexdigest()
    assert spec['cells']==list(core.CELL_ORDER)
    n=len(spec['dyads']);shape=(n,68*2*6*4,1999)
    arrays={name:np.lib.format.open_memmap(ROOT/(name+'.partial.npy'),mode='w+',dtype='float64',shape=shape) for name in core.PRIMARY}
    inputs=[];maps=[];checks=0
    for di,d in enumerate(spec['dyads']):
        index=json.loads((SOURCE/f'dyad{d}_complete.json').read_text());assert index['complete'] and index['signature']==sig and index['roi_tiles']==list(range(68))
        for roi in range(68):
            path=SOURCE/f'dyad{d}_roi{roi:02d}.h5';marker=json.loads(path.with_suffix('.complete.json').read_text());sha=digest(path)
            assert sha==marker['sha256'] and marker['signature']==sig
            with h5py.File(path,'r') as f:
                v=f['values'][:];meta=json.loads(f.attrs['metadata']);assert f.attrs['signature']==sig
            assert v.shape==(2,4,6,1999,4) and np.isfinite(v).all()
            inputs.append(dict(dyad=d,roi=roi,path=str(path),sha256=sha))
            if di==0:
                for g in range(2):
                    for b in range(6):
                        for a in range(4):maps.append(dict(index=len(maps),roi=meta['roi'],follower_roi=meta['roi'] if g==0 else meta['opposite_roi'],geometry=spec['geometries'][g],band=spec['bands'][b],atom=spec['atoms'][a]))
            else:
                assert maps[roi*48]['roi']==meta['roi'] and maps[roi*48+24]['follower_roi']==meta['opposite_roi']
            for name in core.PRIMARY:
                coeff=core.CONTRASTS[name]
                values=np.einsum('c,gcbta->gbat',coeff,v,optimize=False).reshape(48,1999)
                arrays[name][di,roi*48:(roi+1)*48]=values
                # Independent direct cell arithmetic checks every map and centre.
                if name=='Matching_minus_Mirroring':direct=((v[:,1]+v[:,2])-(v[:,0]+v[:,3]))/2
                elif name=='Leader_period_minus_Follower_period':direct=((v[:,0]+v[:,1])-(v[:,2]+v[:,3]))/2
                else:direct=(v[:,1]-v[:,0])-(v[:,2]-v[:,3])
                np.testing.assert_allclose(values,direct.transpose(0,1,3,2).reshape(48,1999),rtol=1e-10,atol=1e-13);checks+=values.size
        print('ASSEMBLED',d,flush=True)
    files={}
    for name in core.PRIMARY:
        arrays[name].flush();arrays[name]._mmap.close()
        partial=ROOT/(name+'.partial.npy');final=ROOT/(name+'.npy');partial.rename(final);files[name]=dict(path=final.name,sha256=digest(final),shape=list(shape))
    save(ROOT/'map_inventory.json',maps);save(ROOT/'source_inventory.json',inputs)
    signs=core.sign_assignments(n,5000,7);np.save(ROOT/'sign_assignments.npy',signs)
    config=dict(cohort='normative',dyads=spec['dyads'],contrasts={name:core.CONTRASTS[name].tolist() for name in core.PRIMARY},cell_order=list(core.CELL_ORDER),
        family='All 3264 ROI/geometry/band/atom maps per contrast; temporal adjacency only',comparisons='Historical per-map correction retained using same sign assignments',
        forming_p=.05,two_sided=True,permutations=5000,seed=7,batch_size=25,map_chunk=96,worker_processes=1,
        times_ms=list(range(-999,1000)),array_files=files,source_inventory_sha256=digest(ROOT/'source_inventory.json'),map_inventory_sha256=digest(ROOT/'map_inventory.json'),
        signs_sha256=digest(ROOT/'sign_assignments.npy'),code_hashes={str(Path(__file__)):digest(__file__),str(Path(core.__file__)):digest(core.__file__)},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,h5py=h5py.__version__),
        inference_scope='Within normative cohort only. No clinical comparison, overlap or surrogate inference.',null_assumption='Dyad contrast vectors are jointly sign-exchangeable under the null')
    save(ROOT/'specification.json',config)
    save(ROOT/'assembly_validation.json',dict(status='passed',source_tiles=1020,maps=3264,shape=list(shape),independent_arithmetic_values_checked=checks,all_source_hashes_matched=True))

def verify(config):
    for path,sha in config['code_hashes'].items():assert digest(path)==sha
    for name,key in [('source_inventory.json','source_inventory_sha256'),('map_inventory.json','map_inventory_sha256'),('sign_assignments.npy','signs_sha256')]:assert digest(ROOT/name)==config[key]
    assert np.__version__==config['versions']['numpy'] and scipy.__version__==config['versions']['scipy']

def null_batch(x,signs,threshold):
    n,m,_=x.shape;null=np.zeros((len(signs),m))
    for lo in range(0,m,96):
        hi=min(lo+96,m);part=np.asarray(x[:,lo:hi]);sumsq=np.einsum('nmt,nmt->mt',part,part,optimize=False)
        for i,assignment in enumerate(signs):
            t=core.t_from_sums(np.einsum('n,nmt->mt',assignment,part,optimize=False),sumsq,n)
            for c in core.cluster_masses(t,threshold):null[i,lo+c['map']]=max(null[i,lo+c['map']],c['mass'])
    return null

def checkpoint(path,signature,expected_shape):
    marker=path.with_suffix('.complete.json')
    if not path.exists() and not marker.exists():return None
    if not path.exists() or not marker.exists():raise RuntimeError('Incomplete checkpoint: '+str(path))
    record=json.loads(marker.read_text());assert record['signature']==signature and record['sha256']==digest(path)
    v=np.load(path,allow_pickle=False);assert v.shape==tuple(expected_shape) and np.isfinite(v).all() and np.all(v>=0)
    return v
def publish(path,v,signature):
    if path.exists() or path.with_suffix('.complete.json').exists():raise FileExistsError(path)
    with path.with_suffix('.partial').open('xb') as f:np.save(f,v)
    path.with_suffix('.partial').rename(path)
    save(path.with_suffix('.complete.json'),dict(signature=signature,sha256=digest(path),shape=list(v.shape)))

def preflight(config):
    verify(config);signs=np.load(ROOT/'sign_assignments.npy')
    x=np.load(ROOT/config['array_files'][core.PRIMARY[0]]['path'],mmap_mode='r')
    threshold=stats.t.ppf(.975,len(x)-1)
    # Full-dimensional runtime check. No p values or observed findings reported.
    started=time.perf_counter();v=null_batch(x,signs[:2],threshold);elapsed=time.perf_counter()-started
    small=np.asarray(x[:,:3,:101]);reference=core.test_family(small,signs[:8])['null_map']
    np.testing.assert_allclose(null_batch(small,signs[:8],threshold),reference,rtol=1e-10,atol=1e-10)
    directory=ROOT/'preflight';directory.mkdir();path=directory/'null.npy';signature='preflight-'+digest(ROOT/'specification.json')
    publish(path,v,signature);sha=digest(path);resumed=checkpoint(path,signature,v.shape);np.testing.assert_array_equal(resumed,v);assert sha==digest(path)
    rejected=0
    for wrong_sig,wrong_shape in [('wrong',v.shape),(signature,(3,v.shape[1]))]:
        try:checkpoint(path,wrong_sig,wrong_shape)
        except AssertionError:rejected+=1
    assert rejected==2
    save(ROOT/'preflight_validation.json',dict(status='passed',full_family_assignments=2,seconds=elapsed,rough_three_contrast_seconds=elapsed/2*5000*3,
        bounded_null_matches_reference=True,checkpoint_resume_exact=True,wrong_signature_and_shape_rejected=True,
        scope='Runtime and correctness test; no study-data p values reported'))
    print('PREFLIGHT',elapsed,'seconds for 2 full-family assignments',flush=True)

def run(config):
    verify(config);assert json.loads((ROOT/'preflight_validation.json').read_text())['status']=='passed'
    sig=digest(ROOT/'specification.json');signs=np.load(ROOT/'sign_assignments.npy');maps=json.loads((ROOT/'map_inventory.json').read_text())
    lock=ROOT/'inference.lock';lock.open('x').write(str(os.getpid()));started=time.time()
    try:
        for name in core.PRIMARY:
            item=config['array_files'][name];assert digest(ROOT/item['path'])==item['sha256'];x=np.load(ROOT/item['path'],mmap_mode='r')
            out=ROOT/name;out.mkdir(exist_ok=True);threshold=stats.t.ppf(.975,len(x)-1);nulls=[]
            for lo in range(0,len(signs),25):
                hi=min(lo+25,len(signs));path=out/f'null_{lo:04d}_{hi:04d}.npy'
                v=checkpoint(path,sig,(hi-lo,len(maps)))
                action='verified_resume'
                if v is None:v=null_batch(x,signs[lo:hi],threshold);publish(path,v,sig);action='computed'
                nulls.append(v);status(dict(state='running',contrast=name,completed_assignments=hi,total_assignments=5000,action=action,elapsed_seconds=time.time()-started))
                print(name,hi,action,flush=True)
            null=np.concatenate(nulls);family=null.max(axis=1);clusters=[]
            for lo in range(0,len(maps),96):
                part=np.asarray(x[:,lo:lo+96]);t=core.t_from_sums(part.sum(0),(part*part).sum(0),len(x))
                for c in core.cluster_masses(t,threshold):
                    c['map']+=lo;cutoff=c['mass']-1e-12*max(1.,abs(c['mass']))
                    c.update(p_primary_family=float(np.mean(family>=cutoff)),p_historical_per_map=float(np.mean(null[:,c['map']]>=cutoff)),
                        start_ms=config['times_ms'][c['start']],end_ms=config['times_ms'][c['stop']-1],coordinates=maps[c['map']])
                    clusters.append(c)
            result=dict(contrast=name,signature=sig,clusters=clusters,primary_significant=sum(c['p_primary_family']<=.05 for c in clusters),historical_per_map_significant=sum(c['p_historical_per_map']<=.05 for c in clusters))
            p=out/'results.json'
            if p.exists():assert json.loads(p.read_text())==result
            else:save(p,result)
        status(dict(state='complete',contrasts=list(core.PRIMARY),elapsed_seconds=time.time()-started))
    except BaseException as error:status(dict(state='failed',error=repr(error)));raise
    finally:lock.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preflight','run']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        config=json.loads((ROOT/'specification.json').read_text())
        if args.stage=='preflight':preflight(config)
        else:run(config)
