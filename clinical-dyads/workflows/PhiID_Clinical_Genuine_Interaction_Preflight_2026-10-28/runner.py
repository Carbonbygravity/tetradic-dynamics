"""Separate clinical interaction. The full-run mode requires an explicit flag."""
import os,sys,json,time,hashlib,importlib.util,platform
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np,h5py,scipy,psutil
from scipy import stats
R=Path(__file__).resolve().parent;W=R.parent
sp=importlib.util.spec_from_file_location('clinical_matched',W/'PhiID_Clinical_Matched_PerMap_2026-10-26/runner.py');c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
e=c.engine;core=c.core;SOURCE=c.SOURCE;COEF=np.array([-1.,1.,-1.,1.])
read=lambda p:json.loads(p.read_text())
def save(n,v):e.save(R/n,v)
def log(s):
 line=time.strftime('%Y-%m-%d %H:%M:%S')+' — '+s
 with (R/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(line+'\n')
 print(line,flush=True)
def config():return read(R/'specification.json')
def verify(cfg):
 for p,h in cfg['hashes'].items():assert e.digest(p)==h,p
def source_gate():
 spec=read(SOURCE/'specification.json');sig=hashlib.sha256(e.canonical(spec).encode()).hexdigest()
 assert spec['dyads']==list(range(1,16)) and spec['cells']==list(core.CELL_ORDER)
 for d in spec['dyads']:
  ix=read(SOURCE/f'dyad{d}_complete.json');assert ix['complete'] and ix['signature']==sig and ix['roi_tiles']==list(range(68))
 return sig
def contrast(v):return np.einsum('c,gcbta->gbat',COEF,v,optimize=False).reshape(48,1999)
def direct(v):return ((v[:,1]-v[:,2])-(v[:,0]-v[:,3])).transpose(0,1,3,2).reshape(48,1999)
def tile(d,roi,sig,maps):
 p=SOURCE/f'dyad{d}_roi{roi:02d}.h5';v,meta,h=c.validate_tile(p,sig)
 assert maps[roi*48]['roi']==meta['roi'] and maps[roi*48+24]['follower_roi']==meta['opposite_roi']
 x=contrast(v);np.testing.assert_allclose(x,direct(v),rtol=1e-10,atol=1e-12)
 return x,dict(path=str(p),sha256=h)
def setup():
 deps=[Path(__file__),Path(c.__file__),Path(e.__file__),Path(core.__file__),SOURCE/'specification.json',W/'PhiID_Clinical_Matched_PerMap_2026-10-26/map_inventory.json',W/'PhiID_Clinical_Matched_PerMap_2026-10-26/sign_assignments.npy',W/'PhiID_Clinical_Controls_2026-10-27/directional_tiers.json']
 cfg=dict(state='prepared',contrast='Role_by_Condition',coefficients=COEF.tolist(),cell_order=list(core.CELL_ORDER),dyads=list(range(1,16)),shape=[15,3264,1999],forming_p=.05,permutations=5000,seed=7,batch=25,workers=1,threads=1,cluster_criterion='p_within_map <= .05, matching existing clinical and normative implementation',maps=str(deps[5]),signs=str(deps[6]),hashes={str(p):e.digest(p) for p in deps},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,h5py=h5py.__version__),benchmark_rois=[0,33,67],benchmark_signs=25,full_run_authorized=False)
 save('specification.json',cfg);log('Prepared the response-period × condition interaction specification.')
def benchmark():
 cfg=config();verify(cfg);sig=source_gate();maps=read(Path(cfg['maps']));signs=np.load(cfg['signs']);np.testing.assert_array_equal(signs,core.sign_assignments(15,5000,7))
 assert len(maps)==3264 and len(np.unique(signs,axis=0))==5000
 rng=np.random.default_rng(20261028);toy=rng.normal(size=(2,4,6,1999,4));np.testing.assert_allclose(contrast(toy),direct(toy),atol=1e-12)
 # A pure pooled period effect must give zero interaction.
 toy[:,0]=2;toy[:,1]=2;toy[:,2]=0;toy[:,3]=0;assert np.max(abs(contrast(toy)))==0
 records=[];sources=[];null_errors=[];stat_errors=[];load_seconds=0;first=None
 for roi in cfg['benchmark_rois']:
  ts=time.perf_counter();values=[]
  for d in cfg['dyads']:
   x,record=tile(d,roi,sig,maps);values.append(x);sources.append(record)
  x=np.array(values);load=time.perf_counter()-ts;load_seconds+=load;threshold=stats.t.ppf(.975,14)
  tt=core.t_from_sums(x.sum(0),(x*x).sum(0),15);ref=stats.ttest_1samp(x,0,axis=0).statistic;np.testing.assert_allclose(tt,ref,rtol=1e-9,atol=1e-10);stat_errors.append(float(abs(tt-ref).max()))
  small=x[:,[0,24],:101];reference=core.test_family(small,signs[:8])['null_map'];got=e.null_batch(small,signs[:8],threshold);np.testing.assert_allclose(got,reference,atol=1e-10,rtol=1e-10);null_errors.append(float(abs(got-reference).max()))
  ts=time.perf_counter();null=e.null_batch(x,signs[:25],threshold);seconds=time.perf_counter()-ts
  p=R/f'benchmark_roi{roi:02d}.npy';e.publish(p,null,e.digest(R/'specification.json'));np.testing.assert_array_equal(e.checkpoint(p,e.digest(R/'specification.json'),(25,48)),null)
  if first is None:first=p
  records.append(dict(roi=roi,maps=48,time_points=1999,dyads=15,assignments=25,load_verify_assemble_seconds=load,null_seconds=seconds,rss_bytes=psutil.Process().memory_info().rss));log(f'Bounded ROI {roi}: 25 assignments, {seconds:.2f} seconds; checkpoint saved.')
 rejected=0
 for signature,shape in [('wrong',(25,48)),(e.digest(R/'specification.json'),(24,48))]:
  try:e.checkpoint(first,signature,shape)
  except AssertionError:rejected+=1
 assert rejected==2;verify(cfg)
 seconds=[x['null_seconds'] for x in records];factor=68*5000/25
 result=dict(status='passed',source_tiles_checked=45,rois=cfg['benchmark_rois'],geometry_band_atom_maps=144,assignments_per_roi=25,statistic_max_error=max(stat_errors),null_max_error=max(null_errors),checkpoint_reload_exact=True,bad_signature_shape_rejected=True,records=records,runtime_estimate_hours=dict(null_min=min(seconds)*factor/3600,null_mean=np.mean(seconds)*factor/3600,null_max=max(seconds)*factor/3600,source_load_verify_assembly=load_seconds/3*68/3600),scope='Three ROIs, all 15 dyads, both geometries, six bands, four atoms and full time axis. Linear timing scenarios, not confidence bounds; full 1020-tile verification and final output overhead remain. No full interaction inference, overlap, shuffle controls or tiering.',full_run_launched=False)
 save('benchmark_source_inventory.json',sources);save('benchmark_results.json',result);log('Bounded benchmark passed; full run remains unlaunched.')
def resume_check():
 cfg=config();verify(cfg);records=[]
 for roi in cfg['benchmark_rois']:
  p=R/f'benchmark_roi{roi:02d}.npy';h=e.digest(p);v=e.checkpoint(p,e.digest(R/'specification.json'),(25,48));assert v is not None and e.digest(p)==h;records.append(dict(roi=roi,sha256=h))
 save('fresh_process_resume_verification.json',dict(status='passed',reused=3,recomputed=0,records=records));log('Fresh-process checkpoint reload passed; no calculations repeated.')
def assemble(cfg):
 sig=source_gate();maps=read(Path(cfg['maps']));p=R/'interaction.partial.npy';assert not p.exists() and not (R/'interaction.npy').exists()
 x=np.lib.format.open_memmap(p,mode='w+',dtype='float64',shape=tuple(cfg['shape']));sources=[]
 for di,d in enumerate(cfg['dyads']):
  for roi in range(68):
   v,record=tile(d,roi,sig,maps);x[di,roi*48:(roi+1)*48]=v;sources.append(record)
  log(f'Full source verification and interaction assembly: dyad {d}.')
 x.flush();x._mmap.close();p.rename(R/'interaction.npy');save('assembly.json',dict(status='passed',array_sha256=e.digest(R/'interaction.npy'),spec_sha256=e.digest(R/'specification.json'),sources=sources))
def run():
 assert '--approved-full-run' in sys.argv,'Pass --approved-full-run to select the full-run mode'
 cfg=config();verify(cfg);assert read(R/'benchmark_results.json')['status']=='passed' and read(R/'fresh_process_resume_verification.json')['status']=='passed'
 lock=R/'run.lock';lock.open('x').write(str(os.getpid()))
 try:
  if not (R/'assembly.json').exists():assemble(cfg)
  a=read(R/'assembly.json');assert a['spec_sha256']==e.digest(R/'specification.json') and a['array_sha256']==e.digest(R/'interaction.npy')
  x=np.load(R/'interaction.npy',mmap_mode='r');signs=np.load(cfg['signs']);maps=read(Path(cfg['maps']));sig=e.digest(R/'assembly.json');threshold=stats.t.ppf(.975,14);out=R/'null_checkpoints';out.mkdir(exist_ok=True);chunks=[]
  for lo in range(0,5000,25):
   p=out/f'null_{lo:04d}.npy';v=e.checkpoint(p,sig,(25,3264))
   if v is None:v=e.null_batch(x,signs[lo:lo+25],threshold);e.publish(p,v,sig)
   chunks.append(v);log(f'Interaction assignments {lo+25}/5000')
  null=np.concatenate(chunks);clusters=[]
  for lo in range(0,3264,96):
   z=np.asarray(x[:,lo:lo+96]);tt=core.t_from_sums(z.sum(0),(z*z).sum(0),15);np.testing.assert_allclose(tt,stats.ttest_1samp(z,0,axis=0).statistic,rtol=1e-9,atol=1e-10)
   for cl in core.cluster_masses(tt,threshold):
    cl['map']+=lo;cut=cl['mass']-1e-12*max(1.,abs(cl['mass']));cl.update(p_within_map=float(np.mean(null[:,cl['map']]>=cut)),start_ms=cl['start']-999,end_ms=cl['stop']-1000,coordinates=maps[cl['map']]);clusters.append(cl)
  save('results.json',dict(contrast='Role_by_Condition',clusters=clusters,significant_count=sum(v['p_within_map']<=.05 for v in clusters)));verify(cfg)
  save('completion_verification.json',dict(state='complete',source_tiles=1020,null_checkpoints=200,assignments=5000,primary_four_contrast_results_unchanged=True));log('Separate interaction inference complete. Overlap and controls not launched.')
 finally:lock.unlink()
if __name__=='__main__':
 {'setup':setup,'benchmark':benchmark,'resume-check':resume_check,'run':run}[sys.argv[1]]()
