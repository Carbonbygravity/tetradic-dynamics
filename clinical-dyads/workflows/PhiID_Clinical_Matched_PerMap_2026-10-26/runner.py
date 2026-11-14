"""Prepared clinical matched per-map tests. No automatic launch or input mutations."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,shutil,platform
from pathlib import Path
import numpy as np,h5py,scipy
from scipy import stats
ROOT=Path(__file__).resolve().parent;W=ROOT.parent
sys.path.insert(0,str(W/'PhiID_Normative_Inference_2026-10-26'))
import pipeline as engine
core=engine.core
SOURCE=W/'PhiID_Clinical_Production_Preflight_2026-10-26/production'
NORM=Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/'PhiID_Matched_PerMap_2026-10-26'
NAMES=('Matching_minus_Mirroring','Leader_minus_Follower_Mirroring','Leader_minus_Follower_Matching','Leader_period_minus_Follower_period')
EXPECTED={NAMES[0]:[-.5,.5,.5,-.5],NAMES[1]:[1,0,0,-1],NAMES[2]:[0,1,-1,0],NAMES[3]:[.5,.5,-.5,-.5]}
def save(p,v):engine.save(p,v)
def log(s):
 line=time.strftime('%Y-%m-%d %H:%M:%S')+' — '+s;print(line,flush=True)
 with (ROOT/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(line+'\n')
def direct(v,name):
 if name==NAMES[0]:return ((v[:,1]+v[:,2])-(v[:,0]+v[:,3]))/2
 if name==NAMES[1]:return v[:,0]-v[:,3]
 if name==NAMES[2]:return v[:,1]-v[:,2]
 if name==NAMES[3]:return ((v[:,0]+v[:,1])-(v[:,2]+v[:,3]))/2
 raise ValueError(name)
def setup():
 original=json.loads((SOURCE/'specification.json').read_text());norm=json.loads((NORM/'specification.json').read_text())
 assert original['dyads']==list(range(1,16));assert original['cells']==list(core.CELL_ORDER);assert original['centres']==list(range(1,2000))
 for n in NAMES:np.testing.assert_array_equal(core.CONTRASTS[n],EXPECTED[n])
 record=dict(cohort='clinical HC-PT',unit='dyad; two actors jointly enter each PhiID estimate',dyads=original['dyads'],contrasts=EXPECTED,cell_order=list(core.CELL_ORDER),source=str(SOURCE),source_spec_sha256=engine.digest(SOURCE/'specification.json'),geometries=original['geometries'],bands=original['bands'],atoms=original['atoms'],shape=[15,3264,1999],times_ms=list(range(-999,1000)),forming_p=.05,two_sided=True,permutations=5000,seed=7,batch_size=25,map_chunk=96,workers=1,correction='Temporal cluster maximum within each ROI/geometry/band/atom map',genuine_interaction='Excluded from this runner; no reuse as a matched target',normative_map_path=str(NORM/'map_inventory.json'),normative_map_sha256=engine.digest(NORM/'map_inventory.json'),normative_signs_path=str(NORM/'sign_assignments.npy'),normative_signs_sha256=engine.digest(NORM/'sign_assignments.npy'),code_hashes={str(p):engine.digest(p) for p in [Path(__file__),Path(engine.__file__),Path(core.__file__)]},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,h5py=h5py.__version__),scope='Prepared only. No automatic launch. Within clinical cohort matched tests, not direct normative-clinical comparison.',null_assumption='Dyad contrast vectors sign-exchangeable under the null',source_gate='Require all 15 complete dyad indexes and 1020 tiles/markers, then verify every hash/signature/shape/finite value before inference')
 for k in ['forming_p','two_sided','permutations','seed','batch_size','map_chunk']:assert record[k]==norm[k]
 save(ROOT/'plan.json',record);log('Prepared four-contrast clinical plan matching normative settings. Clinical estimation and normative controls untouched.')
def verify_plan(plan):
 for p,h in plan['code_hashes'].items():assert engine.digest(p)==h
 assert engine.digest(SOURCE/'specification.json')==plan['source_spec_sha256']
 assert engine.digest(plan['normative_map_path'])==plan['normative_map_sha256']
 assert engine.digest(plan['normative_signs_path'])==plan['normative_signs_sha256']
 assert np.__version__==plan['versions']['numpy'] and scipy.__version__==plan['versions']['scipy']
def completion_gate(plan):
 verify_plan(plan);source=json.loads((SOURCE/'specification.json').read_text());sig=engine.signature(source) if hasattr(engine,'signature') else __import__('hashlib').sha256(engine.canonical(source).encode()).hexdigest();missing=[]
 for d in plan['dyads']:
  p=SOURCE/f'dyad{d}_complete.json'
  if not p.exists():missing.append(p.name);continue
  ix=json.loads(p.read_text());assert ix['complete'] and ix['signature']==sig and ix['roi_tiles']==list(range(68))
  for roi in range(68):
   tile=SOURCE/f'dyad{d}_roi{roi:02d}.h5'
   for p in [tile,tile.with_suffix('.complete.json')]:
    if not p.exists():missing.append(p.name)
 if missing:raise RuntimeError(f'Clinical estimates incomplete: {len(missing)} missing completion/index files; no arrays or inference started. Examples: {missing[:5]}')
 return sig
def validate_tile(path,sig):
 marker=json.loads(path.with_suffix('.complete.json').read_text());assert marker['signature']==sig and engine.digest(path)==marker['sha256']
 with h5py.File(path,'r') as f:
  assert f.attrs['signature']==sig;v=f['values'][:];meta=json.loads(f.attrs['metadata'])
 assert v.shape==(2,4,6,1999,4) and np.isfinite(v).all()
 return v,meta,marker['sha256']
def prepare(plan):
 sig=completion_gate(plan) # This check precedes every output array allocation.
 maps=json.loads(Path(plan['normative_map_path']).read_text());arrays={n:np.lib.format.open_memmap(ROOT/(n+'.partial.npy'),mode='w+',dtype='float64',shape=tuple(plan['shape'])) for n in NAMES};inputs=[]
 for di,d in enumerate(plan['dyads']):
  for roi in range(68):
   path=SOURCE/f'dyad{d}_roi{roi:02d}.h5';v,meta,sha=validate_tile(path,sig)
   assert maps[roi*48]['roi']==meta['roi'] and maps[roi*48+24]['follower_roi']==meta['opposite_roi']
   inputs.append(dict(dyad=d,roi=roi,path=str(path),sha256=sha))
   for n in NAMES:
    calculated=np.einsum('c,gcbta->gbat',np.array(EXPECTED[n]),v,optimize=False).reshape(48,1999)
    reference=direct(v,n).transpose(0,1,3,2).reshape(48,1999);np.testing.assert_allclose(calculated,reference,rtol=1e-10,atol=1e-12);arrays[n][di,roi*48:(roi+1)*48]=calculated
  log(f'Verified and assembled clinical dyad {d}.')
 files={}
 for n,x in arrays.items():
  x.flush();x._mmap.close();p=ROOT/(n+'.partial.npy');final=ROOT/(n+'.npy');p.rename(final);files[n]=dict(path=final.name,sha256=engine.digest(final))
 for source,name in [(plan['normative_map_path'],'map_inventory.json'),(plan['normative_signs_path'],'sign_assignments.npy')]:
  assert not (ROOT/name).exists();shutil.copyfile(source,ROOT/name)
 save(ROOT/'source_inventory.json',inputs);save(ROOT/'assembly.json',dict(status='passed',source_tiles=1020,dyads=15,source_signature=sig,array_files=files,plan_sha256=engine.digest(ROOT/'plan.json'),maps_sha256=engine.digest(ROOT/'map_inventory.json'),signs_sha256=engine.digest(ROOT/'sign_assignments.npy'),source_inventory_sha256=engine.digest(ROOT/'source_inventory.json')))
 log('Full clinical inputs verified and four contrasts assembled. No permutations launched by prepare.')
def test(plan):
 verify_plan(plan);passed=[]
 rng=np.random.default_rng(441);toy=rng.normal(size=(2,4,6,19,4))
 for name,coeff in EXPECTED.items():
  np.testing.assert_allclose(np.einsum('c,gcbta->gbat',coeff,toy),direct(toy,name).transpose(0,1,3,2),atol=1e-14);passed.append('formula '+name)
 signs=np.load(plan['normative_signs_path']);np.testing.assert_array_equal(signs,core.sign_assignments(15,5000,7));assert len(np.unique(signs,axis=0))==5000;passed.append('identical 5000 normative sign assignments')
 # Small matched normative fixture checks the exact shared null implementation.
 norm=np.load(NORM/'Leader_minus_Follower_Mirroring.npy',mmap_mode='r');small=np.asarray(norm[:,:3,:101]);th=stats.t.ppf(.975,14)
 np.testing.assert_allclose(engine.null_batch(small,signs[:8],th),core.test_family(small,signs[:8])['null_map'],atol=1e-10,rtol=1e-10);passed.append('bounded null implementation agreement')
 p=ROOT/'test_checkpoint.npy';sig='preflight';val=np.arange(12.).reshape(3,4);engine.publish(p,val,sig);np.testing.assert_array_equal(engine.checkpoint(p,sig,val.shape),val)
 rejects=0
 for s,shape in [('wrong',val.shape),(sig,(4,3))]:
  try:engine.checkpoint(p,s,shape)
  except AssertionError:rejects+=1
 assert rejects==2;passed.append('checkpoint exact resume and rejection safeguards')
 # Read one already-completed tile only, without treating it as full cohort readiness.
 source=json.loads((SOURCE/'specification.json').read_text());sig=__import__('hashlib').sha256(engine.canonical(source).encode()).hexdigest();v,meta,_=validate_tile(SOURCE/'dyad1_roi00.h5',sig)
 for n in NAMES:np.testing.assert_allclose(np.einsum('c,gcbta->gbat',EXPECTED[n],v),direct(v,n).transpose(0,1,3,2),rtol=1e-10,atol=1e-12)
 passed.append('four contrasts on completed clinical dyad1 ROI0 tile')
 try:completion_gate(plan);gate='All completion indexes present; full hash verification still required in prepare'
 except RuntimeError as error:gate=str(error);passed.append('incomplete clinical production refused before array allocation')
 save(ROOT/'preparation_tests.json',dict(status='passed',checks=passed,completion_gate=gate,scope='Synthetic contrasts, bounded normative null fixture and one complete clinical tile. No clinical group inference or full-cohort verification performed.'))
 log('Preparation tests passed; clinical matched inference remains unlaunched.')
def run(plan):
 verify_plan(plan);completion_gate(plan);assert json.loads((ROOT/'preparation_tests.json').read_text())['status']=='passed'
 assembly=json.loads((ROOT/'assembly.json').read_text());assert assembly['status']=='passed' and assembly['plan_sha256']==engine.digest(ROOT/'plan.json')
 for name,key in [('map_inventory.json','maps_sha256'),('sign_assignments.npy','signs_sha256'),('source_inventory.json','source_inventory_sha256')]:assert engine.digest(ROOT/name)==assembly[key]
 sig=engine.digest(ROOT/'assembly.json');signs=np.load(ROOT/'sign_assignments.npy');maps=json.loads((ROOT/'map_inventory.json').read_text());engine.ROOT=ROOT;lock=ROOT/'inference.lock';lock.open('x').write(str(os.getpid()));start=time.time()
 try:
  for name in NAMES:
   item=assembly['array_files'][name];assert engine.digest(ROOT/item['path'])==item['sha256'];x=np.load(ROOT/item['path'],mmap_mode='r');out=ROOT/name;out.mkdir(exist_ok=True);threshold=stats.t.ppf(.975,14);null=[]
   for lo in range(0,5000,25):
    path=out/f'null_{lo:04d}_{lo+25:04d}.npy';v=engine.checkpoint(path,sig,(25,3264))
    if v is None:v=engine.null_batch(x,signs[lo:lo+25],threshold);engine.publish(path,v,sig)
    null.append(v);engine.status(dict(state='running',cohort='clinical',contrast=name,completed_assignments=lo+25,total_assignments=5000,elapsed_seconds=time.time()-start));print(name,lo+25,flush=True)
   null=np.concatenate(null);clusters=[]
   for lo in range(0,3264,96):
    z=np.asarray(x[:,lo:lo+96]);t=core.t_from_sums(z.sum(0),(z*z).sum(0),15);np.testing.assert_allclose(t,stats.ttest_1samp(z,0,axis=0).statistic,rtol=1e-9,atol=1e-10)
    for c in core.cluster_masses(t,threshold):
     c['map']+=lo;cut=c['mass']-1e-12*max(1.,abs(c['mass']));c.update(p_within_map=float(np.mean(null[:,c['map']]>=cut)),start_ms=plan['times_ms'][c['start']],end_ms=plan['times_ms'][c['stop']-1],coordinates=maps[c['map']]);clusters.append(c)
   result=dict(cohort='clinical HC-PT',contrast=name,signature=sig,correction=plan['correction'],clusters=clusters,significant_count=sum(c['p_within_map']<=.05 for c in clusters));p=out/'results.json'
   if p.exists():assert json.loads(p.read_text())==result
   else:save(p,result)
   log(f'Completed clinical {name}: {result["significant_count"]} per-map clusters.')
  engine.status(dict(state='complete',cohort='clinical',contrasts=list(NAMES),genuine_interaction='excluded',elapsed_seconds=time.time()-start));log('Clinical four-contrast per-map stage complete; overlap and controls remain separate stages.')
 except BaseException as error:engine.status(dict(state='failed',error=repr(error)));log('Stopped on error: '+repr(error));raise
 finally:lock.unlink()
if __name__=='__main__':
 stage=sys.argv[1]
 if stage=='setup':setup()
 else:
  plan=json.loads((ROOT/'plan.json').read_text())
  if stage=='test':test(plan)
  elif stage=='check':print(completion_gate(plan))
  elif stage=='prepare':prepare(plan)
  elif stage=='run':run(plan)
  else:raise ValueError(stage)
