"""Authorized dyad-1 benchmark only. No all-dyad execution mode."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;PRE=W/'PhiID_Matched_Followup_Preflight_2026-10-27'
# The helper contains the same `variables` function definition as the
# inspected preflight, without executing the preflight's top-level body.
from tau_helpers import variables
sys.path.insert(0,str(W/'PhiID_Normative_Controls_2026-10-26'))
import control_core as c
import importlib.util
sp=importlib.util.spec_from_file_location('saved_controls',W/'PhiID_Normative_Controls_2026-10-26/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
import numpy as np,h5py
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
targets=read(PRE/'frozen_targets.json');assert len(targets)==19
taus=[1,2,4,8,16,32];manifest=read(r.PROD/'input_manifest.json')
dependencies=[Path(__file__),Path(sys.modules[variables.__module__].__file__),PRE/'check_baseline.py',PRE/'frozen_targets.json',Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),Path(c.common.__file__),r.PROD/'input_manifest.json',*Path(c.prod.calc.__file__).parent.glob('*.py')]
spec=dict(dyad=1,targets=19,taus=taus,H=1,surrogates=100,seed=1337,workers=1,batch=10,benchmark_time_controls='all targets, irrespective of cohort gate, to measure worst-case cost; no cohort p values or tiers',centres='original window intersect indices 32..1968 inclusive',hashes={str(p):sha(p) for p in dependencies})
sig=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
if (ROOT/'specification.json').exists():assert read(ROOT/'specification.json')==spec
else:save(ROOT/'specification.json',spec)
OUT=ROOT/'checkpoints';OUT.mkdir(exist_ok=True)
def checkpoint(path,shape):
 marker=path.with_suffix('.complete.json')
 if not path.exists() and not marker.exists():return None
 assert path.exists() and marker.exists(),'Incomplete checkpoint'
 m=read(marker);assert m['signature']==sig and m['sha256']==sha(path) and m['shape']==list(shape)
 a=np.load(path,allow_pickle=False);assert a.shape==shape and np.isfinite(a).all();return a
def publish(path,a,seconds):
 assert not path.exists() and not path.with_suffix('.complete.json').exists()
 tmp=path.with_suffix('.partial')
 with tmp.open('xb') as f:np.save(f,a);f.flush();os.fsync(f.fileno())
 tmp.rename(path);save(path.with_suffix('.complete.json'),dict(signature=sig,sha256=sha(path),shape=list(a.shape),seconds=seconds))
 np.testing.assert_array_equal(checkpoint(path,a.shape),a)
def evaluate(p1,p2,ev,rows,rep,kind):
 perm,lag=c.perturbation(ev,1,rep,kind)
 blocks,periods=c.blocks_periods(ev)
 if kind=='trial':
  assert sorted(perm.tolist())==list(range(len(ev)))
  np.testing.assert_array_equal(blocks,blocks[perm]);np.testing.assert_array_equal(periods,periods[perm])
 result=np.empty((len(rows),len(taus)))
 for ti,t in enumerate(rows):
  centres=np.arange(t['start']+1,t['stop']+1);centres=centres[(centres>=32)&(centres<=1968)]
  assert len(centres)>=3
  for ki,tau in enumerate(taus):
   assert centres.min()-tau>=0 and centres.max()+tau<2001
   val=0.
   for ci,coef in enumerate(r.COEFF[t['contrast']]):
    if coef==0:continue
    v=variables(p1,p2,ev,c.prod.CELLS[ci],int(t['geometry']=='cross_hemisphere'),t['band_index'],centres,tau,perm,lag)
    a=np.concatenate([c.estimate_batch(v[k:k+64]) for k in range(0,len(v),64)])
    val+=coef*a[:,t['atom_index']].mean()
   result[ti,ki]=val
 return result
ap=argparse.ArgumentParser();ap.add_argument('--stop-after-batches',type=int,default=0);args=ap.parse_args()
start=time.perf_counter();new=0;reused=0;loads=[];errors=[];replay=None
prior=read(ROOT/'planned_stop.json') if (ROOT/'planned_stop.json').exists() else None
prior_inventory={str(p):sha(p) for p in OUT.glob('*')}
if prior:
 assert prior_inventory==prior['inventory'],'Pre-resume files changed'
for roi in sorted({t['roi_index'] for t in targets}):
 rows=[t for t in targets if t['roi_index']==roi]
 paths=[OUT/f'roi{roi:02d}_{kind}_{lo:03d}.npy' for kind in ['trial','time'] for lo in range(0,100,10)]
 shape=(10,len(rows),6)
 existing={str(p):checkpoint(p,shape) for p in paths}
 ts=time.perf_counter();p1,p2,ev=r.load(1,roi,manifest);loads.append(time.perf_counter()-ts)
 obs=evaluate(p1,p2,ev,rows,0,'observed');op=OUT/f'roi{roi:02d}_observed.npy'
 old=checkpoint(op,obs.shape)
 if old is None:publish(op,obs,0)
 else:np.testing.assert_array_equal(old,obs)
 # Match the separately executed preflight at all offsets and both null types [replicate 0].
 previous=read(PRE/'tau_benchmark.json')['records']
 for kind in ['observed','trial','time']:
  values=obs if kind=='observed' else evaluate(p1,p2,ev,rows,0,kind)
  ref=np.array([[next(x['value'] for x in previous if x['uid']==t['uid'] and x['tau']==tau and x['kind']==kind) for tau in taus] for t in rows])
  np.testing.assert_allclose(values,ref,rtol=1e-7,atol=1e-7);errors.append(float(np.max(np.abs(values-ref))))
 for kind in ['trial','time']:
  for lo in range(0,100,10):
   p=OUT/f'roi{roi:02d}_{kind}_{lo:03d}.npy';old=existing[str(p)]
   if old is not None:
    reused+=1
    if prior and replay is None:
     recomputed=np.array([evaluate(p1,p2,ev,rows,k,kind) for k in range(lo,lo+10)])
     np.testing.assert_array_equal(old,recomputed);replay=dict(path=str(p),bitwise_equal=True,values=int(old.size))
    continue
   ts=time.perf_counter();a=np.array([evaluate(p1,p2,ev,rows,k,kind) for k in range(lo,lo+10)]);publish(p,a,time.perf_counter()-ts);new+=1
   if args.stop_after_batches and new>=args.stop_after_batches:
    save(ROOT/'planned_stop.json',dict(state='intentional checkpoint stop',seconds=time.perf_counter()-start,inventory={str(p):sha(p) for p in OUT.glob('*')}))
    print('PLANNED_STOP',flush=True);sys.exit(0)
 print(json.dumps(dict(roi=roi,new_batches=new,reused_batches=reused,seconds=time.perf_counter()-start)),flush=True)
assert prior and replay
for p,h in prior_inventory.items():assert sha(Path(p))==h
markers=list(OUT.glob('*.complete.json'));n=0;null_seconds=0.;counts={}
for roi in sorted({t['roi_index'] for t in targets}):
 rows=[t for t in targets if t['roi_index']==roi]
 for kind in ['trial','time']:
  for lo in range(0,100,10):
   p=OUT/f'roi{roi:02d}_{kind}_{lo:03d}.npy';a=checkpoint(p,(10,len(rows),6));assert a is not None;n+=a.size
   sec=read(p.with_suffix('.complete.json'))['seconds'];null_seconds+=sec;counts[kind]=counts.get(kind,0)+sec
assert n==19*6*100*2
elapsed=time.perf_counter()-start+prior['seconds']
save(ROOT/'completion_verification.json',dict(state='complete',dyad=1,null_values=n,observed_values=19*6,checkpoint_markers=len(markers),new_batches=new,reused_batches=reused,prior_files_preserved=len(prior_inventory),resume_replay=replay,max_preflight_error=max(errors),loading_seconds=sum(loads),surrogate_compute_seconds=counts,active_wall_seconds_including_planned_stop=elapsed,projected_15_dyad_all_time_minutes=elapsed*15/60,projection_scope='One dyad extrapolation with all time controls; excludes pauses between executions. No cohort tiers or clinical calculations.',signature=sig))
print('COMPLETE '+str(elapsed),flush=True)
