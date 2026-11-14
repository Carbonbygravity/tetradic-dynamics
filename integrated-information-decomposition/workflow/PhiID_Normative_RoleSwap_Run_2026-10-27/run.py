import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
R=Path(__file__).resolve().parent;W=R.parent;P=W/'PhiID_Normative_RoleSwap_Plan_2026-10-27'
sys.path.insert(0,str(W/'PhiID_Normative_Controls_2026-10-26'))
import control_core as c
sp=importlib.util.spec_from_file_location('controls',W/'PhiID_Normative_Controls_2026-10-26/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
targets=read(P/'target_inventory.json');manifest=read(r.PROD/'input_manifest.json');dyads=read(W/'PhiID_Normative_Controls_2026-10-26/specification.json')['dyads']
def save(p,x):
 if p.exists():assert read(p)==x;return
 with p.with_suffix('.partial').open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
 p.with_suffix('.partial').rename(p)
def log(s):
 s=time.strftime('%Y-%m-%d %H:%M:%S')+' '+s;print(s,flush=True)
 with (R/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(s+'\n')
deps=[Path(__file__),P/'target_inventory.json',Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),r.PROD/'input_manifest.json']
spec=dict(targets=targets,dyads=dyads,coefficients=r.COEFF,cell_order=c.prod.CELLS,subsamples=100,seed=1337,bootstrap=10000,bootstrap_seed=20261027,correction='Holm across five exact two-sided sign-flip tests; interaction reported separately',actor='suffix 1 X=P1 Y=P2; suffix 2 X=P2 Y=P1',estimand='late minus early signed contrast',hashes={str(p):sha(p) for p in deps})
save(R/'specification.json',spec);sig=sha(R/'specification.json');out=R/'checkpoints';out.mkdir(exist_ok=True)
def checkpoint(p,shape):
 m=p.with_suffix('.complete.json')
 if not p.exists() and not m.exists():return None
 assert p.exists() and m.exists();v=np.load(p,allow_pickle=False);meta=read(m)
 assert meta['signature']==sig and meta['sha256']==sha(p) and meta['shape']==list(shape) and v.shape==shape and np.isfinite(v).all()
 return v
def publish(p,v):
 old=checkpoint(p,v.shape)
 if old is not None:np.testing.assert_array_equal(old,v);return
 with p.with_suffix('.partial').open('xb') as f:np.save(f,v)
 p.with_suffix('.partial').rename(p);save(p.with_suffix('.complete.json'),dict(signature=sig,sha256=sha(p),shape=list(v.shape)))
def indices(ev,d,rep):
 result=[]
 for h in ['1','2']:
  chosen=[]
  for ci,cell in enumerate(c.prod.CELLS):
   ix=np.flatnonzero(ev==cell+h);assert len(ix)>=10
   if rep>=0:
    m=min(sum(ev==cell+'1'),sum(ev==cell+'2'))
    rng=np.random.default_rng(np.random.SeedSequence([1337,270127,d,rep,ci,int(h)]));ix=np.sort(rng.choice(ix,m,replace=False))
   chosen.extend(ix)
  result.append(np.sort(chosen))
 return result
def cells(p1,p2,ev,t,ix):
 a,b,e=p1[ix],p2[ix],ev[ix];centres=np.arange(t['start']+1,t['stop']+1)
 return np.array([c.cell_atoms(a,b,e,cell,int(t['geometry']=='cross_hemisphere'),t['band_index'],centres,np.arange(len(e)),0)[0][:,t['atom_index']].mean() for cell in c.prod.CELLS])
def baseline(p1,p2,ev,t,d):
 v=cells(p1,p2,ev,t,np.arange(len(ev)));actual=float(v@r.COEFF[t['contrast']]);stored=r.stored(d,t['roi_index'],[t])[0]
 np.testing.assert_allclose(actual,stored,atol=1e-7,rtol=1e-7);return abs(actual-stored)
def preflight():
 start=time.perf_counter();errors=[];times=[];mapping_checks=0
 for ti,t in enumerate(targets):
  ts=time.perf_counter();p1,p2,ev=r.load(1,t['roi_index'],manifest);errors.append(float(baseline(p1,p2,ev,t,1)))
  for h,ix in zip(['1','2'],indices(ev,1,-1)):
   a,b,e=p1[ix],p2[ix],ev[ix];g=int(t['geometry']=='cross_hemisphere');band=t['band_index'];centres=np.array([t['start']+1,(t['start']+t['stop'])//2+1,t['stop']])
   for cell in c.prod.CELLS:
    aa,v=c.cell_atoms(a,b,e,cell,g,band,centres,np.arange(len(e)),0);j=np.flatnonzero(e==cell+h)
    x,y=(a,b) if h=='1' else (b,a)
    explicit=np.stack([x[j,0,band][:,centres-1],y[j,g,band][:,centres-1],x[j,0,band][:,centres+1],y[j,g,band][:,centres+1]],axis=1).transpose(2,1,0)
    np.testing.assert_array_equal(v,explicit);ref=np.array([c.prod.estimate(z) for z in explicit]);np.testing.assert_allclose(aa,ref,rtol=1e-7,atol=1e-7);mapping_checks+=1
  ts2=time.perf_counter();values=[]
  for rep in [-1,0,1,2]:values.append(np.array([cells(p1,p2,ev,t,ix) for ix in indices(ev,1,rep)]))
  times.append(dict(target=ti,load_and_checks_seconds=ts2-ts,full_plus_three_subsamples_seconds=time.perf_counter()-ts2))
  v=np.array(values);path=R/f'benchmark_target{ti}.npy';publish(path,v);oldhash=sha(path);np.testing.assert_array_equal(checkpoint(path,v.shape),v);assert sha(path)==oldhash
  for shape in [(1,2,4)]:
   try:checkpoint(path,shape)
   except AssertionError:pass
   else:raise AssertionError('Bad checkpoint accepted')
  np.testing.assert_array_equal(indices(ev,1,0)[0],indices(ev,1,0)[0]);log(f'Benchmark target {ti+1}/5 passed')
 # Explicit contrast identities, including a case where interaction is zero but pooled period is not.
 q=np.array([3.,5.,2.,0.]);assert q@r.COEFF['Role_by_Condition']==0 and q@r.COEFF['Leader_period_minus_Follower_period']==3
 estimate=sum(x['full_plus_three_subsamples_seconds'] for x in times)*101/4*15
 save(R/'preflight.json',dict(status='passed',dyad=1,targets=5,mapping_cell_checks=mapping_checks,maximum_baseline_error=max(errors),benchmark=times,estimated_compute_seconds=estimate,seconds=time.perf_counter()-start,checkpoint_reload_exact=True,bad_shape_rejected=True,scope='All five dyad-1 targets; explicit actors for both halves and all four cells; pointwise production reference and pooled saved tensors; full and three subsampled repetitions'))
 log(f'Preflight complete; projected compute {estimate/60:.1f} minutes plus loading')
def run():
 assert read(R/'preflight.json')['status']=='passed'
 for p,h in spec['hashes'].items():assert sha(p)==h
 lock=R/'run.lock';lock.write_text(str(os.getpid())) if not lock.exists() else (_ for _ in ()).throw(RuntimeError('Existing lock'))
 start=time.perf_counter();errors=[]
 try:
  for d in dyads:
   for ti,t in enumerate(targets):
    paths=[out/f'dyad{d}_target{ti}_rep{k}.npy' for k in range(-1,100)]
    if all(checkpoint(p,(2,4)) is not None for p in paths):continue
    p1,p2,ev=r.load(d,t['roi_index'],manifest);errors.append(float(baseline(p1,p2,ev,t,d)))
    for rep,path in zip(range(-1,100),paths):
     if checkpoint(path,(2,4)) is None:publish(path,np.array([cells(p1,p2,ev,t,ix) for ix in indices(ev,d,rep)]))
    log(f'Dyad {d}, target {ti+1}/5 complete')
   r.c.prod.atomic_json(R/'status.json',dict(state='running',completed_dyads=dyads.index(d)+1,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
  summarize();save(R/'completion_verification.json',dict(state='complete',dyads=15,targets=5,checkpoints=7575,seconds=time.perf_counter()-start,max_pooled_baseline_error=max(errors) if errors else None));log('Completed; genuine interaction output separate')
  r.c.prod.atomic_json(R/'status.json',dict(state='complete',completed_dyads=15))
 finally:lock.unlink()
def summarize():
 signs=1-2*((np.arange(32768)[:,None]>>np.arange(15))&1);rng=np.random.default_rng(20261027);boot=rng.integers(0,15,size=(10000,15));results=[]
 for ti,t in enumerate(targets):
  v=np.array([[checkpoint(out/f'dyad{d}_target{ti}_rep{k}.npy',(2,4)) for k in range(-1,100)] for d in dyads]);assert v.shape==(15,101,2,4)
  cv=v@np.array(r.COEFF[t['contrast']]);delta=cv[:,:,1]-cv[:,:,0];D=delta[:,0];mean=D.mean();null=signs@D/15
  p=float(np.mean(np.abs(null)>=abs(mean)-1e-14));sd=D.std(ddof=1);means=delta[:,1:].mean(0)
  results.append(dict(target=t,early_mean=float(cv[:,0,0].mean()),late_mean=float(cv[:,0,1].mean()),change=float(mean),dz=float(mean/sd) if sd>0 else None,p=p,bootstrap_95=np.percentile(D[boot].mean(1),[2.5,97.5]).tolist(),dyad_changes=D.tolist(),cell_values=v[:,0].tolist(),subsample_mean_changes=means.tolist(),subsample_quantiles=np.percentile(means,[0,2.5,50,97.5,100]).tolist()))
 order=np.argsort([x['p'] for x in results]);adj=0
 for rank,j in enumerate(order):adj=max(adj,min(1,results[j]['p']*(5-rank)));results[j]['holm_p']=adj
 for filename,flag in [('four_contrast_results.json',False),('genuine_interaction_results.json',True)]:save(R/filename,[x for x in results if (x['target']['contrast']=='Role_by_Condition')==flag])
if __name__=='__main__':
 {'preflight':preflight,'run':run}[sys.argv[1]]()
