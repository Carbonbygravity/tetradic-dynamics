"""Bounded read-only baseline and tau benchmark. No full-run entry point."""
import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent; W=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(W/'PhiID_Normative_Controls_2026-10-26'))
import control_core as c
sp=importlib.util.spec_from_file_location('saved_controls',W/'PhiID_Normative_Controls_2026-10-26/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
import numpy as np,h5py
def read(p):return json.loads(p.read_text())
def save(name,value):
 with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[W/'PhiID_Normative_Controls_2026-10-26/directional_tiers.json',W/'PhiID_Normative_Interaction_Controls_2026-10-26/directional_tiers.json']
targets=[]
for p in sources:
 for t in read(p):
  if t['tier'] in ['A','B']:targets.append(dict(t,source=str(p),uid=p.parent.name+':'+str(t['id'])))
assert len(targets)==19
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
manifest=read(r.PROD/'input_manifest.json')
def variables(p1,p2,ev,cell,g,b,centres,tau,perm,lag,H=1):
 idx=np.flatnonzero([e[:3]==cell for e in ev]);mask=np.array([ev[j][-1]=='1' for j in idx])[:,None];j=perm[idx]
 def take(p,rows,roi,t,shift=0):return p[rows,roi,b][:,(t-shift)%2001]
 def actor(t,x):
  return np.where(mask,take(p1,idx,0,t) if x else take(p2,j,g,t,lag),take(p2,j,0,t,lag) if x else take(p1,idx,g,t))
 xp=actor(centres-tau,True);yp=actor(centres-tau,False)
 if H>1:
  xp=np.mean(np.stack([actor(centres-k*tau,True) for k in range(1,H+1)]),axis=0)
  yp=np.mean(np.stack([actor(centres-k*tau,False) for k in range(1,H+1)]),axis=0)
 return np.stack([xp,yp,actor(centres+tau,True),actor(centres+tau,False)],axis=1).transpose(2,1,0)
def evaluate(p1,p2,ev,t,centres,tau,kind,d):
 perm,lag=c.perturbation(ev,d,0,kind);val=0.;err=0.
 for ci,coef in enumerate(r.COEFF[t['contrast']]):
  if coef==0:continue
  v=variables(p1,p2,ev,c.prod.CELLS[ci],int(t['geometry']=='cross_hemisphere'),t['band_index'],centres,tau,perm,lag)
  a=np.concatenate([c.estimate_batch(v[k:k+64]) for k in range(0,len(v),64)])
  # Independent pointwise production reference at boundary, middle and final centre.
  for k in sorted(set([0,len(v)//2,len(v)-1])):err=max(err,float(np.max(np.abs(a[k]-c.prod.estimate(v[k])))))
  assert err<1e-7
  val+=coef*a[:,t['atom_index']].mean()
 return float(val),err
save('frozen_targets.json',targets)
feas=[]
for t in targets:
 centres=np.arange(t['start']+1,t['stop']+1)
 for H in [1,2,3]:
  for tau in [1,2,4,8,16,32,64]:
   valid=(centres-H*tau>=0)&(centres+tau<2001)
   feas.append(dict(uid=t['uid'],H=H,tau=tau,original_centres=len(centres),valid_centres=int(valid.sum())))
save('window_feasibility.json',feas)
records=[];bench=[];start=time.perf_counter()
for d in [1,6,16]:
 selected=targets if d==1 else [targets[0],next(t for t in targets if t['contrast']=='Role_by_Condition'),next(t for t in targets if t['atom']=='yty')]
 for roi in sorted({t['roi_index'] for t in selected}):
  load_start=time.perf_counter();p1,p2,ev=r.load(d,roi,manifest);load_seconds=time.perf_counter()-load_start
  rows=[t for t in selected if t['roi_index']==roi]
  for t in rows:
   centres=np.arange(t['start']+1,t['stop']+1)
   obs,e=evaluate(p1,p2,ev,t,centres,1,'observed',d);stored=float(r.stored(d,roi,[t])[0]);np.testing.assert_allclose(obs,stored,atol=1e-7,rtol=1e-7)
   folder=Path(t['source']).parent;allrows=[x for x in read(folder/'targets.json') if x['roi_index']==roi]
   op=folder/'trial'/f'dyad{d}_roi{roi:02d}_observed.npy';mk=read(op.with_suffix('.complete.json'));assert sha(op)==mk['sha256']
   saved_control=float(np.load(op)[next(i for i,x in enumerate(allrows) if x['id']==t['id'])]);np.testing.assert_allclose(obs,saved_control,atol=1e-7,rtol=1e-7)
   records.append(dict(dyad=d,uid=t['uid'],observed=obs,stored_tensor=stored,saved_control=saved_control,max_abs_error=max(abs(obs-stored),abs(obs-saved_control)),pointwise_error=e,cell_counts={cell:sum(x[:3]==cell for x in ev) for cell in c.prod.CELLS}))
   if d==1:
    # Same centres for every benchmark tau; baseline full-window checked above.
    common=centres[(centres>=32)&(centres+32<2001)]
    for tau in [1,2,4,8,16,32]:
     for kind in ['observed','trial','time']:
      ts=time.perf_counter();value,error=evaluate(p1,p2,ev,t,common,tau,kind,d)
      bench.append(dict(uid=t['uid'],tau=tau,kind=kind,centres=len(common),seconds=time.perf_counter()-ts,value=value,pointwise_error=error,load_seconds=load_seconds))
  print(json.dumps(dict(dyad=d,roi=roi,checked=len(records),benchmarks=len(bench),elapsed=time.perf_counter()-start)),flush=True)
save('baseline_reproduction.json',dict(status='passed',checks=records,max_abs_error=max(x['max_abs_error'] for x in records),scope='All 19 targets in dyad 1; three targets each in dyads 6 and 16. Full target windows, tensor and saved observed-control comparisons. Not all-dyad validation.'))
save('tau_benchmark.json',dict(records=bench,elapsed_seconds=time.perf_counter()-start,scope='Dyad 1, all 19 targets, six tau offsets, observed plus one trial and one time surrogate per offset. Fixed common centres per target. No significance or tier inference.'))
save('provenance.json',dict(sources={str(p):sha(p) for p in sources+[Path(__file__),Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),r.PROD/'input_manifest.json']},inputs_verified=list(r.HASHED),clinical_untouched=True))
print('COMPLETE',flush=True)
