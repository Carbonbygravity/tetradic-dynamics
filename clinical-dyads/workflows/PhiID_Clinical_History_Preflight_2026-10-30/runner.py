"""Four-worker clinical history sensitivity engine."""
import os,sys,json,time,hashlib,ast,importlib.util,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.spatial import cKDTree
import psutil
R=Path(__file__).resolve().parent;W=R.parent;OLD=W.parent/'phiid';PRE=W.parent/'phiid'
sys.path.insert(0,str(W/'PhiID_Clinical_Controls_2026-10-27'))
import control_core as c
sp=importlib.util.spec_from_file_location('controls',W/'PhiID_Clinical_Controls_2026-10-27/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
for path,names in [(PRE/'lag.py',{'variables'}),(OLD/'history.py',{'design','cv'})]:
 tree=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'))
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
GRID=[(h,t) for h in [1,2,3] for t in [1,2,4,8,16,32,64]]
def save(p,x):
 if p.exists():assert read(p)==x;return
 with p.with_suffix('.partial').open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
 p.with_suffix('.partial').rename(p)
def log(s):
 s=time.strftime('%Y-%m-%d %H:%M:%S')+' '+s;print(s,flush=True)
 with (R/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(s+'\n')
def checkpoint(p,sig):
 m=p.with_suffix('.complete.json')
 if not p.exists() and not m.exists():return None
 assert p.exists() and m.exists(),'Partial checkpoint'
 meta=read(m);a=np.load(p,allow_pickle=False)
 assert meta['signature']==sig and meta['sha256']==sha(p) and a.shape==(21,2) and meta['shape']==[21,2] and np.isfinite(a).all(),'Invalid checkpoint'
 return a

def publish(p,v,sig,info):
 assert checkpoint(p,sig) is None
 with p.with_suffix('.partial').open('xb') as f:np.save(f,v)
 p.with_suffix('.partial').rename(p);save(p.with_suffix('.complete.json'),dict(signature=sig,sha256=sha(p),shape=[21,2],**info))

def path_for(folder,d,ti,kind,rep):return R/folder/f'dyad{d}_target{ti:02d}_{kind}_{rep:03d}.npy'

def calculate(p1,p2,ev,t,d,kind,rep):
 full=np.arange(t['start']+1,t['stop']+1);centres=full[(full>=192)&(full<=1936)];assert len(centres)>=3
 coef=r.COEFF[t['contrast']];g=int(t['geometry']=='cross_hemisphere');b=t['band_index'];perm,lag=c.perturbation(ev,d,rep,kind)
 designs={ci:design(ev,c.prod.CELLS[ci],centres,d) for ci in range(4) if coef[ci]!=0}
 if kind=='trial':
  blocks,periods=c.blocks_periods(ev);assert len(set(perm))==len(ev);np.testing.assert_array_equal(blocks,blocks[perm]);np.testing.assert_array_equal(periods,periods[perm])
 for idx,fold,pos in designs.values():
  for f in range(5):assert not set(perm[idx[fold==f]])&set(perm[idx[fold!=f]])
 values=[]
 for H,tau in GRID:
  assert (centres-H*tau>=0).all() and (centres+tau<2001).all();scores=[];info=0
  for ci,(idx,fold,positions) in designs.items():
   v=variables(p1,p2,ev,c.prod.CELLS[ci],g,b,centres,tau,perm,lag,H=H)
   score,n=cv(v,fold,positions);scores.append(score);a=np.concatenate([c.estimate_batch(v[k:k+64]) for k in range(0,len(v),64)])
   info+=coef[ci]*a[:,t['atom_index']].mean()
  values.append([np.mean(scores),info])
 v=np.array(values);assert np.isfinite(v).all();return v

def worker(job):
 d,tis,kinds,reps,folder,budget=job;started=time.perf_counter();epoch_start=time.time();spec=read(R/'specification.json');sig=sha(R/'specification.json');manifest=read(r.PROD/'input_manifest.json');computed=0;reused=0;records=[];baseline=[];peak=0
 for roi in sorted({spec['targets'][i]['roi_index'] for i in tis}):
  local=[i for i in tis if spec['targets'][i]['roi_index']==roi];pending=[]
  for ti in local:
   for kind in kinds:
    for rep in ([0] if kind=='observed' else reps):
     p=path_for(folder,d,ti,kind,rep)
     if checkpoint(p,sig) is not None:reused+=1
     else:pending.append((ti,kind,rep,p))
  if not pending:continue
  ts=time.perf_counter();p1,p2,ev=r.load(d,roi,manifest);load_seconds=time.perf_counter()-ts
  for ti in local:
   t=spec['targets'][ti];got=r.evaluate(p1,p2,ev,[t],d,0,'observed')[0];ref=r.stored(d,roi,[t])[0];np.testing.assert_allclose(got,ref,atol=1e-7,rtol=1e-7);baseline.append(float(abs(got-ref)))
  for ti,kind,rep,p in pending:
   if time.perf_counter()-started>budget:raise RuntimeError('Bounded worker budget exceeded; completed checkpoints preserved')
   ts=time.perf_counter();v=calculate(p1,p2,ev,spec['targets'][ti],d,kind,rep);secs=time.perf_counter()-ts
   info=dict(dyad=d,target_index=ti,kind=kind,replicate=rep,compute_seconds=secs,load_seconds=load_seconds,pid=os.getpid());publish(p,v,sig,info);records.append(info);computed+=1;peak=max(peak,psutil.Process().memory_info().rss)
 return dict(dyad=d,pid=os.getpid(),computed=computed,reused=reused,seconds=time.perf_counter()-started,start_epoch=epoch_start,end_epoch=time.time(),peak_sampled_rss=peak,records=records,baseline_max_error=max(baseline) if baseline else None)

def pool(jobs):
 start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as ex:results=list(ex.map(worker,jobs))
 return dict(wall_seconds=time.perf_counter()-start,workers=results)
