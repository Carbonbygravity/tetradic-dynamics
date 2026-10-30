"""Four-worker history sensitivity runner. Benchmark only until full-run approval."""
import os,sys,json,time,hashlib,ast,importlib.util,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.spatial import cKDTree
import psutil
R=Path(__file__).resolve().parent;W=R.parent;OLD=W/'PhiID_History_Preflight_2026-10-27';PRE=W/'PhiID_Matched_Followup_Preflight_2026-10-27'
sys.path.insert(0,str(W/'PhiID_Normative_Controls_2026-10-26'))
import control_core as c
sp=importlib.util.spec_from_file_location('controls',W/'PhiID_Normative_Controls_2026-10-26/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
for path,names in [(PRE/'check_baseline.py',{'variables'}),(OLD/'benchmark.py',{'design','cv'})]:
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
def prepare():
 alltargets=read(PRE/'frozen_targets.json');eligible={t['uid'] for t in read(OLD/'eligibility.json')['eligible']};targets=[t for t in alltargets if t['uid'] in eligible];assert len(targets)==17
 selected=[next(i for i,t in enumerate(targets) if t['roi']=='bankssts-lh' and t['atom']=='xtx'),next(i for i,t in enumerate(targets) if t['contrast']=='Role_by_Condition' and t['roi']=='inferiortemporal-lh'),next(i for i,t in enumerate(targets) if t['roi']=='insula-lh' and t['atom']=='yty')]
 deps=[Path(__file__),PRE/'check_baseline.py',PRE/'frozen_targets.json',OLD/'benchmark.py',OLD/'eligibility.json',Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),Path(c.common.__file__),r.PROD/'input_manifest.json',*Path(c.prod.calc.__file__).parent.glob('*.py')]
 spec=dict(targets=targets,excluded=read(OLD/'eligibility.json')['excluded'],grid=GRID,dyads=read(W/'PhiID_Normative_Controls_2026-10-26/specification.json')['dyads'],workers=4,threads_per_worker=1,surrogates=100,seed=1337,coefficients=r.COEFF,selection='minimum equally weighted dyad/cell prediction score; select anew within each surrogate; ties H then tau',benchmark=dict(dyads=[1,6,8,16],target_indices=selected,kinds=['observed','trial','time'],replicate=0,worker_budget_seconds=240),hashes={str(p):sha(p) for p in deps})
 save(R/'specification.json',spec);log('Prepared frozen four-worker specification; no full calculation launched')
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
def benchmark():
 spec=read(R/'specification.json');sig=sha(R/'specification.json')
 for p,h in spec['hashes'].items():assert sha(p)==h
 folder='benchmark_checkpoints';(R/folder).mkdir(exist_ok=True);b=spec['benchmark'];jobs=lambda kinds:[(d,b['target_indices'],kinds,[0],folder,240) for d in b['dyads']]
 phase1=pool(jobs(['observed','trial']));save(R/'benchmark_phase1.json',phase1);before={str(p):sha(p) for p in (R/folder).glob('*')};assert len(before)==48
 log('Four-worker first phase complete: 24 grids saved; restarting workers to resume and add time-shift grids')
 phase2=pool(jobs(['observed','trial','time']));save(R/'benchmark_phase2.json',phase2);assert all(sha(p)==h for p,h in before.items());assert sum(x['reused'] for x in phase2['workers'])==24;assert sum(x['computed'] for x in phase2['workers'])==12
 phase3=pool(jobs(['observed','trial','time']));save(R/'benchmark_cached_resume.json',phase3);assert sum(x['computed'] for x in phase3['workers'])==0 and sum(x['reused'] for x in phase3['workers'])==36
 # Compare all 21 score/estimate pairs against the independently saved serial benchmarks.
 errors=[]
 for d,roi in [(1,'bankssts-lh'),(6,'inferiortemporal-lh')]:
  ti=next(i for i in b['target_indices'] if spec['targets'][i]['roi']==roi)
  for kind in ['observed','trial','time']:
   old=read(OLD/f'benchmark_dyad{d}_{kind}.json');ref=np.array([[x['score'],x['phiid_contrast']] for x in old['records']]);a=checkpoint(path_for(folder,d,ti,kind,0),sig);np.testing.assert_allclose(a,ref,rtol=1e-10,atol=1e-10);errors.append(float(abs(a-ref).max()))
 p=next((R/folder).glob('*.npy'))
 try:checkpoint(p,'wrong-signature')
 except AssertionError:pass
 else:raise AssertionError('Bad signature not rejected')
 markers=list((R/folder).glob('*.complete.json'));assert len(markers)==36
 for m in markers:assert checkpoint(m.with_suffix('').with_suffix('.npy'),sig) is not None
 workers=phase1['workers'];peak_parallel=max(sum(x['start_epoch']<=t<x['end_epoch'] for x in workers) for t in [x['start_epoch'] for x in workers]);assert peak_parallel==4
 seconds=[x['compute_seconds'] for phase in [phase1,phase2] for w in phase['workers'] for x in w['records']]
 # Measured work / wall ratio includes parallel overhead and initial input loading.
 efficiency=sum(seconds)/(phase1['wall_seconds']+phase2['wall_seconds']);lo,med,hi=np.percentile(seconds,[0,50,100]);estimate={}
 for name,grids in [('observed',255),('observed_plus_100_trial',255*101),('observed_trial_and_all_time',255*201)]:estimate[name]=dict(grid_count=grids,ideal_four_worker_hours=[float(grids*x/4/3600) for x in [lo,med,hi]],measured_throughput_hours=float(grids*med/efficiency/3600))
 save(R/'benchmark_verification.json',dict(status='passed',grids=36,configurations=756,simultaneous_workers=peak_parallel,serial_grid_comparisons=6,max_serial_error=max(errors),preserved_initial_grid_checkpoints=24,fresh_worker_resume=True,cached_only_recomputed=0,bad_signature_rejected=True,estimated_runtime=estimate,grid_seconds_min_median_max=[float(lo),float(med),float(hi)],combined_wall_seconds=phase1['wall_seconds']+phase2['wall_seconds'],sampled_worker_rss_bytes=[x['peak_sampled_rss'] for x in workers],scope='Four dyads, three targets, observed and one surrogate of each type; not a full cohort or full surrogate run',full_run_launched=False));log('Bounded benchmark and checkpoint resume verified; full run not launched')
def production_stage(kind):
 # Available for a later explicitly approved staged run. Not called by benchmark.
 spec=read(R/'specification.json');assert read(R/'benchmark_verification.json')['status']=='passed'
 for p,h in spec['hashes'].items():assert sha(p)==h
 (R/'production_checkpoints').mkdir(exist_ok=True);tis=list(range(17))
 if kind=='time':tis=read(R/'time_gate.json')
 reps=[0] if kind=='observed' else list(range(100))
 result=pool([(d,tis,[kind],reps,'production_checkpoints',float('inf')) for d in spec['dyads']]);save(R/f'{kind}_execution.json',result);summarize_stage(kind)
def summarize_stage(kind):
 spec=read(R/'specification.json');sig=sha(R/'specification.json');tis=list(range(17)) if kind!='time' else read(R/'time_gate.json');result=[]
 for ti in tis:
  reps=[0] if kind=='observed' else list(range(100));selections=[]
  for rep in reps:
   vv=[checkpoint(path_for('production_checkpoints',d,ti,kind,rep),sig) for d in spec['dyads']];assert all(v is not None for v in vv);v=np.array(vv);best=int(np.argmin(v[:,:,0].mean(0)))
   selections.append(dict(replicate=rep,selected_grid_index=best,H=GRID[best][0],tau=GRID[best][1],mean_scores=v[:,:,0].mean(0).tolist(),dyad_contrasts=v[:,best,1].tolist()))
  result.append(dict(target_index=ti,target=spec['targets'][ti],selections=selections))
 for title,interaction in [('four_contrast',False),('genuine_interaction',True)]:save(R/f'{kind}_{title}.json',[x for x in result if (x['target']['contrast']=='Role_by_Condition')==interaction])
 if kind=='trial':
  observed=read(R/'observed_four_contrast.json')+read(R/'observed_genuine_interaction.json');om={x['target_index']:x for x in observed};gate=[];tests=[]
  for row in result:
   ti=row['target_index'];direction=row['target']['trial']['direction'];obs=np.array(om[ti]['selections'][0]['dyad_contrasts']);null=np.array([s['dyad_contrasts'] for s in row['selections']]);gp=float((1+sum(direction*null.mean(1)>=direction*obs.mean()))/101);dp=(1+(direction*null>=direction*obs).sum(0))/101
   tests.append(dict(target_index=ti,group_p=gp,dyad_p=dp.tolist(),n_pass=int(sum(dp<=.05))))
   if gp<=.10:gate.append(ti)
  save(R/'trial_tests.json',tests);save(R/'time_gate.json',gate)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='benchmark':benchmark()
 elif sys.argv[1]=='stage' and len(sys.argv)>3 and sys.argv[3]=='--approved-full-run':production_stage(sys.argv[2])
 else:raise ValueError('Use prepare or benchmark. Production requires separate explicit approval.')
