"""Bounded history preflight, no full-run entry point. Existing data read only."""
import os,sys,time,json,hashlib,ast,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;PRE=W/'PhiID_Matched_Followup_Preflight_2026-10-27'
sys.path.insert(0,str(W/'PhiID_Normative_Controls_2026-10-26'))
import control_core as c
sp=importlib.util.spec_from_file_location('controls',W/'PhiID_Normative_Controls_2026-10-26/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
import numpy as np
from scipy.spatial import cKDTree
tree=ast.parse((PRE/'check_baseline.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='variables'],type_ignores=[]),str(PRE/'check_baseline.py'),'exec'))
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (ROOT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
targets=read(PRE/'frozen_targets.json');manifest=read(r.PROD/'input_manifest.json');GRID=[(h,t) for h in [1,2,3] for t in [1,2,4,8,16,32,64]]
def design(ev,cell,centres,d):
 idx=np.flatnonzero([e[:3]==cell for e in ev]);fold=np.empty(len(idx),int)
 rng=c.common.stable_rng(2026,'history_folds',d,cell)
 for suffix in ['1','2']:
  ix=np.flatnonzero([ev[j][-1]==suffix for j in idx]);assert len(ix)>=5
  shuffled=rng.permutation(ix);fold[shuffled]=np.arange(len(ix))%5
 # Equal samples per trial, cap <=25,000 per cell across all folds.
 ntime=min(len(centres),25000//len(idx));assert ntime>=1
 positions=np.sort(rng.choice(len(centres),ntime,replace=False))
 return idx,fold,positions
def cv(v,fold,positions):
 # v centres x variables x paired trials. Each physical trial appears in one fold.
 z=v[positions].transpose(2,0,1).reshape(-1,4).astype('float64');ids=np.repeat(np.arange(len(fold)),len(positions));foldpoints=fold[ids]
 errors=np.empty((len(fold),2));scored=np.zeros(len(fold),bool)
 for f in range(5):
  tr=foldpoints!=f;te=~tr
  assert not set(ids[tr])&set(ids[te]);assert tr.sum()>=10 and te.any()
  mean=z[tr,:2].mean(0);sd=z[tr,:2].std(0,ddof=1);scale=z[tr,2:].var(0,ddof=1);assert (sd>0).all() and (scale>0).all()
  tree=cKDTree((z[tr,:2]-mean)/sd);_,nn=tree.query((z[te,:2]-mean)/sd,k=10,workers=1)
  pred=z[tr,2:][nn].mean(1);loss=(pred-z[te,2:])**2/scale
  for j in np.unique(ids[te]):errors[j]=loss[ids[te]==j].mean(0);scored[j]=True
 assert scored.all();return float(errors.mean()),len(z)
eligible=[];excluded=[]
for t in targets:
 centres=np.arange(t['start']+1,t['stop']+1);centres=centres[(centres>=192)&(centres<=1936)]
 rec=dict(uid=t['uid'],roi=t['roi'],geometry=t['geometry'],atom=t['atom'],contrast=t['contrast'],n_centres=len(centres),original_centres=t['stop']-t['start'])
 (eligible if len(centres)>=3 else excluded).append(rec)
save('eligibility.json',dict(eligible=eligible,excluded=excluded))
cases=[(1,next(t for t in targets if t['roi']=='bankssts-lh' and t['atom']=='xtx')),(6,next(t for t in targets if t['contrast']=='Role_by_Condition' and t['roi']=='inferiortemporal-lh'))]
save('benchmark_specification.json',dict(cases=[dict(dyad=d,target=t) for d,t in cases],grid=GRID,point_cap_per_cell=25000,allocation='equal time positions per trial, shared across configurations and transformations',folds=5,strata='event suffix within response-period/condition cell',k=10,normalization='training-only predictor mean/SD and training-only future variance',score='equal actor and trial means within cell; equal contributing cell means; full run would use equal dyad means',seed=2026,transformations=['observed','trial replicate 0','time replicate 0'],estimator='existing pointwise across-trials Gaussian/MMI on all common centres',case_time_budget_seconds=240,full_run=False,hashes={str(p):sha(p) for p in [Path(__file__),PRE/'check_baseline.py',PRE/'frozen_targets.json',Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),Path(c.common.__file__),r.PROD/'input_manifest.json',*Path(c.prod.calc.__file__).parent.glob('*.py')]}))
results=[];baselines=[]
for d,t in cases:
 begin=time.perf_counter();p1,p2,ev=r.load(d,t['roi_index'],manifest);load_seconds=time.perf_counter()-begin
 full=np.arange(t['start']+1,t['stop']+1);centres=full[(full>=192)&(full<=1936)];coef=r.COEFF[t['contrast']];g=int(t['geometry']=='cross_hemisphere');b=t['band_index']
 # Full-window baseline against both existing control assembler and saved production tensor.
 got=r.evaluate(p1,p2,ev,[t],d,0,'observed')[0];ref=r.stored(d,t['roi_index'],[t])[0];np.testing.assert_allclose(got,ref,atol=1e-7,rtol=1e-7)
 ident=np.arange(len(ev));maxerr=0.;manualerr=0.;history_one=0.
 designs={ci:design(ev,c.prod.CELLS[ci],centres,d) for ci in range(4) if coef[ci]!=0}
 for ci,(idx,fold,positions) in designs.items():
  cell=c.prod.CELLS[ci];v=variables(p1,p2,ev,cell,g,b,centres,1,ident,0,H=1);old=c.cell_atoms(p1,p2,ev,cell,g,b,centres,ident,0)[1];np.testing.assert_array_equal(v,old)
  # Independent per-trial actor history check at H3,tau64, including both suffixes.
  v3=variables(p1,p2,ev,cell,g,b,centres,64,ident,0,H=3)
  for suffix in ['1','2']:
   j=next(k for k,row in enumerate(idx) if ev[row][-1]==suffix);row=idx[j];X,Y=(p1,p2) if suffix=='1' else (p2,p1)
   for q in [0,len(centres)//2,len(centres)-1]:
    cc=centres[q];expected=np.array([np.mean([X[row,0,b,cc-k*64] for k in [1,2,3]]),np.mean([Y[row,g,b,cc-k*64] for k in [1,2,3]]),X[row,0,b,cc+64],Y[row,g,b,cc+64]])
    np.testing.assert_array_equal(v3[q,:,j],expected);manualerr=max(manualerr,float(abs(v3[q,:,j]-expected).max()))
  a=c.estimate_batch(v);history_one+=coef[ci]*a[:,t['atom_index']].mean()
  for q in [0,len(centres)//2,len(centres)-1]:
   refatom=c.prod.estimate(v[q]);np.testing.assert_allclose(a[q],refatom,atol=1e-7,rtol=1e-7);maxerr=max(maxerr,float(abs(a[q]-refatom).max()))
 baselines.append(dict(dyad=d,uid=t['uid'],full_window_error=float(abs(got-ref)),H1_assembler_exact=True,H3_manual_actor_error=manualerr,pointwise_error=maxerr,common_window_H1_tau1=float(history_one),centres=len(centres),cell_trials={c.prod.CELLS[ci]:len(x[0]) for ci,x in designs.items()},physical_trial_fold_separation=True))
 for kind in ['observed','trial','time']:
  start=time.perf_counter();perm,lag=c.perturbation(ev,d,0,kind)
  if kind=='trial':
   blocks,period=c.blocks_periods(ev);np.testing.assert_array_equal(blocks,blocks[perm]);np.testing.assert_array_equal(period,period[perm]);assert len(set(perm))==len(ev)
  records=[]
  for H,tau in GRID:
   if time.perf_counter()-begin>240:
    save(f'bounded_stop_dyad{d}.json',dict(reason='240-second case budget reached',kind=kind,completed_configurations=len(records)));raise RuntimeError('Bounded benchmark stopped before full grid; do not extrapolate as completed')
   st=time.perf_counter();scores=[];info=0.;points=[];prediction_seconds=0.;estimation_seconds=0.
   for ci,(idx,fold,positions) in designs.items():
    v=variables(p1,p2,ev,c.prod.CELLS[ci],g,b,centres,tau,perm,lag,H=H)
    # Bijective P2 permutation: physical P2 trial cannot straddle train/test folds.
    for f in range(5):assert not set(perm[idx[fold==f]])&set(perm[idx[fold!=f]])
    s=time.perf_counter();score,n=cv(v,fold,positions);prediction_seconds+=time.perf_counter()-s;scores.append(score);points.append(n)
    s=time.perf_counter();a=np.concatenate([c.estimate_batch(v[k:k+64]) for k in range(0,len(v),64)]);info+=coef[ci]*a[:,t['atom_index']].mean();estimation_seconds+=time.perf_counter()-s
   records.append(dict(H=H,tau=tau,score=float(np.mean(scores)),phiid_contrast=float(info),prediction_seconds=prediction_seconds,estimation_seconds=estimation_seconds,total_seconds=time.perf_counter()-st,points_per_cell=points))
  best=min(records,key=lambda x:(x['score'],x['H'],x['tau']));result=dict(dyad=d,uid=t['uid'],contrast=t['contrast'],geometry=t['geometry'],kind=kind,load_seconds=load_seconds,grid_seconds=time.perf_counter()-start,records=records,one_dyad_best=dict(H=best['H'],tau=best['tau'],score=best['score']),scope='Benchmark selection within one dyad only; full selection would aggregate all dyads before choosing a setting')
  save(f'benchmark_dyad{d}_{kind}.json',result);results.append(result);print(json.dumps({k:result[k] for k in ['dyad','kind','grid_seconds','one_dyad_best']}),flush=True)
save('verification.json',dict(status='passed',baseline_checks=baselines,cases=2,grid_evaluations=len(results)*21,full_run_launched=False,clinical_controls_untouched=True))
save('benchmark_summary.json',dict(results=[{k:x[k] for k in ['dyad','uid','kind','load_seconds','grid_seconds','one_dyad_best']} for x in results],eligible_targets=len(eligible),excluded_targets=len(excluded)))
print('COMPLETE',flush=True)
