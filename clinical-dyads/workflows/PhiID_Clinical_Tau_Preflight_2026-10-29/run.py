"""Clinical lag sensitivity with separate interaction output and immutable checkpoints."""
import os,sys,json,time,hashlib,ast,argparse,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;PRE=W.parent/'phiid'
sys.path.insert(0,str(W/'PhiID_Clinical_Controls_2026-10-27'))
import control_core as c
sp=importlib.util.spec_from_file_location('controls',W/'PhiID_Clinical_Controls_2026-10-27/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
import numpy as np
tree=ast.parse((PRE/'lag.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='variables'],type_ignores=[]),str(PRE/'lag.py'),'exec'))
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 if p.exists():assert read(p)==x;return
 tmp=p.with_suffix(p.suffix+'.partial')
 with tmp.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 tmp.rename(p)
def log(msg):
 s=time.strftime('%Y-%m-%d %H:%M:%S')+' '+msg;print(s,flush=True)
 with (ROOT/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(s+'\n')
def status(**kw):
 tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(dict(kw,updated=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
 for i in range(60):
  try:os.replace(tmp,ROOT/'status.json');break
  except PermissionError:time.sleep(.25)
 else:raise RuntimeError('Status publish failed')
targets=read(ROOT/'frozen_targets.json');assert len(targets)==58
pairs=[dict(target=t,tau=tau,key=t['uid']+':tau'+str(tau)) for t in targets for tau in [1,2,4,8,16,32]]
dyads=read(W/'PhiID_Clinical_Controls_2026-10-27/specification.json')['dyads'];manifest=read(r.PROD/'input_manifest.json')
deps=[Path(__file__),PRE/'lag.py',ROOT/'frozen_targets.json',Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),Path(c.common.__file__),r.PROD/'input_manifest.json',*Path(c.prod.calc.__file__).parent.glob('*.py')]
spec=dict(cohort='clinical HC-PT',rng_namespace='clinical',dyads=dyads,pairs=pairs,contrasts=r.COEFF,workers=1,surrogates=100,seed=1337,H=1,centres='original target window intersect indices 32..1968',tau_offsets=[1,2,4,8,16,32],partner='physical P2',trial_strata='chronological block and response period',time_shift='circular 20..999 inclusive; deterministic common repetition across offsets',gate='fixed parent direction, cohort-mean trial p<=.10 per geometry-target/tau',p='(1+count(direction*null>=direction*observed))/101',tiers='A trial>=4 and time>=3 dyads; B trial>=4; C otherwise; dyad p<=.05',interaction='separate output file; no pooling with other contrasts',hashes={str(p):sha(p) for p in deps})
save(ROOT/'specification.json',spec);sig=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
OUT=ROOT/'checkpoints';OUT.mkdir(exist_ok=True)
def checkpoint(path,shape):
 m=path.with_suffix('.complete.json')
 if not path.exists() and not m.exists():return None
 assert path.exists() and m.exists(),'Partial checkpoint needs review'
 meta=read(m);assert meta['signature']==sig and meta['sha256']==sha(path) and meta['shape']==list(shape)
 a=np.load(path,allow_pickle=False);assert a.shape==shape and np.isfinite(a).all();return a
def publish(path,a):
 old=checkpoint(path,a.shape)
 if old is not None:np.testing.assert_array_equal(a,old);return
 tmp=path.with_suffix('.partial')
 with tmp.open('xb') as f:np.save(f,a);f.flush();os.fsync(f.fileno())
 tmp.rename(path);save(path.with_suffix('.complete.json'),dict(signature=sig,sha256=sha(path),shape=list(a.shape)))
def evaluate(p1,p2,ev,rows,d,rep,kind):
 perm,lag=c.perturbation(ev,d,rep,kind);result=[]
 for row in rows:
  t=row['target'];tau=row['tau'];centres=np.arange(t['start']+1,t['stop']+1);centres=centres[(centres>=32)&(centres<=1968)];assert len(centres)>=3
  val=0.
  for ci,coef in enumerate(r.COEFF[t['contrast']]):
   if coef==0:continue
   v=variables(p1,p2,ev,c.prod.CELLS[ci],int(t['geometry']=='cross_hemisphere'),t['band_index'],centres,tau,perm,lag)
   a=np.concatenate([c.estimate_batch(v[k:k+64]) for k in range(0,len(v),64)])
   val+=coef*a[:,t['atom_index']].mean()
  result.append(val)
 return np.array(result)
def baseline(p1,p2,ev,rows,d,roi):
 ts=[x['target'] for x in rows if x['tau']==1];values=r.evaluate(p1,p2,ev,ts,d,0,'observed');stored=r.stored(d,roi,ts);np.testing.assert_allclose(values,stored,atol=1e-7,rtol=1e-7)
 save(ROOT/f'baseline_dyad{d}_roi{roi:02d}.json',dict(targets=[t['uid'] for t in ts],max_error=float(np.max(np.abs(values-stored))),count=len(ts)))
def stage(kind,selected):
 for d in dyads:
  for roi in sorted({row['target']['roi_index'] for row in selected}):
   rows=[row for row in selected if row['target']['roi_index']==roi];prefix=f'dyad{d}_roi{roi:02d}_{kind}';paths=[OUT/f'{prefix}_{lo:03d}.npy' for lo in range(0,100,10)]
   if all(checkpoint(p,(10,len(rows))) is not None for p in paths):continue
   p1,p2,ev=r.load(d,roi,manifest)
   if kind=='trial':baseline(p1,p2,ev,rows,d,roi)
   obs=evaluate(p1,p2,ev,rows,d,0,'observed');publish(OUT/f'{prefix}_observed.npy',obs)
   for lo,p in zip(range(0,100,10),paths):
    if checkpoint(p,(10,len(rows))) is not None:continue
    a=np.array([evaluate(p1,p2,ev,rows,d,j,kind) for j in range(lo,lo+10)]);publish(p,a);status(state='running',stage=kind,dyad=d,roi=roi,surrogates=lo+10)
  log(f'{kind}: dyad {d} completed')
def summarize(kind,selected):
 obs=np.empty((15,len(selected)));null=np.empty((15,100,len(selected)))
 for di,d in enumerate(dyads):
  for roi in sorted({x['target']['roi_index'] for x in selected}):
   ix=[i for i,x in enumerate(selected) if x['target']['roi_index']==roi];prefix=f'dyad{d}_roi{roi:02d}_{kind}'
   a=checkpoint(OUT/f'{prefix}_observed.npy',(len(ix),));assert a is not None;obs[di,ix]=a
   blocks=[checkpoint(OUT/f'{prefix}_{lo:03d}.npy',(10,len(ix))) for lo in range(0,100,10)];assert all(x is not None for x in blocks);null[di][:,ix]=np.concatenate(blocks)
 signs=np.array([int(x['target']['parent_cluster'].rsplit(':',1)[1]) for x in selected]);assert np.isin(signs,[-1,1]).all()
 group=(1+(null.mean(0)*signs>=obs.mean(0)*signs).sum(0))/101
 per=(1+(null*signs>=obs[:,None,:]*signs).sum(1))/101
 result=[dict(key=x['key'],group_p=float(group[i]),dyad_p=per[:,i].tolist(),n_pass=int((per[:,i]<=.05).sum()),mean_observed=float(obs[:,i].mean())) for i,x in enumerate(selected)]
 save(ROOT/f'{kind}_summary.json',result);return result
def run():
 assert (ROOT/'execution_authorization.json').exists(), 'Missing execution_authorization.json'
 assert read(ROOT/'preflight.json')['status']=='passed'
 assert read(ROOT/'fresh_process_resume.json')['status']=='passed'
 for p,h in spec['hashes'].items():assert sha(Path(p))==h
 lock=ROOT/'run.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  log('Starting or resuming the clinical lag sweep');stage('trial',pairs);tr=summarize('trial',pairs);ids={x['key'] for x in tr if x['group_p']<=.10};selected=[x for x in pairs if x['key'] in ids];save(ROOT/'time_gate.json',dict(keys=sorted(ids)));log(f'Trial stage verified; {len(selected)}/{len(pairs)} target-offsets enter time controls')
  if selected:stage('time',selected);tm=summarize('time',selected)
  else:tm=[]
  a={x['key']:x for x in tr};b={x['key']:x for x in tm};result=[]
  for row in pairs:
   tt=a[row['key']];st=b.get(row['key']);tier='A' if tt['n_pass']>=4 and st and st['n_pass']>=3 else 'B' if tt['n_pass']>=4 else 'C'
   result.append(dict(row,trial=tt,time=st,tier=tier))
  files=[]
  for name,interaction in [('four_contrast_tau_results.json',False),('genuine_interaction_tau_results.json',True)]:
   save(ROOT/name,[x for x in result if (x['target']['contrast']=='Role_by_Condition')==interaction]);files.append(ROOT/name)
  save(ROOT/'completion_verification.json',dict(state='complete',target_offsets=len(pairs),dyads=15,time_gate_pass=len(selected),hashes={str(p):sha(p) for p in files},scope='Separate per-offset sensitivity tiers; primary tiers unchanged. All checkpoints checked during summaries.'));status(state='complete');log('Tau sweep completed and checkpoint summaries verified')
 except BaseException as e:status(state='failed',error=repr(e));log('Failed: '+repr(e));raise
 finally:lock.unlink()
if __name__=='__main__':
 if sys.argv[1]=='run':run()
 else:raise ValueError('Unknown mode')
