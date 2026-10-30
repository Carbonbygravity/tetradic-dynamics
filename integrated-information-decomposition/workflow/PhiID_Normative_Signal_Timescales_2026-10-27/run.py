"""Normative Tier-A signal diagnostics. No inference or clinical calculations."""
import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parent;W=ROOT.parent
sys.path.insert(0,str(W/'PhiID_Normative_Controls_2026-10-26'))
import control_core as c
sp=importlib.util.spec_from_file_location('controls',W/'PhiID_Normative_Controls_2026-10-26/runner.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
import numpy as np
from scipy.fft import rfft,irfft,next_fast_len
import timescale_helpers as helper
r.COEFF['Role_by_Condition']=[-1,1,-1,1]
names={'normalized_autocorr','first_zero_crossing','first_local_minimum','histogram_mi','mi_curve'}
namespace={name:getattr(helper,name) for name in names}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 if p.exists():assert read(p)==x;return
 tmp=p.with_suffix('.partial')
 with tmp.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
 tmp.rename(p)
def log(x):
 msg=time.strftime('%Y-%m-%d %H:%M:%S')+' '+x;print(msg,flush=True)
 with (ROOT/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(msg+'\n')
sources=[W/'PhiID_Normative_Controls_2026-10-26/directional_tiers.json',W/'PhiID_Normative_Interaction_Controls_2026-10-26/directional_tiers.json']
targets=[]
for source in sources:
 for t in read(source):
  if t['tier']=='A':targets.append(dict(t,uid=source.parent.name+':'+str(t['id'])))
assert len(targets)==11
dyads=read(W/'PhiID_Normative_Controls_2026-10-26/specification.json')['dyads'];manifest=read(r.PROD/'input_manifest.json')
deps=[Path(__file__),Path(helper.__file__),Path(c.__file__),Path(r.__file__),Path(c.prod.__file__),r.PROD/'input_manifest.json',*sources]
spec=dict(targets=targets,dyads=dyads,mi_bins=16,min_pairs=20,feature_max_lag=96,epoch_max_lag=500,fs=1000,actor_mapping='suffix1 X=P1,Y=P2; suffix2 X=P2,Y=P1',aggregation='trial median within cell/actor/dyad, equal-cell median within actor/dyad, then equal-dyad median and quartiles; null when absent',hashes={str(p):sha(p) for p in deps})
save(ROOT/'specification.json',spec);sig=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
OUT=ROOT/'checkpoints';OUT.mkdir(exist_ok=True)
def ac_curves(x,maxlag):
 x=np.asarray(x,float);assert np.isfinite(x).all();x=x-x.mean(1,keepdims=True);n=x.shape[1];limit=min(maxlag,n-2,n-20)
 if limit<2:return np.full((len(x),1),np.nan)
 l=np.arange(limit+1);m=n-l;size=next_fast_len(2*n-1);f=rfft(x,n=size,axis=1);cross=irfft(f*f.conj(),n=size,axis=1)[:,:limit+1]
 s=np.c_[np.zeros(len(x)),np.cumsum(x,axis=1)];q=np.c_[np.zeros(len(x)),np.cumsum(x*x,axis=1)]
 sa=s[:,n-l];sb=s[:,n,None]-s[:,l];qa=q[:,n-l];qb=q[:,n,None]-q[:,l]
 va=qa/m-(sa/m)**2;vb=qb/m-(sb/m)**2
 with np.errstate(invalid='ignore',divide='ignore'):ac=(cross/m-sa*sb/m**2)/np.sqrt(va*vb)
 ac[:,0]=1.;ac[(va<=0)|(vb<=0)]=np.nan
 return ac
def mi_batch(x,lag):
 a=x[:,:-lag];b=x[:,lag:];n=a.shape[1];out=np.full(len(x),np.nan)
 if n<20:return out
 good=(a.var(1)>0)&(b.var(1)>0)
 if not good.any():return out
 a=a[good];b=b[good];amin=a.min(1)[:,None];bmin=b.min(1)[:,None]
 ia=np.minimum(15,np.floor((a-amin)/(a.max(1)[:,None]-amin)*16).astype(int));ib=np.minimum(15,np.floor((b-bmin)/(b.max(1)[:,None]-bmin)*16).astype(int))
 index=ia*16+ib+np.arange(len(a))[:,None]*256;hist=np.bincount(index.ravel(),minlength=len(a)*256).reshape(-1,16,16).astype(float);p=hist/n;den=p.sum(2)[:,:,None]*p.sum(1)[:,None,:]
 with np.errstate(divide='ignore',invalid='ignore'):term=np.where(p>0,p*np.log(p/den),0)
 out[good]=term.sum((1,2));return out
def diagnostics(x,maxlag):
 x=np.asarray(x,float);ac=ac_curves(x,maxlag);out=np.full((len(x),3),np.nan)
 for i,row in enumerate(ac):out[i,0]=namespace['first_zero_crossing'](row);out[i,1]=namespace['first_local_minimum'](row)
 active=np.arange(len(x));prev2=np.full(len(x),np.nan);prev=np.full(len(x),np.nan)
 limit=min(maxlag,x.shape[1]-2,x.shape[1]-20)
 for lag in range(1,limit+1):
  if not len(active):break
  cur=mi_batch(x[active],lag);hit=np.isfinite(prev2[active])&np.isfinite(prev[active])&np.isfinite(cur)&(prev[active]<=prev2[active])&(prev[active]<=cur)
  out[active[hit],2]=lag-1;prev2[active]=prev[active];prev[active]=cur;active=active[~hit]
 return out
def selected_series(p1,p2,ev,t,cell,actor,span):
 idx=np.flatnonzero([e[:3]==cell for e in ev]);mask=np.array([ev[j][-1]=='1' for j in idx])[:,None];band=t['band_index'];g=int(t['geometry']=='cross_hemisphere')
 x=np.where(mask,p1[idx,0,band],p2[idx,0,band]) if actor=='Leader' else np.where(mask,p2[idx,g,band],p1[idx,g,band])
 return x[:,t['start']+1:t['stop']+1] if span=='feature_window' else x
def checkpoint(p,shape):
 marker=p.with_suffix('.complete.json')
 if not p.exists() and not marker.exists():return None
 assert p.exists() and marker.exists();m=read(marker);assert m['signature']==sig and m['sha256']==sha(p);a=np.load(p,allow_pickle=False);assert a.shape==shape and m['shape']==list(shape);assert not np.isinf(a).any();return a
def publish(p,a):
 old=checkpoint(p,a.shape)
 if old is not None:np.testing.assert_array_equal(old,a);return
 tmp=p.with_suffix('.partial')
 with tmp.open('xb') as f:np.save(f,a)
 tmp.rename(p);save(p.with_suffix('.complete.json'),dict(signature=sig,sha256=sha(p),shape=list(a.shape),nan_meaning='No crossing/minimum within tested lags, or insufficient/constant data'))
def preflight():
 errors=[];count=0;start=time.perf_counter()
 for t in [targets[0],targets[-1]]:
  p1,p2,ev=r.load(1,t['roi_index'],manifest)
  for actor in ['Leader','Follower']:
   for span,ml in [('feature_window',96),('full_epoch',500)]:
    x=selected_series(p1,p2,ev,t,'LMR',actor,span)[[0,1,2,3]];a=diagnostics(x,ml)
    for i,row in enumerate(x):
     ac=namespace['normalized_autocorr'](row,ml);mc=namespace['mi_curve'](row,ml);ref=np.array([namespace['first_zero_crossing'](ac),namespace['first_local_minimum'](ac),namespace['first_local_minimum'](mc)])
     np.testing.assert_array_equal(a[i],ref);ours=ac_curves(row[None],ml)[0];np.testing.assert_allclose(ours,ac[:len(ours)],atol=1e-9,rtol=1e-9,equal_nan=True)
     for lag in [1,2,3,min(ml,len(row)-20)]:
      old=namespace['histogram_mi'](row[:-lag],row[lag:]);new=mi_batch(row[None].astype(float),lag)[0];np.testing.assert_allclose(old,new,atol=1e-12,rtol=1e-12);errors.append(float(abs(old-new)))
     count+=1
 save(ROOT/'preflight.json',dict(status='passed',series_checked=count,max_mi_error=max(errors),seconds=time.perf_counter()-start,scope='Two real dyad-1 targets, both actors and spans, four trials each; scalar historical diagnostic definitions matched.'))
 log('Diagnostic preflight passed')
def stat(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 return dict(n=len(v),median_ms=float(np.median(v)) if len(v) else None,q25_ms=float(np.percentile(v,25)) if len(v) else None,q75_ms=float(np.percentile(v,75)) if len(v) else None)
def run():
 assert read(ROOT/'preflight.json')['status']=='passed';lock=ROOT/'run.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 start=time.perf_counter();summaries=[]
 try:
  log('Approved normative Tier A diagnostics started; clinical controls unchanged')
  for d in dyads:
   for ti,t in enumerate(targets):
    cells=[c.prod.CELLS[i] for i,k in enumerate(r.COEFF[t['contrast']]) if k!=0];p1,p2,ev=r.load(d,t['roi_index'],manifest)
    for actor in ['Leader','Follower']:
     for span,ml in [('feature_window',96),('full_epoch',500)]:
      cellstats=[]
      for cell in cells:
       x=selected_series(p1,p2,ev,t,cell,actor,span);p=OUT/f'dyad{d}_target{ti:02d}_{actor}_{span}_{cell}.npy';a=checkpoint(p,(len(x),3))
       if a is None:a=np.concatenate([diagnostics(x[k:k+64],ml) for k in range(0,len(x),64)]);publish(p,a)
       cellstats.append(dict(cell=cell,trials=len(x),metrics=[stat(a[:,j]) for j in range(3)]))
      medians=[stat([v['metrics'][j]['median_ms'] for v in cellstats if v['metrics'][j]['median_ms'] is not None])['median_ms'] for j in range(3)]
      summaries.append(dict(dyad=d,uid=t['uid'],actor=actor,span=span,cell_summaries=cellstats,dyad_medians=medians))
    save(ROOT/f'dyad{d}_target{ti:02d}.json',[s for s in summaries if s['dyad']==d and s['uid']==t['uid']]);print(f'dyad {d} target {ti+1}/11',flush=True)
   log(f'dyad {d} completed');tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(dict(state='running',completed_dyads=dyads.index(d)+1,updated=time.strftime('%Y-%m-%d %H:%M:%S'))));os.replace(tmp,ROOT/'status.json')
  result=[]
  for t in targets:
   for actor in ['Leader','Follower']:
    for span in ['feature_window','full_epoch']:
     ss=[s for s in summaries if s['uid']==t['uid'] and s['actor']==actor and s['span']==span];assert len(ss)==15
     result.append(dict(target=t,actor=actor,span=span,metrics={name:stat([s['dyad_medians'][j] for s in ss if s['dyad_medians'][j] is not None]) for j,name in enumerate(['acf_zero','acf_minimum','mi_minimum'])}))
  files=[]
  for name,interaction in [('four_contrast_diagnostics.json',False),('genuine_interaction_diagnostics.json',True)]:
   p=ROOT/name;save(p,[x for x in result if (x['target']['contrast']=='Role_by_Condition')==interaction]);files.append(p)
  save(ROOT/'completion_verification.json',dict(state='complete',targets=11,dyads=15,summary_rows=len(result),seconds=time.perf_counter()-start,hashes={str(p):sha(p) for p in files},checkpoint_count=len(list(OUT.glob('*.complete.json')))));log('Diagnostics complete; genuine interaction saved separately')
 except BaseException as e:log('FAILED '+repr(e));raise
 finally:lock.unlink()
if __name__=='__main__':
 if sys.argv[1]=='preflight':preflight()
 elif sys.argv[1]=='run':run()
 else:raise ValueError('Unknown mode')
