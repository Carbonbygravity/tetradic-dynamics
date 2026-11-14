"""Separate approved interaction controls; shared numerical functions copied unchanged."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,sys,time,hashlib,uuid,platform
from pathlib import Path
import numpy as np,h5py,scipy
import control_core as c
ROOT=Path(__file__).resolve().parent;W=ROOT.parent
OVER=W/'PhiID_Matched_PerMap_2026-10-26/geometry_overlap';PROD=W/'PhiID_Production_Preflight_2026-10-25'
ATOMS=c.prod.ATOMS;BANDS=c.prod.BANDS
NAMES=['Role_by_Condition']
COEFF={k:c.common.CONTRASTS[k].tolist() for k in NAMES}

HASHED=set()

def sha(p):return c.prod.digest(Path(p))

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))

def save(p,v):
 if p.exists():raise FileExistsError(p)
 c.prod.atomic_json(p,v)

def status(v):c.prod.atomic_json(ROOT/'status.json',dict(v,updated=time.strftime('%Y-%m-%d %H:%M:%S')))

def log(v):
 s=time.strftime('%Y-%m-%d %H:%M:%S')+' — '+v;print(s,flush=True)
 with (ROOT/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(s+'\n')

def publish(p,v,sig):
 if p.exists() or p.with_suffix('.complete.json').exists():raise FileExistsError(p)
 tmp=p.with_suffix('.partial');
 with tmp.open('xb') as f:np.save(f,v)
 tmp.rename(p);save(p.with_suffix('.complete.json'),dict(signature=sig,sha256=sha(p),shape=list(v.shape)))

def checkpoint(p,sig,shape):
 m=p.with_suffix('.complete.json')
 if not p.exists() and not m.exists():return None
 if not p.exists() or not m.exists():raise RuntimeError('Partial checkpoint '+str(p))
 r=json.loads(m.read_text());assert r['signature']==sig and r['sha256']==sha(p)
 v=np.load(p,allow_pickle=False);assert v.shape==tuple(shape) and np.isfinite(v).all();return v

def verify(spec):
 assert sha(ROOT/'targets.json')==spec['targets_sha256'];assert sha(spec['manifest'])==spec['manifest_sha256'];assert sha(PROD/'production/specification.json')==spec['production_spec_sha256']
 for p,h in {**spec['sources'],**spec['code_hashes']}.items():assert sha(p)==h

def load(d,roi,manifest):
 src=sorted([s for s in manifest['inputs'] if s['dyad']==d],key=lambda s:s['partner'])
 # Each input is fully hashed once per process/dyad, never written.
 for s in src:
  if s['path'] not in HASHED:assert sha(s['path'])==s['sha256'];HASHED.add(s['path'])
 with h5py.File(src[0]['path'],'r') as a,h5py.File(src[1]['path'],'r') as b:
  for k in ['events','selection','event_labels','roi_labels','times_seconds']:np.testing.assert_array_equal(a[k][:],b[k][:])
  ev=a['event_labels'].asstr()[:];labs=a['roi_labels'].asstr()[:].tolist();lab=labs[roi];opp=labs.index(lab[:-3]+('-rh' if lab.endswith('-lh') else '-lh'))
  assert all(e in ['LMR1','FMR1','LMA1','FMA1','LMA2','FMA2','LMR2','FMR2'] for e in ev)
  return c.prod.load_power(a,[roi,opp]),c.prod.load_power(b,[roi,opp]),ev

def evaluate(p1,p2,events,rows,dyad,rep,kind):
 perm,lag=c.perturbation(events,dyad,rep,kind);result=np.empty(len(rows))
 groups={}
 for j,t in enumerate(rows):groups.setdefault((t['geometry'],t['band_index']),[]).append((j,t))
 for (g,band),items in groups.items():
  centres=np.unique(np.concatenate([np.arange(t['start']+1,t['stop']+1) for _,t in items]));needed=[ci for ci in range(4) if any(COEFF[t['contrast']][ci]!=0 for _,t in items)]
  a={}
  for ci in needed:a[ci]=c.cell_atoms(p1,p2,events,c.prod.CELLS[ci],int(g=='cross_hemisphere'),band,centres,perm,lag)[0]
  for j,t in items:
   take=np.flatnonzero((centres>=t['start']+1)&(centres<t['stop']+1));ai=t['atom_index'];coef=COEFF[t['contrast']]
   result[j]=sum(coef[ci]*a[ci][take,ai].mean() for ci in needed if coef[ci]!=0)
 return result

def stored(d,roi,rows):
 path=PROD/f'production/dyad{d}_roi{roi:02d}.h5';marker=json.loads(path.with_suffix('.complete.json').read_text());assert sha(path)==marker['sha256']
 with h5py.File(path,'r') as f:v=f['values'][:]
 return np.array([np.dot(COEFF[t['contrast']],v[int(t['geometry']=='cross_hemisphere'),:,t['band_index'],t['start']:t['stop'],t['atom_index']].mean(axis=1)) for t in rows])

def preflight(spec):
 verify(spec);manifest=json.loads(Path(spec['manifest']).read_text());targets=json.loads((ROOT/'targets.json').read_text());errors=[];n=0;started=time.time();bench=[]
 for d,roi in [(1,0),(8,11),(16,22)]:
  p1,p2,ev=load(d,roi,manifest);ident,lag=c.perturbation(ev,d,0,'observed');bp,per=c.blocks_periods(ev)
  permutation,_=c.perturbation(ev,d,0,'trial');assert sorted(permutation.tolist())==list(range(len(ev)));np.testing.assert_array_equal(bp,bp[permutation]);np.testing.assert_array_equal(per,per[permutation])
  with h5py.File(PROD/f'production/dyad{d}_roi{roi:02d}.h5','r') as f:old=f['values'][:]
  for geometry in range(2):
   for ci,cell in enumerate(c.prod.CELLS):
    for band in [0,2,5]:
     centres=np.array([250,750,1000,1250,1750]);a,v=c.cell_atoms(p1,p2,ev,cell,geometry,band,centres,ident,0);ref=np.array([c.prod.estimate(z) for z in v])
     np.testing.assert_allclose(a,ref,rtol=1e-7,atol=1e-7);np.testing.assert_allclose(a,old[geometry,ci,band,centres-1],rtol=1e-7,atol=1e-7);errors.append(float(np.max(np.abs(a-ref))));n+=a.size
  # Check both null transformations against the unchanged pointwise reference estimator.
  for kind in ['trial','time']:
   perm,lag=c.perturbation(ev,d,0,kind)
   for geom in range(2):
    for cell in ['LMR','FMA']:
     a,v=c.cell_atoms(p1,p2,ev,cell,geom,0,np.array([750,1000,1250]),perm,lag);ref=np.array([c.prod.estimate(z) for z in v]);np.testing.assert_allclose(a,ref,rtol=1e-7,atol=1e-7);errors.append(float(np.max(np.abs(a-ref))));n+=a.size
  rows=[t for t in targets if t['roi_index']==roi]
  if rows:
   start=time.time();obs=evaluate(p1,p2,ev,rows,d,0,'observed');np.testing.assert_allclose(obs,stored(d,roi,rows),rtol=1e-7,atol=1e-7)
   trial=evaluate(p1,p2,ev,rows,d,0,'trial');bench.append(dict(dyad=d,roi=roi,targets=len(rows),observed_plus_one_trial_seconds=time.time()-start));assert np.isfinite(trial).all()
  log(f'Preflight dyad {d}, ROI {roi}: production and null-estimator comparisons passed.')
 # Check persistent restart and refusal of damaged metadata without touching production checkpoints.
 sig=sha(ROOT/'specification.json');p=ROOT/'preflight_checkpoint.npy';val=np.arange(12.).reshape(3,4);publish(p,val,sig);np.testing.assert_array_equal(checkpoint(p,sig,val.shape),val)
 rejected=0
 for badsig,shape in [('wrong',val.shape),(sig,(4,3))]:
  try:checkpoint(p,badsig,shape)
  except AssertionError:rejected+=1
 assert rejected==2
 save(ROOT/'preflight.json',dict(status='passed',atom_values_compared=n,max_absolute_error=max(errors),seconds=time.time()-started,benchmarks=bench,checkpoint_resume_exact=True,bad_signature_and_shape_rejected=True,scope='Selected 3 dyads x 3 bands x both geometries x all 4 cells, observed and transformed input comparisons. All target observations will be checked against stored tensors during production.'))

def pvalue(obs,null):return (1+np.sum(null>=obs,axis=0))/(len(null)+1)

def stage(spec,kind,targets,manifest,sig):
 out=ROOT/kind;out.mkdir(exist_ok=True)
 for d in spec['dyads']:
  for roi in sorted({t['roi_index'] for t in targets}):
   rows=[t for t in targets if t['roi_index']==roi];prefix=f'dyad{d}_roi{roi:02d}';shape=(10,len(rows));paths=[out/f'{prefix}_{lo:03d}_{lo+10:03d}.npy' for lo in range(0,100,10)]
   existing=[checkpoint(p,sig,shape) for p in paths]
   if all(v is not None for v in existing):continue
   p1,p2,ev=load(d,roi,manifest);obs=evaluate(p1,p2,ev,rows,d,0,'observed');np.testing.assert_allclose(obs,stored(d,roi,rows),rtol=1e-7,atol=1e-7)
   op=out/(prefix+'_observed.npy');previous=checkpoint(op,sig,obs.shape)
   if previous is None:publish(op,obs,sig)
   else:np.testing.assert_array_equal(previous,obs)
   for bi,path in enumerate(paths):
    if existing[bi] is not None:continue
    start=time.time();v=np.array([evaluate(p1,p2,ev,rows,d,rep,kind) for rep in range(bi*10,bi*10+10)]);publish(path,v,sig)
    status(dict(state='running',stage=kind,dyad=d,roi=roi,completed_surrogates_for_roi=(bi+1)*10,targets_for_roi=len(rows),last_batch_seconds=time.time()-start));print(kind,d,roi,(bi+1)*10,flush=True)
  log(f'{kind}: dyad {d} finished all eligible ROI controls.')
