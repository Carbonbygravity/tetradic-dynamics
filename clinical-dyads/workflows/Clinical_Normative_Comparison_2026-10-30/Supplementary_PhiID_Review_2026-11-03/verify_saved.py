import json, hashlib, io, sys, warnings
from pathlib import Path
from collections import Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor
pool=ThreadPoolExecutor(max_workers=12)
sys.stdout.reconfigure(encoding='utf-8')
import os
W=Path(os.environ['TETRADIC_RESULTS_ROOT']).resolve(); OUT=Path(os.environ['TETRADIC_VERIFICATION_OUTPUT']).resolve()
OUT.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Read-only verification of saved checkpoints, selection and reporting arithmetic. No EEG re-estimation or new surrogate generation.', 'hash_checks':0}
def hashes(mapping):
 for p,h in mapping.items():
  assert sha(p)==h,p
  report['hash_checks']+=1
def ck(p,sig,shape=None,allow_nan=False):
 raw=p.read_bytes();m=read(p.with_suffix('.complete.json'));a=np.load(io.BytesIO(raw),allow_pickle=False)
 assert hashlib.sha256(raw).hexdigest()==m['sha256'] and sig==m['signature'],str(p)
 assert list(a.shape)==m['shape'] and (shape is None or a.shape==shape),str(p)
 assert (not np.isinf(a).any() if allow_nan else np.isfinite(a).all()),str(p)
 return a
def tier(tr,tm):return 'A' if tr>=4 and tm>=3 else 'B' if tr>=4 else 'C'
def pv(o,nu,sign):
 return float((1+np.sum(sign*nu.mean(1)>=sign*o.mean()))/101),(1+(sign*nu>=sign*o).sum(0))/101
primary={}
for group,folder in [('main','PhiID_Clinical_Controls_2026-10-27'),('interaction','PhiID_Clinical_Interaction_Overlap_Controls_2026-10-28')]:
 for x in read(W/folder/'directional_tiers.json'):
  assert x['tier']==tier(x['trial']['n_pass'],x['time_shift']['n_pass'] if x['time_shift'] else 0)
  if x['tier'] in 'AB':primary[f'clinical:{group}:{x["id"]}']=x
report['primary_counts']=dict(Counter(k.split(':')[1] for k in primary))
# Tau: independently reconstruct p values from all production arrays.
R=W/'PhiID_Clinical_Tau_Preflight_2026-10-29';s=read(R/'specification.json');sig=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
hashes(s['hashes']);hashes(read(R/'execution_authorization.json')['protected_hashes']);hashes(read(R/'completion_verification.json')['hashes'])
assert read(R/'execution_exit.json')['exit_code']==0
rows=s['pairs'];counts=0;results={}
for kind in ['trial','time']:
 selected=rows if kind=='trial' else [x for x in rows if x['key'] in read(R/'time_gate.json')['keys']]
 obs=np.empty((15,len(selected)));null=np.empty((15,100,len(selected)))
 for di,d in enumerate(s['dyads']):
  for roi in sorted({x['target']['roi_index'] for x in selected}):
   ix=[j for j,x in enumerate(selected) if x['target']['roi_index']==roi]
   pre=R/'checkpoints'/f'dyad{d}_roi{roi:02d}_{kind}'
   obs[di,ix]=ck(Path(str(pre)+'_observed.npy'),sig,(len(ix),));counts+=1
   null[di][:,ix]=np.concatenate([ck(Path(str(pre)+f'_{lo:03d}.npy'),sig,(10,len(ix))) for lo in range(0,100,10)]);counts+=10
 saved=read(R/f'{kind}_summary.json');assert len(saved)==len(selected)
 for j,x in enumerate(selected):
  gp,dp=pv(obs[:,j],null[:,:,j].T,x['target']['trial']['direction']);q=saved[j]
  assert q['key']==x['key'] and q['group_p']==gp and q['n_pass']==int(sum(dp<=.05));np.testing.assert_array_equal(dp,q['dyad_p'])
  np.testing.assert_allclose(obs[:,j].mean(),q['mean_observed'],rtol=0,atol=1e-14)
 results[kind]={x['key']:x for x in saved}
 assert kind!='trial' or sorted(x['key'] for x in saved if x['group_p']<=.1)==sorted(read(R/'time_gate.json')['keys'])
tau={};stable={};direction_changes={}
for group,fn in [('main','four_contrast_tau_results.json'),('interaction','genuine_interaction_tau_results.json')]:
 rr=read(R/fn);tau[group]={str(t):dict(Counter(x['tier'] for x in rr if x['tau']==t)) for t in [1,2,4,8,16,32]}
 for x in rr:
  assert x['trial']==results['trial'][x['key']] and x['time']==results['time'].get(x['key'])
  assert x['tier']==tier(x['trial']['n_pass'],x['time']['n_pass'] if x['time'] else 0)
 stable[group]=sum(all(x['tier'] in 'AB' for x in rr if x['target']['uid']==uid) for uid in {x['target']['uid'] for x in rr})
 direction_changes[group]=[x['key'] for x in rr if np.sign(x['trial']['mean_observed'])!=x['target']['trial']['direction']]
report['tau']={'checkpoints_checked':counts,'tiers':tau,'all_offsets_AB':stable,'direction_changes':direction_changes}
print('Tau: all saved-array arithmetic and checkpoint checks passed',flush=True)
# History: observed and surrogate grid selection, directional p values and final tiers.
R=W/'PhiID_Clinical_History_Preflight_2026-10-30';s=read(R/'specification.json');sig=sha(R/'specification.json')
hashes(s['hashes']);hashes(read(R/'controls_execution_authorization.json')['hashes']);hashes(read(R/'controls_launch_preservation.json'))
assert read(R/'controls_exit.json')['returncode']==0
observed=read(R/'observed_four_contrast.json')+read(R/'observed_genuine_interaction.json');o={};hcount=0
def gridrow(ti,kind,rep,folder,saved):
 global hcount
 a=np.array(list(pool.map(lambda d:ck(R/folder/f'dyad{d}_target{ti:02d}_{kind}_{rep:03d}.npy',sig,(21,2)),s['dyads'])));hcount+=15
 scores=a[:,:,0].mean(0);best=min(range(21),key=lambda j:(scores[j],*s['grid'][j]))
 assert saved['selected_grid_index']==best and [saved['H'],saved['tau']]==s['grid'][best]
 np.testing.assert_array_equal(scores,saved['mean_scores']);np.testing.assert_array_equal(a[:,best,1],saved['dyad_contrasts'])
 return a[:,best,1]
for x in observed:o[x['target_index']]=gridrow(x['target_index'],'observed',0,'observed_checkpoints',x['selections'][0])
tests={}
for kind in ['trial','time']:
 tests[kind]={};records=read(R/f'{kind}_four_contrast.json')+read(R/f'{kind}_genuine_interaction.json')
 for ii,x in enumerate(records):
  ti=x['target_index'];assert x['target']==s['targets'][ti]
  null=np.array([gridrow(ti,kind,rep,'control_checkpoints',v) for rep,v in enumerate(x['selections'])]);assert len(null)==100
  gp,dp=pv(o[ti],null,x['target']['trial']['direction'])
  tests[kind][ti]=dict(target_index=ti,group_p=gp,dyad_p=dp.tolist(),n_pass=int(sum(dp<=.05)))
  if ii%10==0:print(f'History {kind}: {ii+1}/{len(records)} targets checked',flush=True)
 assert len(records)==(54 if kind=='trial' else 49)
assert list(tests['trial'].values())==read(R/'trial_tests.json')
assert [i for i,x in tests['trial'].items() if x['group_p']<=.1]==read(R/'time_gate.json')
hs={}
for group,fn in [('main','four_contrast_history_tiers.json'),('interaction','genuine_interaction_history_tiers.json')]:
 rr=read(R/fn)
 for x in rr:
  ti=x['trial']['target_index'];tr=tests['trial'][ti];tm=tests['time'].get(ti)
  assert x['trial']==tr and x['history_sensitivity_tier']==tier(tr['n_pass'],tm['n_pass'] if tm else 0)
  assert x['time_shift']==({k:v for k,v in tm.items() if k!='target_index'} if tm else None)
 hs[group]={'tiers':dict(Counter(x['history_sensitivity_tier'] for x in rr)),'time_tested':sum(x['time_shift'] is not None for x in rr),'observed_selections':dict(Counter(f"H{x['observed_selection']['H']}/tau{x['observed_selection']['tau']}" for x in rr))}
report['history']={'checkpoints_checked':hcount,'groups':hs}
print('History verification passed',flush=True)
# Chronological half comparisons: recompute statistics from full and matched-trial cells.
R=W/'PhiID_Clinical_RoleSwap_Preflight_2026-10-29';s=read(R/'specification.json');sig=sha(R/'specification.json');hashes(s['hashes'])
rr=read(R/'four_contrast_results.json')+read(R/'genuine_interaction_results.json');by={x['target']['uid']:x for x in rr}
signs=1-2*((np.arange(32768)[:,None]>>np.arange(15))&1);boot=np.random.default_rng(20261027).integers(0,15,(10000,15));ps=[]
for ti,t in enumerate(s['targets']):
 x=by[t['uid']];a=np.array([[ck(R/'checkpoints'/f'dyad{d}_target{ti}_rep{k}.npy',sig,(2,4)) for k in range(-1,100)] for d in s['dyads']]);cv=a@np.array(s['coefficients'][t['contrast']]);delta=cv[:,:,1]-cv[:,:,0];D=delta[:,0]
 p=float(np.mean(abs(signs@D/15)>=abs(D.mean())-1e-14));ps.append(p);assert p==x['p']
 for actual,saved in [(D,x['dyad_changes']),(D.mean(),x['change']),(D.mean()/D.std(ddof=1),x['dz']),(np.percentile(D[boot].mean(1),[2.5,97.5]),x['bootstrap_95']),(delta[:,1:].mean(0),x['subsample_mean_changes'])]:np.testing.assert_allclose(actual,saved,atol=1e-13,rtol=1e-12)
adj=0
for rank,j in enumerate(np.argsort(ps)):
 adj=max(adj,min(1,ps[j]*(11-rank)));assert adj==by[s['targets'][j]['uid']]['holm_p']
report['task_half']={'checkpoints_checked':16665,'holm_significant':sum(x['holm_p']<=.05 for x in rr),'holm_range':[min(x['holm_p'] for x in rr),max(x['holm_p'] for x in rr)]}
print('Task-half verification passed',flush=True)
# Diagnostics: checkpoint integrity and independently aggregate stored trial-level diagnostic outputs.
R=W/'PhiID_Clinical_Diagnostics_Preflight_2026-10-29';s=read(R/'specification.json');sig=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest();hashes(s['hashes'])
rr=read(R/'four_contrast_diagnostics.json')+read(R/'genuine_interaction_diagnostics.json');dc=0
def stat(v):
 v=np.array([x for x in v if x is not None],float);v=v[np.isfinite(v)]
 return dict(n=len(v),median_ms=float(np.median(v)) if len(v) else None,q25_ms=float(np.percentile(v,25)) if len(v) else None,q75_ms=float(np.percentile(v,75)) if len(v) else None)
for ti,t in enumerate(s['targets']):
 for actor in ['Leader','Follower']:
  for span in ['feature_window','full_epoch']:
   med=[]
   for d in s['dyads']:
    arrays=[ck(p,sig,allow_nan=True) for p in sorted((R/'checkpoints').glob(f'dyad{d}_target{ti:02d}_{actor}_{span}_*.npy'))];dc+=len(arrays);assert len(arrays) in [2,4]
    med.append([stat([stat(a[:,j])['median_ms'] for a in arrays])['median_ms'] for j in range(3)])
   actual={name:stat([m[j] for m in med]) for j,name in enumerate(['acf_zero','acf_minimum','mi_minimum'])}
   saved=next(x for x in rr if x['target']['uid']==t['uid'] and x['actor']==actor and x['span']==span)
   assert actual==saved['metrics']
report['diagnostics']={'checkpoints_checked':dc,'summary_rows_reproduced':len(rr),'missing_diagnostic_values_preserved':True}
report['passed']=True
(OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('ALL CHECKS PASSED',flush=True)
