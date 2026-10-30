import json,hashlib,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
kind=sys.argv[1];assert kind in ['trial','time'];spec=read(R/'specification.json');sig=sha(R/'specification.json')
for p,h in spec['hashes'].items():assert sha(p)==h
records=read(R/f'{kind}_four_contrast.json')+read(R/f'{kind}_genuine_interaction.json');tis=list(range(17)) if kind=='trial' else read(R/'time_gate.json')
assert sorted(x['target_index'] for x in records)==sorted(tis)
observed=read(R/'observed_four_contrast.json')+read(R/'observed_genuine_interaction.json');obs={x['target_index']:np.array(x['selections'][0]['dyad_contrasts']) for x in observed}
tests=[];count=0
for row in records:
 ti=row['target_index'];assert row['target']==spec['targets'][ti];assert len(row['selections'])==100;null=[]
 for rep in range(100):
  vv=[]
  for d in spec['dyads']:
   p=R/'production_checkpoints'/f'dyad{d}_target{ti:02d}_{kind}_{rep:03d}.npy';m=read(p.with_suffix('.complete.json'));a=np.load(p,allow_pickle=False)
   assert m['sha256']==sha(p) and m['signature']==sig and a.shape==(21,2) and m['shape']==[21,2] and np.isfinite(a).all();vv.append(a);count+=1
  a=np.array(vv);scores=a[:,:,0].mean(0);best=min(range(21),key=lambda j:(scores[j],*spec['grid'][j]));s=row['selections'][rep]
  assert s['replicate']==rep and s['selected_grid_index']==best and [s['H'],s['tau']]==spec['grid'][best];np.testing.assert_array_equal(scores,s['mean_scores']);np.testing.assert_array_equal(a[:,best,1],s['dyad_contrasts']);null.append(a[:,best,1])
 null=np.array(null);direction=row['target']['trial']['direction'];o=obs[ti];gp=float((1+sum(direction*null.mean(1)>=direction*o.mean()))/101);dp=(1+(direction*null>=direction*o).sum(0))/101
 tests.append(dict(target_index=ti,group_p=gp,dyad_p=dp.tolist(),n_pass=int(sum(dp<=.05))))
if kind=='trial':
 assert tests==read(R/'trial_tests.json');assert [x['target_index'] for x in tests if x['group_p']<=.10]==read(R/'time_gate.json')
else:
 tiers=read(R/'four_contrast_history_tiers.json')+read(R/'genuine_interaction_history_tiers.json');tm={x['target_index']:x for x in tests};tr={x['target_index']:x for x in read(R/'trial_tests.json')}
 for ti,t in enumerate(spec['targets']):
  x=next(x for x in tiers if x['target']['uid']==t['uid']);ts=tm.get(ti);nt=tr[ti]['n_pass'];ns=ts['n_pass'] if ts else 0
  assert x['history_sensitivity_tier']==('A' if nt>=4 and ns>=3 else 'B' if nt>=4 else 'C') and x['primary_tier']==t['tier']
  if ts:assert x['time_shift']=={k:v for k,v in ts.items() if k!='target_index'}
for p,h in read(R/'controls_launch_preservation.json').items():assert sha(p)==h
result=dict(status='passed',stage=kind,targets=len(tis),dyads=15,replicates=100,checkpoints=count,grid_selections_reproduced=100*len(tis),primary_and_observed_hashes_unchanged=True,genuine_interaction_separate=True)
with (R/f'{kind}_independent_verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
