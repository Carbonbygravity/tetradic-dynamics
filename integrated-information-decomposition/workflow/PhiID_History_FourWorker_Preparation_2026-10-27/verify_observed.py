"""Verify observed-only grid and derive estimates; never launch controls."""
import json,hashlib,collections,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'production.lock').exists(),'Observed worker pool still active'
spec=read(R/'specification.json');sig=sha(R/'specification.json')
for p,h in spec['hashes'].items():assert sha(p)==h
prep=read(R/'preparation_completion.json')
for p,h in prep['files'].items():assert sha(p)==h
paths=list((R/'production_checkpoints').glob('*.npy'));assert len(paths)==255
expected={(d,i) for d in spec['dyads'] for i in range(17)};found=set();records=[];arrays={};hashes={}
for p in paths:
 m=read(p.with_suffix('.complete.json'));a=np.load(p,allow_pickle=False)
 assert m['kind']=='observed' and m['replicate']==0 and m['signature']==sig and m['sha256']==sha(p) and m['shape']==[21,2] and a.shape==(21,2) and np.isfinite(a).all()
 key=(m['dyad'],m['target_index']);assert key not in found;found.add(key);arrays[key]=a;records.append(m);hashes[str(p)]=m['sha256']
assert found==expected
main=read(R/'observed_four_contrast.json');inter=read(R/'observed_genuine_interaction.json');assert len(main)==14 and len(inter)==3
assert all(x['target']['contrast']!='Role_by_Condition' for x in main) and all(x['target']['contrast']=='Role_by_Condition' for x in inter)
summary=[]
for row in main+inter:
 i=row['target_index'];assert row['target']==spec['targets'][i];a=np.array([arrays[(d,i)] for d in spec['dyads']]);scores=np.mean(a[:,:,0],axis=0);choice=min(range(21),key=lambda j:(scores[j],*spec['grid'][j]));sel=row['selections'];assert len(sel)==1;sel=sel[0]
 assert choice==sel['selected_grid_index'] and spec['grid'][choice]==[sel['H'],sel['tau']]
 np.testing.assert_array_equal(scores,sel['mean_scores']);np.testing.assert_array_equal(a[:,choice,1],sel['dyad_contrasts'])
 summary.append(dict(target_index=i,uid=row['target']['uid'],contrast=row['target']['contrast'],atom=row['target']['atom'],roi=row['target']['roi'],geometry=row['target']['geometry'],H=sel['H'],tau=sel['tau'],score=float(scores[choice]),mean_contrast=float(a[:,choice,1].mean())))
attempts=list(R.glob('observed_attempt_*.json'));assert len(attempts)==1;execution=read(attempts[0]);assert sum(w['computed'] for w in execution['workers'])==255 and len(execution['workers'])==15
assert all(w['baseline_max_error']<1e-7 for w in execution['workers'])
bench={}
for p in (R/'benchmark_checkpoints').glob('*.complete.json'):
 m=read(p);bench[(m['dyad'],m['target_index'],m['kind'])]=m
ratios={k:[m['compute_seconds']/bench[(d,t,'observed')]['compute_seconds'] for (d,t,kind),m in bench.items() if kind==k] for k in ['trial','time']}
loads={d:sum({spec['targets'][m['target_index']]['roi_index']:m['load_seconds'] for m in records if m['dyad']==d}.values()) for d in spec['dyads']}
compute={d:sum(m['compute_seconds'] for m in records if m['dyad']==d) for d in spec['dyads']}
def makespan(costs):
 slots=[0.]*4
 for d in spec['dyads']:
  j=int(np.argmin(slots));slots[j]+=costs[d]
 return max(slots)
est={}
for k in ['trial','time']:
 rr=np.percentile(ratios[k],[0,50,100]);hours=[makespan({d:loads[d]+100*compute[d]*q for d in spec['dyads']})/3600 for q in rr]
 est[k]=dict(matched_benchmark_ratio_min_median_max=rr.tolist(),hours_min_median_max=hours)
time_worst=est['time']['hours_min_median_max'];trial=est['trial']['hours_min_median_max']
result=dict(status='passed',dyads=15,targets=17,main_targets=14,interaction_targets=3,checkpoints=255,configurations=5355,all_observed_summary_selections_reproduced=True,source_and_code_hashes_unchanged=True,observed_wall_seconds=execution['wall_seconds'],total_grid_compute_seconds=sum(compute.values()),max_full_window_baseline_error=max(w['baseline_max_error'] for w in execution['workers']),selected_settings=dict(collections.Counter(f"H{x['H']}_tau{x['tau']}" for x in summary)),runtime_estimates=est,trial_plus_all_time_hours_min_median_max=[a+b for a,b in zip(trial,time_worst)],estimate_method='All observed dyad/target timings; surrogate/observed ratios from matched four-worker benchmark; simulate same ordered 15-dyad scheduling on four workers with per-dyad source loads. Excludes unmeasured large checkpoint-I/O overhead and final verification. These are scenarios, not confidence bounds.',controls_launched=False)
for name,obj in [('observed_independent_verification.json',result),('observed_selection_review.json',summary),('observed_checkpoint_inventory.json',hashes)]:
 with (R/name).open('x') as f:json.dump(obj,f,indent=2)
with (R/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' Verified 255 observed checkpoints and 17 cohort-level grid selections; kept 14 main-contrast and 3 genuine-interaction targets separate. Updated surrogate runtime estimates from all observed timings. Shuffle controls not launched; existing results unchanged.\n')
print(json.dumps(result,indent=2))
