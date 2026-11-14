"""Freeze clinical targets and verify metadata only; no scientific sensitivity run."""
from pathlib import Path
import os
import json,hashlib,collections
import numpy as np,h5py
O=Path(__file__).resolve().parent;W=O.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (O/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
C=W/'PhiID_Clinical_Controls_2026-10-27';I=W/'PhiID_Clinical_Interaction_Overlap_Controls_2026-10-28';P=W/'PhiID_Clinical_Production_Preflight_2026-10-26'
deps=[C/'directional_tiers.json',I/'directional_tiers.json',C/'completion_verification.json',I/'completion_verification.json',P/'input_manifest.json',C/'specification.json',I/'specification.json',Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/'PhiID_Normative_Signal_Timescales_2026-10-27/run.py',Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/'PhiID_Normative_RoleSwap_Run_2026-10-27/run.py']
before={str(p):sha(p) for p in deps}; targets=[]
for group,folder in [('main',C),('interaction',I)]:
 rows=read(folder/'directional_tiers.json');assert read(folder/'completion_verification.json')['state']=='complete'
 for t in rows:
  if t['tier'] not in ['A','B']:continue
  nt=sum(v<=.05 for v in t['trial']['dyad_p']);ns=sum(v<=.05 for v in t['time_shift']['dyad_p']) if t['time_shift'] else 0
  assert t['tier']==('A' if nt>=4 and ns>=3 else 'B' if nt>=4 else 'C')
  targets.append(dict(t,uid=f"clinical:{group}:{t['id']}",analysis_group=group))
assert len(targets)==58
elig=[]
for t in targets:
 centres=np.arange(t['start']+1,t['stop']+1);hist=centres[(centres>=192)&(centres<=1936)];tau=centres[(centres>=32)&(centres<=1968)]
 elig.append(dict(uid=t['uid'],group=t['analysis_group'],diagnostics=t['tier']=='A',role_swap=t['atom'] in ['xtx','yty'],tau_eligible=len(tau)>=3,tau_centres=len(tau),history_eligible=len(hist)>=3,history_centres=len(hist),history_shortened=len(hist)!=len(centres),history_start_ms=int(hist[0]-1000) if len(hist) else None,history_end_ms=int(hist[-1]-1000) if len(hist) else None))
manifest=read(P/'input_manifest.json');counts=[];metadata=[]
for d in range(1,16):
 src=sorted([x for x in manifest if x['dyad']==d],key=lambda x:x['participant']);assert [x['participant'] for x in src]==['P1','P2']
 with h5py.File(src[0]['path'],'r') as a,h5py.File(src[1]['path'],'r') as b:
  for key in ['events','event_labels','selection','roi_labels','band_labels','times_seconds']:np.testing.assert_array_equal(a[key][:],b[key][:])
  assert a['data_raw'].shape==b['data_raw'].shape and a['data_raw'].shape[1:]==(68,6,2001)
  ev=a['event_labels'].asstr()[:].tolist();mapping=read(P/f'dyad{d}_epoch_actor_mapping.json');assert ev==[x['event'] for x in mapping]
  for e,row in zip(ev,mapping):
   assert row['X']==('P1' if e[-1]=='1' else 'P2') and row['Y']==('P2' if e[-1]=='1' else 'P1')
   assert int(row['block'])=={'MR1':1,'MA1':2,'MA2':3,'MR2':4}[e[1:]]
  for cell in ['LMR','LMA','FMA','FMR']:
   for half in ['1','2']:counts.append(dict(dyad=d,cell=cell,suffix=half,count=ev.count(cell+half)))
  metadata.append(dict(dyad=d,epochs=len(ev),paired_metadata_equal=True,actor_mapping_equal=True))
assert min(x['count'] for x in counts)>=10
save('targets_main.json',[t for t in targets if t['analysis_group']=='main']);save('targets_interaction.json',[t for t in targets if t['analysis_group']=='interaction']);save('eligibility.json',elig);save('half_cell_counts.json',counts);save('paired_metadata_verification.json',metadata)
summary={}
for group in ['main','interaction']:
 ee=[x for x in elig if x['group']==group]
 summary[group]=dict(AB=len(ee),diagnostics=sum(x['diagnostics'] for x in ee),role_swap=sum(x['role_swap'] for x in ee),tau_eligible=sum(x['tau_eligible'] for x in ee),history_eligible=sum(x['history_eligible'] for x in ee),history_shortened_eligible=sum(x['history_eligible'] and x['history_shortened'] for x in ee))
coeff={**read(C/'specification.json')['contrasts'],'Role_by_Condition':[-1,1,-1,1]}
unique=[t for t in targets if t['atom'] in ['xtx','yty']]
save('diagnostics_specification.json',dict(status='prepared_not_run',targets=[t for t in targets if t['tier']=='A'],dyads=list(range(1,16)),definitions=['first nonpositive autocorrelation lag','first local autocorrelation minimum','first local histogram-MI minimum'],mi_bins=16,min_paired_points=20,window_cap_ms=96,epoch_cap_ms=500,aggregation='trial median within cell/actor/dyad, equal-cell median within actor/dyad, equal-dyad median and quartiles',undefined='Retain missing crossings/minima as undefined; report available counts',mapping='Clinical source manifest; suffix 1 X=P1[HC], suffix 2 X=P2[PT]; actual geometry pairings',coefficients=coeff,preflight='Compare accelerated diagnostics with scalar definitions on real clinical inputs; benchmark before full launch',reference=str(deps[-2])))
save('role_swap_specification.json',dict(status='prepared_not_run',targets=unique,dyads=list(range(1,16)),cell_order=['LMR','LMA','FMA','FMR'],coefficients=coeff,estimand='D = original contrast computed from late-half PhiID minus original contrast computed from early-half PhiID',recompute='Recalculate pointwise PhiID within each suffix, at original target centres; do not split pooled nonlinear estimates',permutations=32768,test='Exact two-sided sign flips; include equality; sign-symmetry assumption',holm_family_size=len(unique),holm_family_uids=[t['uid'] for t in unique],alpha=.05,bootstrap=10000,bootstrap_seed=20261027,interval='Pointwise percentile 95%, whole-dyad resampling shared across targets',effects=['mean early','mean late','mean paired change','dz'],count_matched_repetitions=100,subsample_seed=1337,namespace='clinical role-swap',subsampling='Without replacement to minimum early/late count within each cell, paired P1/P2 rows and identical selection across targets; descriptive summaries only',qualification='Participant clinical identity, task role, chronological half and reversed condition order coincide. This is not an isolated HC-versus-PT effect; nonsignificance is not equivalence.',preflight='Verify pooled baseline against production, both halves and geometries, finite nonsingular covariances, deterministic subsamples and fresh-process checkpoint reuse; bounded benchmark before full launch',reference=str(deps[-1])))
save('summary.json',dict(targets=summary,paired_epochs=sum(x['epochs'] for x in metadata),half_cell_min=min(x['count'] for x in counts),half_cell_max=max(x['count'] for x in counts),role_swap_family=len(unique)))
assert all(sha(Path(p))==h for p,h in before.items());save('source_hashes.json',before)
save('preparation_verification.json',dict(status='passed',targets_recounted=58,paired_files_checked=30,half_cells_checked=120,source_hashes_unchanged=True,scope='Saved primary tiers, eligibility and paired source metadata/mapping. Full power-array contents not freshly hashed; no numerical sensitivity tests or benchmarks run.'))
print(json.dumps(read(O/'summary.json'),indent=2))
