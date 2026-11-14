from pathlib import Path
import os
import json,hashlib,ast,csv,collections
O=Path(__file__).resolve().parent;W=O.parent
R=W/'PhiID_Clinical_Genuine_Interaction_Preflight_2026-10-28';C=W/'PhiID_Clinical_Controls_2026-10-27';N=Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/'PhiID_Normative_Interaction_Controls_2026-10-26'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n
 assert not p.exists(),p
 p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
protected=[R/'results.json',R/'independent_verification.json',C/'directional_tiers.json',N/'directional_tiers.json'];before={str(p):sha(p) for p in protected}
v=read(O/'Role_by_Condition_overlap.json');pairs=v['pairs'];onlyh=v['homologous_only'];onlya=v['cross_hemisphere_only'];targets=read(O/'targets.json');excluded=read(O/'excluded_targets.json')
# Independent exhaustive overlap audit from source clusters, including unmatched membership.
clusters=[c for c in read(R/'results.json')['clusters'] if c['p_within_map']<=.05]
key=lambda c:(c['map'],c['start'],c['stop'],c['sign'])
feature=lambda c:(c['coordinates']['roi'],tuple(c['coordinates']['band']),c['coordinates']['atom'])
expected=set()
for h in clusters:
 if h['coordinates']['geometry']!='homologous':continue
 for a in clusters:
  if a['coordinates']['geometry']!='cross_hemisphere' or feature(h)!=feature(a):continue
  intersection=set(range(h['start'],h['stop'])) & set(range(a['start'],a['stop']))
  if intersection:expected.add((key(h),key(a),min(intersection),max(intersection)+1))
actual={(key(p['homologous']),key(p['cross_hemisphere']),p['start_index'],p['stop_index_exclusive']) for p in pairs};assert actual==expected
matched={k for pair in expected for k in pair[:2]}
assert {key(c) for c in onlyh+onlya}=={key(c) for c in clusters if key(c) not in matched}
assert all(t['stop']-t['start']>=3 for t in targets)
assert all(t['stop']-t['start']<3 for t in excluded)
checks={}
for fn,names in {'runner.py':['load','evaluate','stored','stage','checkpoint','publish','pvalue'],'control_core.py':['estimate_batch','blocks_periods','cell_atoms','perturbation'],'finish_controls.py':['direction_summary']}.items():
 for name in names:
  assert funcs(C/fn)[name]==funcs(O/fn)[name];checks[fn+':'+name]='identical to verified clinical implementation'
for name in ['estimate_batch','blocks_periods','cell_atoms']:assert funcs(N/'control_core.py')[name]==funcs(O/'control_core.py')[name]
for fn in ['runner.py','control_core.py','finish_controls.py','trial_generation.py']:compile((O/fn).read_text(encoding='utf-8'),str(O/fn),'exec')
spec=read(C/'specification.json');spec.update(contrasts={'Role_by_Condition':[-1.,1.,-1.,1.]},targets_sha256=sha(O/'targets.json'),sources={**before,str(O/'Role_by_Condition_overlap.json'):sha(O/'Role_by_Condition_overlap.json')},genuine_interaction='Only contrast; separate from four primary contrasts',authorization='Interaction target preparation.')
spec['code_hashes']={p:h for p,h in spec['code_hashes'].items() if Path(p).parent!=C}
spec['code_hashes'].update({str(O/fn):sha(O/fn) for fn in ['runner.py','control_core.py','finish_controls.py','trial_generation.py']})
save('specification.json',spec);save('direction_decision.json',dict(approved=True,scope='Same parent-cluster direction rule as the four main contrasts',postprocessor_sha256=sha(O/'finish_controls.py')))
summary=dict(significant_clusters=len(clusters),homologous=74,cross_hemisphere=95,overlap_pairs=len(pairs),unique_overlap_features=len({feature(p['homologous']) for p in pairs}),same_direction=sum(p['same_direction'] for p in pairs),opposite_direction=sum(not p['same_direction'] for p in pairs),homologous_only=len(onlyh),cross_hemisphere_only=len(onlya),targets=len(targets),excluded_short=len(excluded),by_partition=dict(collections.Counter(t['partition'] for t in targets)),by_atom=dict(collections.Counter(t['atom'] for t in targets)),roi_count=len({t['roi_index'] for t in targets}))
save('summary.json',summary)
with (O/'overlap_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
 fields=['pair','atom','leader_roi','homologous_follower_roi','cross_follower_roi','band','start_ms','end_ms','homologous_sign','cross_sign','homologous_p','cross_p','homologous_tier','cross_tier'];wr=csv.DictWriter(f,fields);wr.writeheader()
 for i,p in enumerate(pairs):
  hc,ac=p['homologous'],p['cross_hemisphere'];x=hc['coordinates']
  wr.writerow(dict(pair=i,atom=x['atom'],leader_roi=x['roi'],homologous_follower_roi=x['follower_roi'],cross_follower_roi=ac['coordinates']['follower_roi'],band=x['band'],start_ms=p['start_ms'],end_ms=p['end_ms'],homologous_sign=hc['sign'],cross_sign=ac['sign'],homologous_p=hc['p_within_map'],cross_p=ac['p_within_map'],homologous_tier='Pending',cross_tier='Pending'))
assert all(sha(p)==h for p,h in before.items())
save('preparation_verification.json',dict(status='passed',scope='Exhaustive overlap reconstruction, unmatched membership, short-window rules, function AST comparison and syntax; no new numerical preflight or surrogates',function_checks=checks,original_source_hashes=before,original_sources_unchanged=True,production_started=False))
print(json.dumps(summary,indent=2))
