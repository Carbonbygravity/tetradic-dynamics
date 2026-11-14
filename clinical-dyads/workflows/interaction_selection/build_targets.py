from pathlib import Path
import os
import json,hashlib,ast,csv,collections,datetime
W=Path(__file__).resolve().parents[1]; O=W/'PhiID_Clinical_Interaction_Overlap_Controls_2026-10-28'
O.mkdir(exist_ok=True)
outputs = ("Role_by_Condition_overlap.json", "targets.json", "excluded_targets.json")
if any((O / name).exists() for name in outputs):
    raise FileExistsError("Interaction target outputs already exist")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
    with (O/name).open('x',encoding='utf-8') as stream:
        json.dump(v,stream,indent=2)
R=W/'PhiID_Clinical_Genuine_Interaction_Preflight_2026-10-28'; C=W/'PhiID_Clinical_Controls_2026-10-27'; N=Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/'PhiID_Normative_Interaction_Controls_2026-10-26'
protected=[R/'results.json',R/'independent_verification.json',C/'directional_tiers.json',N/'directional_tiers.json']
before={str(p):sha(p) for p in protected}
v=json.loads((R/'independent_verification.json').read_text());assert v['status']=='passed'
clusters=[dict(c,cluster_key=f"Role_by_Condition:{c['map']}:{c['start']}:{c['stop']}:{c['sign']}") for c in json.loads((R/'results.json').read_text())['clusters'] if c['p_within_map']<=.05]
assert len(clusters)==169
h=[c for c in clusters if c['coordinates']['geometry']=='homologous']; a=[c for c in clusters if c['coordinates']['geometry']=='cross_hemisphere']
feature=lambda c:(c['coordinates']['roi'],tuple(c['coordinates']['band']),c['coordinates']['atom'])
pairs=[];matched=set()
for hc in h:
 for ac in a:
  if feature(hc)!=feature(ac):continue
  start=max(hc['start'],ac['start']);stop=min(hc['stop'],ac['stop'])
  common=set(range(hc['start'],hc['stop']))&set(range(ac['start'],ac['stop']))
  assert bool(common)==(stop>start)
  if not common:continue
  assert len(common)==stop-start and min(common)==start and max(common)==stop-1
  pairs.append(dict(homologous=hc,cross_hemisphere=ac,start_index=start,stop_index_exclusive=stop,start_ms=start-999,end_ms=stop-1000,common_samples=stop-start,same_direction=hc['sign']==ac['sign']))
  matched.update([hc['cluster_key'],ac['cluster_key']])
onlyh=[c for c in h if c['cluster_key'] not in matched];onlya=[c for c in a if c['cluster_key'] not in matched]
save('Role_by_Condition_overlap.json',dict(pairs=pairs,homologous_only=onlyh,cross_hemisphere_only=onlya,matching='Same contrast, Leader-indexed ROI, band, atom and at least one shared sample; direction recorded, not filtered. No tier requirement.'))
maps=json.loads((W/'PhiID_Clinical_Matched_PerMap_2026-10-26/map_inventory.json').read_text());rois=[maps[i*48]['roi'] for i in range(68)]
bands=[(4,7),(8,9),(10,12),(13,19),(20,29),(30,45)];atoms=['rtr','xtx','yty','sts'];targets=[];excluded=[]
def add(c,start,stop,partition,pair=None):
 x=c['coordinates'];row=dict(id=len(targets),contrast='Role_by_Condition',partition=partition,pair_id=pair,geometry=x['geometry'],roi=x['roi'],follower_roi=x['follower_roi'],roi_index=rois.index(x['roi']),band=x['band'],band_index=bands.index(tuple(x['band'])),atom=x['atom'],atom_index=atoms.index(x['atom']),start=start,stop=stop,start_ms=start-999,end_ms=stop-1000,parent_cluster=c['cluster_key'])
 assert c['start']<=start<stop<=c['stop']
 if stop-start<3:excluded.append(dict(row,reason='Fewer than three sampled centres'));return
 targets.append(row)
for i,p in enumerate(pairs):
 for g in ['homologous','cross_hemisphere']:add(p[g],p['start_index'],p['stop_index_exclusive'],'overlap',f'Role_by_Condition:pair{i}')
for label,rows in [('homologous_only',onlyh),('cross_hemisphere_only',onlya)]:
 for c in rows:add(c,c['start'],c['stop'],label)
save('targets.json',targets);save('excluded_targets.json',excluded)
