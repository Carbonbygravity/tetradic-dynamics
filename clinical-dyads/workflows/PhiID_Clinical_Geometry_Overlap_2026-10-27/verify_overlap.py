import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,collections
from pathlib import Path
import numpy as np
from scipy import stats
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'PhiID_Clinical_Matched_PerMap_2026-10-26'
sys.path.insert(0,str(ROOT))
import runner as r
e=r.engine;core=r.core
config=json.loads((ROOT/'plan.json').read_text());r.verify_plan(config)
assert json.loads((ROOT/'status.json').read_text())['state']=='complete'
assembly=json.loads((ROOT/'assembly.json').read_text());sig=e.digest(ROOT/'assembly.json')
assert assembly['status']=='passed' and assembly['plan_sha256']==e.digest(ROOT/'plan.json')
maps=json.loads((ROOT/'map_inventory.json').read_text())
assert e.digest(ROOT/'map_inventory.json')==assembly['maps_sha256']
summary={};inventory=[];checks=0
for name in r.NAMES:
 source=ROOT/name/'results.json';saved=json.loads(source.read_text());assert saved['signature']==sig
 item=assembly['array_files'][name];assert e.digest(ROOT/item['path'])==item['sha256'];x=np.load(ROOT/item['path'],mmap_mode='r')
 null=[]
 for lo in range(0,5000,25):
  v=e.checkpoint(source.parent/f'null_{lo:04d}_{lo+25:04d}.npy',sig,(25,3264));assert v is not None;null.append(v);checks+=1
 null=np.concatenate(null);key=lambda c:(c['map'],c['start'],c['stop'],c['sign'])
 original={key(c):c for c in saved['clusters']};assert len(original)==len(saved['clusters']);observed=[]
 for lo in range(0,3264,96):
  z=np.asarray(x[:,lo:lo+96]);t=stats.ttest_1samp(z,0,axis=0).statistic
  for c in core.cluster_masses(t,stats.t.ppf(.975,14)):
   c['map']+=lo;s=original[key(c)];np.testing.assert_allclose(c['mass'],s['mass'],rtol=1e-10,atol=1e-9)
   cutoff=s['mass']-1e-12*max(1.,abs(s['mass']));p=float(np.mean(null[:,c['map']]>=cutoff))
   assert p==s['p_within_map'];assert s['coordinates']==maps[c['map']]
   assert s['start_ms']==config['times_ms'][s['start']] and s['end_ms']==config['times_ms'][s['stop']-1]
   observed.append(key(c))
 assert len(observed)==len(original)
 survivors=[dict(c,cluster_key=f"{name}:{c['map']}:{c['start']}:{c['stop']}:{c['sign']}") for c in saved['clusters'] if c['p_within_map']<=.05]
 assert len(survivors)==saved['significant_count']
 record=dict(total=len(survivors),by_geometry=dict(collections.Counter(c['coordinates']['geometry'] for c in survivors)))
 inventory.append(dict(path=str(source),sha256=e.digest(source)))
 h=[c for c in survivors if c['coordinates']['geometry']=='homologous'];a=[c for c in survivors if c['coordinates']['geometry']=='cross_hemisphere']
 feature=lambda c:(c['coordinates']['roi'],tuple(c['coordinates']['band']),c['coordinates']['atom'])
 pairs=[];matched=set()
 for hc in h:
  for ac in a:
   if feature(hc)!=feature(ac):continue
   start=max(hc['start'],ac['start']);stop=min(hc['stop'],ac['stop'])
   if stop<=start:continue
   ns=stop-start;union=max(hc['stop'],ac['stop'])-min(hc['start'],ac['start'])
   # Independent set-intersection check of endpoints and sample counts.
   assert ns==len(set(range(hc['start'],hc['stop']))&set(range(ac['start'],ac['stop'])))
   pairs.append(dict(homologous=hc,cross_hemisphere=ac,start_index=start,stop_index_exclusive=stop,start_ms=config['times_ms'][start],end_ms=config['times_ms'][stop-1],common_samples=ns,jaccard=ns/union,same_direction=hc['sign']==ac['sign']))
   matched.update([hc['cluster_key'],ac['cluster_key']])
 onlyh=[c for c in h if c['cluster_key'] not in matched];onlya=[c for c in a if c['cluster_key'] not in matched]
 record['overlap']=dict(pairs=len(pairs),unique_features=len({feature(p['homologous']) for p in pairs}),same_direction=sum(p['same_direction'] for p in pairs),opposite_direction=sum(not p['same_direction'] for p in pairs),homologous_only=len(onlyh),cross_hemisphere_only=len(onlya))
 e.save(OUT/(name+'.json'),dict(contrast=name,matching='Same contrast, Leader-indexed ROI, frequency band and atom; at least one common sample. Direction recorded, not used as a filter.',pairs=pairs,homologous_only=onlyh,cross_hemisphere_only=onlya))
 summary[name]=record;print(name,record,flush=True)
e.save(OUT/'summary.json',dict(status='passed',verified_null_batches=checks,contrasts=summary,sources=inventory,script_sha256=e.digest(Path(__file__)),scope='All four clinical contrasts reconstructed from saved arrays and null checkpoints. Same normative temporal geometry-overlap rules. No new permutations or surrogate inference. Genuine interaction excluded.'))
(OUT/'CHANGE_LOG.txt').write_text('2026-10-27: Verified 800 null batches, all observed cluster statistics, p values and coordinates; computed four matched clinical geometry overlap sets. All original outputs preserved.\n',encoding='utf-8')
