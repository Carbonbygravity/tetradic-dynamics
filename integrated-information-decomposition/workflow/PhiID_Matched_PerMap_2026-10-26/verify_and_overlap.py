"""Verify matched results and compute normative geometry overlap, without new inference."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,sys,collections,time
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import runner as r
e=r.engine;core=r.core
OUT=ROOT/'geometry_overlap';OUT.mkdir(exist_ok=False)
summary={};inventory=[];checks=0
for separate,folder in [(False,'matched_results'),(True,'separate_interaction')]:
 for path in sorted((ROOT/folder).glob('*.json')):
  exported=json.loads(path.read_text());source=Path(exported['source']['path']);assert e.digest(source)==exported['source']['sha256']
  saved=json.loads(source.read_text());base=source.parent.parent;config=json.loads((base/'specification.json').read_text());sig=e.digest(base/'specification.json');assert saved['signature']==sig
  name=exported['contrast'];item=config['array_files'][name];assert e.digest(base/item['path'])==item['sha256'];x=np.load(base/item['path'],mmap_mode='r')
  null=[]
  for lo in range(0,5000,25):
   v=e.checkpoint(source.parent/f'null_{lo:04d}_{lo+25:04d}.npy',sig,(25,3264));assert v is not None;null.append(v);checks+=1
  null=np.concatenate(null);key=lambda c:(c['map'],c['start'],c['stop'],c['sign'])
  original={key(c):c for c in saved['clusters']};displayed={key(c):c for c in exported['clusters']};assert original.keys()==displayed.keys()
  observed=[]
  for lo in range(0,3264,96):
   z=np.asarray(x[:,lo:lo+96]);t=stats.ttest_1samp(z,0,axis=0).statistic
   for c in core.cluster_masses(t,stats.t.ppf(.975,14)):
    c['map']+=lo;s=original[key(c)];d=displayed[key(c)];np.testing.assert_allclose(c['mass'],s['mass'],rtol=1e-10,atol=1e-9)
    cutoff=s['mass']-1e-12*max(1.,abs(s['mass']));p=float(np.mean(null[:,c['map']]>=cutoff))
    assert p==s['p_historical_per_map']==d['p_within_map'];assert d['coordinates']==s['coordinates'];assert d['start_ms']==config['times_ms'][d['start']] and d['end_ms']==config['times_ms'][d['stop']-1];observed.append(key(c))
  assert len(observed)==len(original)
  survivors=[dict(c,cluster_key=f"{name}:{c['map']}:{c['start']}:{c['stop']}:{c['sign']}") for c in exported['clusters'] if c['p_within_map']<=.05]
  assert len(survivors)==exported['significant_count']
  counts=dict(collections.Counter(c['coordinates']['geometry'] for c in survivors))
  atoms=dict(collections.Counter(c['coordinates']['atom'] for c in survivors))
  direction=dict(collections.Counter('positive' if c['sign']==1 else 'negative' for c in survivors))
  record=dict(separate_interaction=separate,total=len(survivors),by_geometry=counts,by_atom=atoms,by_direction=direction)
  inventory.append(dict(path=str(path),sha256=e.digest(path),source=str(source),source_sha256=e.digest(source)))
  if not separate:
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
e.save(OUT/'summary.json',dict(status='passed',verified_null_batches=checks,scope='All five contrast exports checked against saved inputs, observed statistics and null batches. Permutations and surrogates not rerun. Geometry overlap calculated for the four matched contrasts only; no tier assignment.',sources=inventory,contrasts=summary))
(OUT/'README.txt').write_text('Normative geometry overlap, 26 October 2026.\nAll four matched per-map survivor sets are matched by contrast, Leader-indexed ROI, band and atom, with at least one common sample. Half-open saved sample indices determine intersections; reported times use the saved -999..999 ms axis. Both actual ROI pairings remain in each record. Same/opposite directions are reported separately. Multiple cluster pairs per feature are retained. Geometry-only means no temporal match in the other geometry. The genuine interaction is summarized separately and excluded from overlap and target selection. No surrogate tests, recurrence tiers, clinical inference or new significance tests were performed. summary.json records input hashes and verification scope.\n',encoding='utf-8')
r.log('Verified all five contrast exports and 1,000 null-batch checkpoints; completed geometry overlap for the four matched normative contrasts. Genuine interaction excluded; controls and tiering pending.')
