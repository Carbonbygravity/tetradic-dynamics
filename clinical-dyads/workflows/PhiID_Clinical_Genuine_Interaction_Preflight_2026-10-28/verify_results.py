"""Independent reconstruction of finished clinical interaction cluster summaries."""
from pathlib import Path
import json,hashlib,csv,collections
import numpy as np
from scipy import stats
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert read(R/'completion_verification.json')['state']=='complete'
cfg=read(R/'specification.json');assembly=read(R/'assembly.json');res=read(R/'results.json')
for p,h in cfg['hashes'].items():assert sha(Path(p))==h,p
assert len(assembly['sources'])==1020 and assembly['array_sha256']==sha(R/'interaction.npy')
assert len({x['path'] for x in assembly['sources']})==1020
sig=sha(R/'assembly.json');parts=[]
for lo in range(0,5000,25):
 p=R/'null_checkpoints'/f'null_{lo:04d}.npy';m=read(p.with_suffix('.complete.json'));v=np.load(p,allow_pickle=False)
 assert m['signature']==sig and m['sha256']==sha(p) and v.shape==(25,3264) and np.isfinite(v).all() and (v>=0).all()
 parts.append(v)
null=np.concatenate(parts);x=np.load(R/'interaction.npy',mmap_mode='r');assert x.shape==(15,3264,1999)
maps=read(Path(cfg['maps']));threshold=stats.t.ppf(.975,14)
expected={(a['map'],a['start'],a['stop'],a['sign']):a for a in res['clusters']};assert len(expected)==len(res['clusters'])
seen=set();significant=[];max_mass_error=0.;max_p_error=0.
for lo in range(0,3264,48):
 tt=stats.ttest_1samp(np.asarray(x[:,lo:lo+48]),0,axis=0).statistic;assert np.isfinite(tt).all()
 for j,t in enumerate(tt):
  mi=lo+j
  for direction in [1,-1]:
   mask=direction*t>threshold;edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
   for start,stop in zip(starts,stops):
    key=(mi,int(start),int(stop),direction);record=expected[key];seen.add(key)
    mass=float((direction*t[start:stop]).sum());err=abs(mass-record['mass']);max_mass_error=max(max_mass_error,err);assert np.isclose(mass,record['mass'],rtol=1e-9,atol=1e-9)
    cutoff=mass-1e-12*max(1.,abs(mass));p=float((null[:,mi]>=cutoff).mean());max_p_error=max(max_p_error,abs(p-record['p_within_map']));assert p==record['p_within_map']
    assert record['coordinates']==maps[mi] and record['start_ms']==start-999 and record['end_ms']==stop-1000
    if p<=.05:
     z=np.asarray(x[:,mi,start:stop]).mean(1);coord=maps[mi];significant.append(dict(map=mi,geometry=coord['geometry'],atom=coord['atom'],roi=coord['roi'],follower_roi=coord['follower_roi'],band=str(coord['band']),start_ms=int(start-999),end_ms=int(stop-1000),sign=direction,p_within_map=p,mass=record['mass'],mean_window_interaction=float(z.mean()),window_dz=float(z.mean()/z.std(ddof=1))))
assert seen==set(expected) and len(significant)==res['significant_count']
summary=dict(total=len(significant),by_geometry=dict(collections.Counter(x['geometry'] for x in significant)),by_atom=dict(collections.Counter(x['atom'] for x in significant)),by_geometry_atom={g:dict(collections.Counter(x['atom'] for x in significant if x['geometry']==g)) for g in cfg.get('geometries',['homologous','cross_hemisphere'])})
with (R/'significant_clusters.csv').open('x',encoding='utf-8',newline='') as f:
 fields=list(significant[0]) if significant else ['map','geometry','atom','roi','follower_roi','band','start_ms','end_ms','sign','p_within_map','mass','mean_window_interaction','window_dz']
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(sorted(significant,key=lambda x:(x['p_within_map'],x['geometry'],x['map'],x['start_ms'])))
report=dict(status='passed',source_tile_inventory=1020,checkpoint_hashes_signatures_shapes_verified=200,assignments=5000,maps=3264,observed_clusters_reconstructed=len(seen),max_mass_error=max_mass_error,max_p_error=max_p_error,source_array_hash_verified=True,existing_dependencies_primary_tiers_unchanged=True,counts=summary,scope='Independent SciPy t statistics and contiguous-cluster reconstruction with saved null maxima; no second full permutation calculation. Source hashes checked during assembly; this check verifies the assembly inventory and array hash.',overlap_controls_run=False)
with (R/'independent_verification.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
with (R/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write('2026-10-28 — Independently reconstructed observed interaction clusters, p values and count summaries from the verified assembled array and 200 null checkpoints. Checked unchanged frozen dependencies and primary tiers. Saved significant_clusters.csv and independent_verification.json. Geometry overlap and shuffle controls remain unlaunched.\n')
print(json.dumps(report),flush=True)
