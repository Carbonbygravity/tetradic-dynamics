"""Prepare and preflight only; production controls require an explicit run command."""
import ast,json,sys,runpy,time
from pathlib import Path
import os
import numpy as np
import runner as r
ROOT=r.ROOT;W=r.W;OVER=W/'PhiID_Clinical_Geometry_Overlap_2026-10-27'
def prepare():
 summary=json.loads((OVER/'summary.json').read_text());assert summary['status']=='passed' and summary['verified_null_batches']==800
 for s in summary['sources']:assert r.sha(s['path'])==s['sha256']
 original=json.loads((r.PROD/'production/specification.json').read_text());r.c.prod.verify(original)
 assert original['dyads']==list(range(1,16))
 maps=json.loads((W/'PhiID_Clinical_Matched_PerMap_2026-10-26/map_inventory.json').read_text());rois=[maps[i*48]['roi'] for i in range(68)]
 targets=[];excluded=[];sources={str(OVER/'summary.json'):r.sha(OVER/'summary.json')}
 def add(name,partition,cluster,start,stop,pair_id=None):
  xyz=cluster['coordinates'];row=dict(id=len(targets),contrast=name,partition=partition,pair_id=pair_id,geometry=xyz['geometry'],roi=xyz['roi'],follower_roi=xyz['follower_roi'],roi_index=rois.index(xyz['roi']),band=xyz['band'],band_index=r.BANDS.index(tuple(xyz['band'])),atom=xyz['atom'],atom_index=r.ATOMS.index(xyz['atom']),start=start,stop=stop,start_ms=start-999,end_ms=stop-1000,parent_cluster=cluster['cluster_key'])
  if stop-start<3:excluded.append(dict(row,reason='Fewer than three sampled centres'));return
  targets.append(row)
 for name in r.NAMES:
  p=OVER/(name+'.json');sources[str(p)]=r.sha(p);v=json.loads(p.read_text())
  for i,pair in enumerate(v['pairs']):
   for geom in ['homologous','cross_hemisphere']:add(name,'overlap',pair[geom],pair['start_index'],pair['stop_index_exclusive'],name+':pair'+str(i))
  for key in ['homologous_only','cross_hemisphere_only']:
   for row in v[key]:add(name,key,row,row['start'],row['stop'])
 assert all(t['contrast']!='Role_by_Condition' for t in targets)
 r.save(ROOT/'targets.json',targets);r.save(ROOT/'excluded_targets.json',excluded)
 files=[ROOT/x for x in ['runner.py','control_core.py','prepare.py','finish_controls.py','trial_generation.py']]
 files += [Path(r.c.prod.__file__),Path(r.c.common.__file__),*Path(r.c.prod.calc.__file__).parent.glob('*.py')]
 hashes={str(p):r.sha(p) for p in files};hashes.update(original['dependency_hashes'])
 spec=dict(cohort='clinical HC-PT',dyads=original['dyads'],contrasts=r.COEFF,cell_order=r.c.prod.CELLS,targets_sha256=r.sha(ROOT/'targets.json'),sources=sources,manifest=str(r.PROD/'input_manifest.json'),manifest_sha256=r.sha(r.PROD/'input_manifest.json'),production_spec_sha256=r.sha(r.PROD/'production/specification.json'),code_hashes=hashes,surrogates=100,seed=1337,rng_namespace='clinical',partner='physical P2 [PT], before actor alignment',trial_strata='chronological block and response period',time_shift='one circular lag per dyad and repetition, integers 20..999 inclusive, shared across ROIs/bands',p_value='Fixed parent-cluster direction: (1+count(sign*null>=sign*observed))/101',time_gate='direction-aligned cohort mean trial-shuffle p <= .10 independently per geometry-target',tiers=dict(alpha=.05,trial_min_dyads=4,time_min_dyads=3,overlap='separate tier per geometry; no combined tier'),observed_statistic='mean across saved sampled centres of pointwise production PhiID cell contrast',genuine_interaction='excluded',workers=1,versions=dict(python=r.platform.python_version(),numpy=np.__version__,scipy=r.scipy.__version__,h5py=r.h5py.__version__),numeric_tolerance=dict(rtol=1e-7,atol=1e-7),batch_size=10,authorization='Prepare and preflight clinical shuffle controls and recurrence tiering; no production launch.')
 r.save(ROOT/'specification.json',spec)
 r.save(ROOT/'direction_decision.json',dict(approved=True,scope='Use the same parent-cluster direction rule as the normative controls',postprocessor_sha256=r.sha(ROOT/'finish_controls.py')))
 r.log(f'Prepared {len(targets)} clinical geometry targets; {len(excluded)} short windows excluded. Genuine interaction excluded; production not launched.')
def test():
 spec=json.loads((ROOT/'specification.json').read_text());r.verify(spec)
 old=Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/'PhiID_Normative_Controls_2026-10-26'
 def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
 nc=funcs(old/'control_core.py');cc=funcs(ROOT/'control_core.py')
 assert all(cc[n]==nc[n] for n in ['estimate_batch','blocks_periods','cell_atoms'])
 nr=funcs(old/'runner.py');cr=funcs(ROOT/'runner.py')
 assert all(nr[n]==cr[n] for n in ['evaluate','stored','stage','checkpoint','publish','pvalue'])
 assert funcs(old/'finish_controls.py')['direction_summary']==funcs(ROOT/'finish_controls.py')['direction_summary']
 events=np.array(['LMR1','LMR1','FMR1','FMR1','LMA1','FMA1','LMA2','FMA2','LMR2','FMR2'])
 p,lag=r.c.perturbation(events,1,0,'trial');np.testing.assert_array_equal(p,r.c.perturbation(events,1,0,'trial')[0]);assert lag==0
 for rep in range(100):assert 20<=r.c.perturbation(events,1,rep,'time')[1]<=999
 assert np.allclose(r.pvalue(np.array([80.]),np.arange(100.)[:,None]),21/101)
 assert np.allclose(r.pvalue(np.array([-80.])*-1,-np.arange(100.)[:,None]*-1),21/101)
 r.preflight(spec)
 r.save(ROOT/'adaptation_tests.json',dict(status='passed',numerical_functions_ast_identical=True,directional_summary_ast_identical=True,positive_negative_tail=True,time_shift_bounds=True,deterministic_clinical_rng=True,adaptations=['Clinical production adapter and verified clinical actor manifest','Clinical dyads 1 through 15','Clinical input manifest uses participant rather than partner field','Seed 1337 with clinical cohort namespace','Bounded status-write retries'],production_started=False))
 r.log('Clinical preparation and numerical preflight passed. No production trial shuffles, time shifts or tiers launched.')
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='test':test()
 elif sys.argv[1]=='run':
  assert json.loads((ROOT/'adaptation_tests.json').read_text())['status']=='passed'
  runpy.run_path(str(ROOT/'trial_generation.py'),run_name='__main__')
  runpy.run_path(str(ROOT/'finish_controls.py'),run_name='__main__')
 else:raise ValueError(sys.argv[1])
