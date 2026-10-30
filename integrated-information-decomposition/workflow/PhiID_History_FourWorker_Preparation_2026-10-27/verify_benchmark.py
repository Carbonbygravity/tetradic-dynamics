import json,hashlib,ast
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=read(R/'specification.json');v=read(R/'benchmark_verification.json');assert v['status']=='passed' and not v['full_run_launched']
for p,h in spec['hashes'].items():assert sha(p)==h
grids=[];groups={'four_contrast':[],'genuine_interaction':[]}
for p in (R/'benchmark_checkpoints').glob('*.npy'):
 m=read(p.with_suffix('.complete.json'));a=np.load(p,allow_pickle=False);assert m['sha256']==sha(p) and m['signature']==sha(R/'specification.json') and a.shape==(21,2) and np.isfinite(a).all()
 t=spec['targets'][m['target_index']];key='genuine_interaction' if t['contrast']=='Role_by_Condition' else 'four_contrast';groups[key].append(dict(target=t,record=m,grid_scores_and_contrasts=a.tolist()));grids.append(m)
assert len(grids)==36 and len(groups['genuine_interaction'])==12 and len(groups['four_contrast'])==24
assert {(x['dyad'],x['target_index'],x['kind']) for x in grids}=={(d,t,k) for d in spec['benchmark']['dyads'] for t in spec['benchmark']['target_indices'] for k in ['observed','trial','time']}
for name,rows in groups.items():
 with (R/(name+'_benchmark_results.json')).open('x') as f:json.dump(rows,f,indent=2)
cached=read(R/'benchmark_cached_resume.json');assert sum(x['computed'] for x in cached['workers'])==0 and sum(x['reused'] for x in cached['workers'])==36
for p in [R/'runner.py',R/'execute_stage.py']:compile(p.read_text(),str(p),'exec')
record=dict(status='passed',grids_checked=36,configurations=756,all_dependency_hashes_match=True,all_expected_dyad_target_kind_combinations_present=True,full_run_started=False,prepared_script_hashes={str(p):sha(p) for p in [R/'runner.py',R/'execute_stage.py',R/'specification.json']},scope='Four-dyad benchmark checkpoint verification; full-stage orchestration is prepared, not executed')
with (R/'independent_verification.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
