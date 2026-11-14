import time,json,sys,importlib.util
from pathlib import Path
sp=importlib.util.spec_from_file_location('history_runner',Path(__file__).with_name('runner.py'));h=importlib.util.module_from_spec(sp);sys.modules['history_runner']=h;sp.loader.exec_module(h)
R=h.R
def main():
 spec=h.read(R/'specification.json');b=spec['benchmark'];folder='sustained_checkpoints';(R/folder).mkdir(exist_ok=True)
 cases=list(zip(b['dyads'],[b['target_indices'][0],b['target_indices'][1],b['target_indices'][2],b['target_indices'][1]]))
 jobs=[(d,[ti],['trial','time'],list(range(10)),folder,240) for d,ti in cases]
 result=h.pool(jobs);h.save(R/'sustained_benchmark.json',result)
 assert sum(w['computed'] for w in result['workers'])==80
 import numpy as np
 maxerr=0
 for d,ti in cases:
  for kind in ['trial','time']:
   a=h.checkpoint(h.path_for(folder,d,ti,kind,0),h.sha(R/'specification.json'));ref=h.checkpoint(h.path_for('benchmark_checkpoints',d,ti,kind,0),h.sha(R/'specification.json'));np.testing.assert_array_equal(a,ref);maxerr=max(maxerr,float(abs(a-ref).max()))
 records=[x for w in result['workers'] for x in w['records']];assert len(list((R/folder).glob('*.npy')))==80
 for p in (R/folder).glob('*.npy'):assert h.checkpoint(p,h.sha(R/'specification.json')) is not None
 rates={kind:np.mean([x['compute_seconds'] for x in records if x['kind']==kind]) for kind in ['trial','time']};rate=80/result['wall_seconds']
 h.save(R/'sustained_verification.json',dict(status='passed',grid_count=80,configurations=1680,repeat0_matches_initial_benchmark=True,max_repeat0_error=maxerr,wall_seconds=result['wall_seconds'],grids_per_wall_second=rate,mean_compute_seconds_by_kind=rates,projected_trial_hours=25500/rate/3600,projected_trial_and_all_time_hours=51000/rate/3600,scope='Four dyad/target cases, ten trial and ten time surrogates each, cached source inputs. Linear planning estimates, not measured full-cohort runtimes.'))
 h.log('Sustained four-worker benchmark complete; all 80 grids verified, no full run launched')
if __name__=='__main__':main()
