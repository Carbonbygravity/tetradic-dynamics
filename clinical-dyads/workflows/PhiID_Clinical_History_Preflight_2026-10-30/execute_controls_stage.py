import sys,os,time,json,importlib.util,subprocess
from pathlib import Path
sp=importlib.util.spec_from_file_location('clinical_history_controls_runner',Path(__file__).with_name('runner.py'))
h=importlib.util.module_from_spec(sp);sys.modules[sp.name]=h;sp.loader.exec_module(h)
import numpy as np
R=h.R
read=h.read;save=h.save;sha=h.sha;checkpoint=h.checkpoint;path_for=h.path_for;GRID=h.GRID
def summarize_stage(kind):
 spec=read(R/'specification.json');sig=sha(R/'specification.json');tis=list(range(len(spec['targets']))) if kind!='time' else read(R/'time_gate.json');result=[]
 for ti in tis:
  reps=[0] if kind=='observed' else list(range(100));selections=[]
  for rep in reps:
   vv=[checkpoint(path_for('control_checkpoints',d,ti,kind,rep),sig) for d in spec['dyads']];assert all(v is not None for v in vv);v=np.array(vv);best=int(np.argmin(v[:,:,0].mean(0)))
   selections.append(dict(replicate=rep,selected_grid_index=best,H=GRID[best][0],tau=GRID[best][1],mean_scores=v[:,:,0].mean(0).tolist(),dyad_contrasts=v[:,best,1].tolist()))
  result.append(dict(target_index=ti,target=spec['targets'][ti],selections=selections))
 for title,interaction in [('four_contrast',False),('genuine_interaction',True)]:save(R/f'{kind}_{title}.json',[x for x in result if (x['target']['contrast']=='Role_by_Condition')==interaction])
 if kind=='trial':
  observed=read(R/'observed_four_contrast.json')+read(R/'observed_genuine_interaction.json');om={x['target_index']:x for x in observed};gate=[];tests=[]
  for row in result:
   ti=row['target_index'];direction=row['target']['trial']['direction'];obs=np.array(om[ti]['selections'][0]['dyad_contrasts']);null=np.array([s['dyad_contrasts'] for s in row['selections']]);gp=float((1+sum(direction*null.mean(1)>=direction*obs.mean()))/101);dp=(1+(direction*null>=direction*obs).sum(0))/101
   tests.append(dict(target_index=ti,group_p=gp,dyad_p=dp.tolist(),n_pass=int(sum(dp<=.05))))
   if gp<=.10:gate.append(ti)
  save(R/'trial_tests.json',tests);save(R/'time_gate.json',gate)
def finalize(spec):
 observed=h.read(R/'observed_four_contrast.json')+h.read(R/'observed_genuine_interaction.json');obs={x['target_index']:x for x in observed}
 timerecs=h.read(R/'time_four_contrast.json')+h.read(R/'time_genuine_interaction.json');tm={x['target_index']:x for x in timerecs};trial={x['target_index']:x for x in h.read(R/'trial_tests.json')};results=[]
 for ti,t in enumerate(spec['targets']):
  direction=t['trial']['direction'];o=np.array(obs[ti]['selections'][0]['dyad_contrasts']);ts=None
  if ti in tm:
   n=np.array([s['dyad_contrasts'] for s in tm[ti]['selections']]);dp=(1+(direction*n>=direction*o).sum(0))/101
   ts=dict(group_p=float((1+sum(direction*n.mean(1)>=direction*o.mean()))/101),dyad_p=dp.tolist(),n_pass=int(sum(dp<=.05)))
  nt=trial[ti]['n_pass'];ns=ts['n_pass'] if ts else 0;tier='A' if nt>=4 and ns>=3 else 'B' if nt>=4 else 'C'
  results.append(dict(target=t,primary_tier=t['tier'],history_sensitivity_tier=tier,observed_selection=obs[ti]['selections'][0],trial=trial[ti],time_shift=ts))
 for label,flag in [('four_contrast',False),('genuine_interaction',True)]:h.save(R/f'{label}_history_tiers.json',[x for x in results if (x['target']['contrast']=='Role_by_Condition')==flag])
 expected=len(spec['dyads'])*len(spec['targets'])*100+1500*len(tm);paths=list((R/'control_checkpoints').glob('*.npy'));assert len(paths)==expected
 for p in paths:assert h.checkpoint(p,h.sha(R/'specification.json')) is not None
 h.save(R/'completion_verification.json',dict(state='complete',targets=len(spec['targets']),dyads=15,checkpoints=expected,genuine_interaction_separate=True,primary_results_unchanged=True))

def main():
 assert len(sys.argv)==3 and sys.argv[2]=='--approved-full-run','Expected a stage name and --approved-full-run'
 kind=sys.argv[1];assert kind in ['trial','time']
 auth=read(R/'controls_execution_authorization.json');assert auth['full_run_approved'] is True
 for p,digest in auth['hashes'].items():assert sha(p)==digest
 assert read(R/'controls_preparation_verification.json')['status']=='passed'
 assert read(R/'observed_independent_verification.json')['status']=='passed'
 spec=read(R/'specification.json')
 for p,digest in spec['hashes'].items():assert sha(p)==digest
 for p,digest in read(R/'controls_launch_preservation.json').items():assert sha(p)==digest
 if kind=='time':assert read(R/'trial_independent_verification.json')['status']=='passed'
 tis=list(range(len(spec['targets']))) if kind=='trial' else read(R/'time_gate.json')
 lock=R/'controls_stage.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  (R/'control_checkpoints').mkdir(exist_ok=True)
  result=h.pool([(d,tis,[kind],list(range(100)),'control_checkpoints',float('inf')) for d in spec['dyads']]) if tis else dict(workers=[],wall_seconds=0)
  save(R/f'{kind}_attempt_{time.time_ns()}.json',result)
  summarize_stage(kind)
  if kind=='time':finalize(spec)
 finally:lock.unlink()
if __name__=='__main__':main()
