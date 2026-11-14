"""Prepared staged production entry point; not authorized by benchmark approval."""
import sys,time,os,importlib.util
from pathlib import Path
sp=importlib.util.spec_from_file_location('history_runner',Path(__file__).with_name('runner.py'));h=importlib.util.module_from_spec(sp);sys.modules['history_runner']=h;sp.loader.exec_module(h)
import numpy as np
R=h.R
def main():
 assert len(sys.argv)==3 and sys.argv[2]=='--approved-full-run','Separate full-stage approval required'
 kind=sys.argv[1];assert kind in ['observed','trial','time'];spec=h.read(R/'specification.json');assert h.read(R/'benchmark_verification.json')['status']=='passed'
 for p,v in spec['hashes'].items():assert h.sha(p)==v
 lock=R/'production.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  tis=list(range(17)) if kind!='time' else h.read(R/'time_gate.json');reps=[0] if kind=='observed' else list(range(100));(R/'production_checkpoints').mkdir(exist_ok=True)
  h.log('Approved stage started/resumed: '+kind)
  result=h.pool([(d,tis,[kind],reps,'production_checkpoints',float('inf')) for d in spec['dyads']]);h.save(R/f'{kind}_attempt_{time.time_ns()}.json',result)
  h.summarize_stage(kind)
  if kind=='time':finalize(spec)
  h.log('Stage complete and summaries written separately by contrast family: '+kind)
 finally:lock.unlink()
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
 expected=255*101+1500*len(tm);paths=list((R/'production_checkpoints').glob('*.npy'));assert len(paths)==expected
 for p in paths:assert h.checkpoint(p,h.sha(R/'specification.json')) is not None
 h.save(R/'completion_verification.json',dict(state='complete',targets=17,excluded=2,dyads=15,checkpoints=expected,genuine_interaction_separate=True,primary_results_unchanged=True))
if __name__=='__main__':main()
