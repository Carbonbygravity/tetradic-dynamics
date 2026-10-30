"""Approved direction-aware summaries, gated time shifts, and separate geometry tiers."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib
from pathlib import Path
import numpy as np
import runner as r
ROOT=Path(__file__).resolve().parent
def direction_summary(spec,kind,targets,sig):
 ids=[t['id'] for t in targets];pos={tid:i for i,tid in enumerate(ids)};obs=np.empty((len(spec['dyads']),len(ids)));null=np.empty((len(spec['dyads']),100,len(ids)));out=ROOT/kind
 for di,d in enumerate(spec['dyads']):
  for roi in sorted({t['roi_index'] for t in targets}):
   rows=[t for t in targets if t['roi_index']==roi];idx=[pos[t['id']] for t in rows];prefix=f'dyad{d}_roi{roi:02d}'
   a=r.checkpoint(out/(prefix+'_observed.npy'),sig,(len(rows),));assert a is not None;obs[di,idx]=a
   values=[r.checkpoint(out/f'{prefix}_{lo:03d}_{lo+10:03d}.npy',sig,(10,len(rows))) for lo in range(0,100,10)];assert all(v is not None for v in values);null[di][:,idx]=np.concatenate(values)
 signs=np.array([int(t['parent_cluster'].rsplit(':',1)[1]) for t in targets]);assert np.isin(signs,[-1,1]).all()
 # Fixed observed-cluster direction, never selected again per dyad or surrogate.
 group=r.pvalue(obs.mean(0)*signs,null.mean(0)*signs)
 per=np.array([r.pvalue(obs[i]*signs,null[i]*signs) for i in range(len(obs))])
 result=[dict(target_id=t['id'],direction=int(signs[i]),group_p=float(group[i]),dyad_p=per[:,i].tolist(),n_pass=int(np.sum(per[:,i]<=.05)),mean_observed=float(obs[:,i].mean())) for i,t in enumerate(targets)]
 path=out/'directional_summary.json'
 if path.exists():assert json.loads(path.read_text())==result
 else:r.save(path,result)
 return result

def main():
 spec=json.loads((ROOT/'specification.json').read_text());r.verify(spec);decision=json.loads((ROOT/'direction_decision.json').read_text());assert decision['approved'] and decision['postprocessor_sha256']==r.sha(__file__)
 targets=json.loads((ROOT/'targets.json').read_text());manifest=json.loads(Path(spec['manifest']).read_text());trialsig=r.sha(ROOT/'specification.json')
 r.log('Direction-aware continuation waiting for verified trial-shuffle generation; no duplicate calculations launched.')
 while not (ROOT/'trial_generation_complete.json').exists():
  st=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
  if st.get('state')=='failed':raise RuntimeError('Trial generation failed: '+str(st))
  time.sleep(30)
 # Trial completion marker is written just before its process releases its lock.
 for _ in range(10):
  if not (ROOT/'run.lock').exists():break
  time.sleep(1)
 lock=ROOT/'run.lock';lock.open('x').write(str(os.getpid()))
 try:
  tr=direction_summary(spec,'trial',targets,trialsig);allowed={t['target_id'] for t in tr if t['group_p']<=.10};gated=[t for t in targets if t['id'] in allowed]
  gate=dict(rule='Direction-aligned cohort-mean trial-shuffle p<=.10',ids=sorted(allowed),direction_decision_sha256=r.sha(ROOT/'direction_decision.json'))
  gp=ROOT/'directional_time_gate.json'
  if gp.exists():assert json.loads(gp.read_text())==gate
  else:r.save(gp,gate)
  timesig=hashlib.sha256(r.canonical(dict(generation_signature=trialsig,gate=gate)).encode()).hexdigest()
  r.log(f'Direction-aware trial results verified. {len(gated)} of {len(targets)} geometry-targets enter time-shift controls.')
  if gated:r.stage(spec,'time',gated,manifest,timesig);ts=direction_summary(spec,'time',gated,timesig)
  else:ts=[]
  tmap={t['target_id']:t for t in tr};smap={t['target_id']:t for t in ts};tiers=[]
  for t in targets:
   trial=tmap[t['id']];shift=smap.get(t['id']);nt=trial['n_pass'];ns=shift['n_pass'] if shift else 0
   tiers.append(dict(t,trial=trial,time_shift=shift,time_shift_tested=shift is not None,tier='A' if nt>=4 and ns>=3 else 'B' if nt>=4 else 'C'))
  r.save(ROOT/'directional_tiers.json',tiers);pairs={}
  for t in tiers:
   if t['pair_id']:pairs.setdefault(t['pair_id'],{})[t['geometry']]=dict(target_id=t['id'],tier=t['tier'],trial_support=t['trial']['n_pass'],time_support=t['time_shift']['n_pass'] if t['time_shift'] else None,time_shift_tested=t['time_shift_tested'])
  assert all(set(p)=={'homologous','cross_hemisphere'} for p in pairs.values())
  r.save(ROOT/'overlap_directional_tiers.json',pairs)
  complete=dict(state='complete',scope='Normative controls only; genuine interaction excluded; separate tiers per geometry',targets=len(targets),overlap_pairs=len(pairs),time_tested=len(gated),tier_counts={k:sum(t['tier']==k for t in tiers) for k in 'ABC'},trial_signature=trialsig,time_signature=timesig,verification='All observation and surrogate checkpoint hashes/shapes/signatures checked during summaries; p values and recurrence counts calculated from saved arrays')
  r.save(ROOT/'completion_verification.json',complete);r.status(complete);r.log('Normative directional controls and separate geometry tiers complete; all checkpoint records verified.')
 except BaseException as error:r.status(dict(state='failed',error=repr(error)));r.log('Continuation stopped: '+repr(error));raise
 finally:lock.unlink()
if __name__=='__main__':main()
