"""Prepare, verify, or queue the separately approved genuine-interaction controls."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time,sys,hashlib,ast
from pathlib import Path
import numpy as np
import runner as r
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'PhiID_Normative_Controls_2026-10-26'
FROZEN_TARGETS=ROOT.parent.parent/'frozen_records/interaction_targets.json'
FROZEN_MAPS=ROOT.parent.parent/'frozen_records/matched_map_inventory.json'
def read(p):return json.loads(Path(p).read_text())
def once(p,v):
 if p.exists():assert read(p)==v
 else:r.save(p,v)
def prepare():
 maps=read(FROZEN_MAPS)
 rois=[maps[i*48]['roi'] for i in range(68)]
 targets=read(FROZEN_TARGETS)
 for t in targets:
  assert rois[t['roi_index']]==t['roi']
  assert r.BANDS[t['band_index']]==tuple(t['band'])
  assert r.ATOMS[t['atom_index']]==t['atom']
  assert t['direction']==int(t['parent_cluster'].rsplit(':',1)[1])
 assert len(targets)==152 and r.COEFF=={'Role_by_Condition':[-1.,1.,-1.,1.]}
 r.save(ROOT/'targets.json',targets)
 spec=read(PREVIOUS/'specification.json')
 spec.update(contrasts=r.COEFF,targets_sha256=r.sha(ROOT/'targets.json'),sources={str(FROZEN_TARGETS):r.sha(FROZEN_TARGETS),str(FROZEN_MAPS):r.sha(FROZEN_MAPS)},p_value='Fixed parent-cluster direction: (1+count(sign*null>=sign*observed))/101',genuine_interaction='Only contrast; kept separate from four matched contrasts',predecessor=str(PREVIOUS),code_hashes={str(p):r.sha(p) for p in [ROOT/'runner.py',ROOT/'control_core.py',Path(__file__),Path(r.c.prod.__file__),Path(r.c.common.__file__),*Path(r.c.prod.calc.__file__).parent.glob('*.py')]})
 r.save(ROOT/'specification.json',spec)
 r.log('Prepared 152 separate genuine-interaction targets; no production controls launched.')
def summarize(root,spec,kind,targets,sig,write):
 ids=[t['id'] for t in targets];pos={k:i for i,k in enumerate(ids)}
 obs=np.empty((len(spec['dyads']),len(ids)));null=np.empty((len(spec['dyads']),100,len(ids)))
 for di,d in enumerate(spec['dyads']):
  for roi in sorted({t['roi_index'] for t in targets}):
   rows=[t for t in targets if t['roi_index']==roi];idx=[pos[t['id']] for t in rows];prefix=f'dyad{d}_roi{roi:02d}'
   a=r.checkpoint(root/kind/(prefix+'_observed.npy'),sig,(len(rows),));assert a is not None;obs[di,idx]=a
   values=[r.checkpoint(root/kind/f'{prefix}_{lo:03d}_{lo+10:03d}.npy',sig,(10,len(rows))) for lo in range(0,100,10)]
   assert all(v is not None for v in values);null[di][:,idx]=np.concatenate(values)
 signs=np.array([int(t['parent_cluster'].rsplit(':',1)[1]) for t in targets]);assert np.isin(signs,[-1,1]).all()
 group=r.pvalue(obs.mean(0)*signs,null.mean(0)*signs)
 per=np.array([r.pvalue(obs[i]*signs,null[i]*signs) for i in range(len(obs))])
 result=[dict(target_id=t['id'],direction=int(signs[i]),group_p=float(group[i]),dyad_p=per[:,i].tolist(),n_pass=int(np.sum(per[:,i]<=.05)),mean_observed=float(obs[:,i].mean())) for i,t in enumerate(targets)]
 p=root/kind/'directional_summary.json'
 if write:once(p,result)
 else:assert read(p)==result
 return result
def tier_name(trial,time_support):return 'A' if trial>=4 and time_support>=3 else 'B' if trial>=4 else 'C'
def verify_predecessor():
 done=read(PREVIOUS/'completion_verification.json');assert done['state']=='complete'
 assert not (PREVIOUS/'run.lock').exists()
 spec=read(PREVIOUS/'specification.json');targets=read(PREVIOUS/'targets.json')
 assert r.sha(PREVIOUS/'targets.json')==spec['targets_sha256']
 for p,h in {**spec['sources'],**spec['code_hashes']}.items():assert r.sha(p)==h
 decision=read(PREVIOUS/'direction_decision.json');assert decision['approved'] and r.sha(PREVIOUS/'finish_controls.py')==decision['postprocessor_sha256']
 sig=r.sha(PREVIOUS/'specification.json');assert sig==done['trial_signature']
 tr=summarize(PREVIOUS,spec,'trial',targets,sig,False)
 allowed={x['target_id'] for x in tr if x['group_p']<=.10};gate=read(PREVIOUS/'directional_time_gate.json');assert gate['ids']==sorted(allowed)
 assert gate['direction_decision_sha256']==r.sha(PREVIOUS/'direction_decision.json')
 tsig=hashlib.sha256(r.canonical(dict(generation_signature=sig,gate=gate)).encode()).hexdigest();assert tsig==done['time_signature']
 gated=[t for t in targets if t['id'] in allowed]
 ts=summarize(PREVIOUS,spec,'time',gated,tsig,False) if gated else []
 tm={x['target_id']:x for x in tr};sm={x['target_id']:x for x in ts}
 tiers=read(PREVIOUS/'directional_tiers.json');assert len(tiers)==len(targets)==done['targets']
 for t,original in zip(tiers,targets):
  assert all(t[k]==v for k,v in original.items());i=t['id'];s=sm.get(i)
  assert t['trial']==tm[i] and t['time_shift']==s and t['time_shift_tested']==(s is not None)
  assert t['tier']==tier_name(tm[i]['n_pass'],s['n_pass'] if s else 0)
 return dict(state='passed',predecessor=str(PREVIOUS),targets=len(targets),time_tested=len(gated),completion_sha256=r.sha(PREVIOUS/'completion_verification.json'),verification='Read-only recomputation of directional summaries and tiers from hash/shape/signature-checked predecessor checkpoints.')
def tests(spec):
 # Frozen copied numerical functions must remain identical to the active runner.
 def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
 old=funcs(PREVIOUS/'runner.py');new=funcs(ROOT/'runner.py');assert all(old[k]==v for k,v in new.items())
 assert r.sha(ROOT/'control_core.py')==r.sha(PREVIOUS/'control_core.py')
 a=np.array([2.,7.,3.,1.]);assert np.dot(r.COEFF['Role_by_Condition'],a)==(7-3)-(2-1)==3
 assert np.dot(r.COEFF['Role_by_Condition'],[1.,2.,4.,3.])==0
 null=np.arange(100.)[:,None];obs=np.array([-80.]);negative=r.pvalue(obs*-1,-null*-1)
 assert np.allclose(negative,21/101)
 assert [tier_name(4,3),tier_name(4,2),tier_name(3,5)]==['A','B','C']
 targets=read(ROOT/'targets.json');assert {t['direction'] for t in targets}=={-1,1}
 assert {t['geometry'] for t in targets}=={'homologous','cross_hemisphere'}
 r.preflight(spec)
 # Actual interaction summaries in both geometries and directions, across three dyads.
 manifest=read(spec['manifest']);selected=[]
 for sign in [-1,1]:
  for geom in ['homologous','cross_hemisphere']:
   selected.append(next(t for t in targets if t['direction']==sign and t['geometry']==geom))
 maximum=0.
 for d in [1,8,16]:
  for roi in sorted({t['roi_index'] for t in selected}):
   rows=[t for t in selected if t['roi_index']==roi];p1,p2,ev=r.load(d,roi,manifest)
   obs=r.evaluate(p1,p2,ev,rows,d,0,'observed');saved=r.stored(d,roi,rows)
   np.testing.assert_allclose(obs,saved,rtol=1e-7,atol=1e-7);maximum=max(maximum,float(np.max(np.abs(obs-saved))))
 r.save(ROOT/'interaction_preflight.json',dict(status='passed',copied_functions_identical=True,core_hash_identical=True,contrast_arithmetic=True,negative_tail=True,tier_boundaries=True,actual_interaction_targets=len(selected)*3,both_signs_and_geometries=True,max_interaction_summary_error=maximum,predecessor_not_touched=True))
 r.log('Interaction preflight passed; reference atom/null checks, actual interaction summaries, direction, tiers and checkpoints verified. Production not launched.')
def execute(spec):
 r.verify(spec);targets=read(ROOT/'targets.json');manifest=read(spec['manifest']);sig=r.sha(ROOT/'specification.json')
 r.stage(spec,'trial',targets,manifest,sig);tr=summarize(ROOT,spec,'trial',targets,sig,True)
 allowed={x['target_id'] for x in tr if x['group_p']<=.10};gated=[t for t in targets if t['id'] in allowed]
 gate=dict(rule='Fixed-direction group trial p<=.10',ids=sorted(allowed));once(ROOT/'directional_time_gate.json',gate)
 tsig=hashlib.sha256(r.canonical(dict(generation_signature=sig,gate=gate)).encode()).hexdigest()
 r.log(f'Interaction trial controls verified; {len(gated)}/{len(targets)} targets qualify for time shifts.')
 if gated:r.stage(spec,'time',gated,manifest,tsig)
 ts=summarize(ROOT,spec,'time',gated,tsig,True) if gated else []
 tm={x['target_id']:x for x in tr};sm={x['target_id']:x for x in ts};tiers=[];pairs={}
 for t in targets:
  tr=tm[t['id']];sh=sm.get(t['id']);tier=tier_name(tr['n_pass'],sh['n_pass'] if sh else 0)
  row=dict(t,trial=tr,time_shift=sh,time_shift_tested=sh is not None,tier=tier);tiers.append(row)
  if t['pair_id']:pairs.setdefault(t['pair_id'],{})[t['geometry']]=dict(target_id=t['id'],tier=tier,trial_support=tr['n_pass'],time_support=sh['n_pass'] if sh else None,time_shift_tested=sh is not None)
 assert all(set(p)=={'homologous','cross_hemisphere'} for p in pairs.values())
 once(ROOT/'directional_tiers.json',tiers);once(ROOT/'overlap_directional_tiers.json',pairs)
 done=dict(state='complete',scope='Normative genuine interaction only; separate geometry tiers',targets=len(targets),overlap_pairs=len(pairs),time_tested=len(gated),tier_counts={k:sum(t['tier']==k for t in tiers) for k in 'ABC'},trial_signature=sig,time_signature=tsig,outputs_sha256={n:r.sha(ROOT/n) for n in ['directional_tiers.json','overlap_directional_tiers.json']},verification='All expected observed/null checkpoint hashes, shapes, finite values and signatures checked in summaries. Directional p values and recurrence counts derived from saved arrays.')
 once(ROOT/'completion_verification.json',done);r.status(done);r.log('Separate interaction controls and recurrence complete.')
def queue(spec):
 r.verify(spec)
 assert read(ROOT/'preflight.json')['status']=='passed' and read(ROOT/'interaction_preflight.json')['status']=='passed'
 lock=ROOT/'queue.lock';lock.open('x').write(str(os.getpid()))
 try:
  r.status(dict(state='queued',waiting_for=str(PREVIOUS/'completion_verification.json'),pid=os.getpid()));r.log('Queued after all four-contrast normative controls; no active analysis modified.')
  while True:
   st=read(PREVIOUS/'status.json')
   if st.get('state')=='failed':raise RuntimeError('Predecessor failed; interaction not launched: '+str(st))
   if (PREVIOUS/'completion_verification.json').exists() and not (PREVIOUS/'run.lock').exists():break
   time.sleep(30)
  r.verify(spec);verification=verify_predecessor();once(ROOT/'predecessor_verification.json',verification)
  runlock=ROOT/'run.lock';runlock.open('x').write(str(os.getpid()))
  try:r.log('Predecessor completion independently verified; starting interaction controls with one worker.');execute(spec)
  finally:runlock.unlink()
 except BaseException as exc:r.status(dict(state='failed',error=repr(exc)));r.log('Stopped without altering predecessor: '+repr(exc));raise
 finally:lock.unlink()
def run_ready(spec):
 """Run only after the four-contrast predecessor has completed and exited."""
 r.verify(spec)
 assert read(ROOT/'preflight.json')['status']=='passed' and read(ROOT/'interaction_preflight.json')['status']=='passed'
 if not (PREVIOUS/'completion_verification.json').exists() or (PREVIOUS/'run.lock').exists():
  raise RuntimeError('Four-contrast controls must be complete and stopped before interaction controls run')
 verification=verify_predecessor();once(ROOT/'predecessor_verification.json',verification)
 runlock=ROOT/'run.lock';runlock.open('x').write(str(os.getpid()))
 try:execute(spec)
 except BaseException as exc:r.status(dict(state='failed',error=repr(exc)));raise
 finally:runlock.unlink()
if __name__=='__main__':
 command=sys.argv[1]
 if command=='prepare':prepare()
 elif command=='preflight':tests(read(ROOT/'specification.json'))
 elif command=='queue':queue(read(ROOT/'specification.json'))
 elif command=='run-ready':run_ready(read(ROOT/'specification.json'))
 else:raise ValueError(command)
