"""Generate the approved trial-shuffle distributions only; no tail decision or tiers."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import runner as r
ROOT=Path(__file__).resolve().parent
spec=json.loads((ROOT/'specification.json').read_text());r.verify(spec)
assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
targets=json.loads((ROOT/'targets.json').read_text());manifest=json.loads(Path(spec['manifest']).read_text());sig=r.sha(ROOT/'specification.json')
lock=ROOT/'run.lock';lock.open('x').write(str(os.getpid()))
try:
 r.log('Started trial-shuffle distribution generation only. Tail selection, time-shift gate and tiers remain pending the explicit direction decision. Surrogate values do not depend on tail choice.')
 r.stage(spec,'trial',targets,manifest,sig)
 # Recheck every completed distribution and observation, without calculating p values.
 batches=0
 for d in spec['dyads']:
  for roi in sorted({t['roi_index'] for t in targets}):
   rows=[t for t in targets if t['roi_index']==roi];prefix=f'dyad{d}_roi{roi:02d}'
   assert r.checkpoint(ROOT/'trial'/(prefix+'_observed.npy'),sig,(len(rows),)) is not None
   for lo in range(0,100,10):
    assert r.checkpoint(ROOT/'trial'/f'{prefix}_{lo:03d}_{lo+10:03d}.npy',sig,(10,len(rows))) is not None;batches+=1
 r.save(ROOT/'trial_generation_complete.json',dict(complete=True,verified_batches=batches,targets=len(targets),dyads=len(spec['dyads']),surrogates=100,p_values_computed=False))
 r.status(dict(state='trial_distributions_complete',verified_batches=batches,next='Apply approved direction rule, then time-shift gate and separate geometry tiers'))
 r.log('Trial distributions complete and verified. No p values or tiers produced by this generation-only command.')
except BaseException as error:r.status(dict(state='failed',error=repr(error)));r.log('Stopped: '+repr(error));raise
finally:lock.unlink()
