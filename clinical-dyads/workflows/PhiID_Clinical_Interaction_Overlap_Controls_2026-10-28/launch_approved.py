"""Run the separate response-period × condition interaction controls."""
import os,sys,json,time,runpy
from pathlib import Path
import runner as r
P=Path(__file__).resolve().parent
if __name__=='__main__':
 spec=json.loads((P/'specification.json').read_text());r.verify(spec)
 for n in ['preflight.json','fresh_process_resume_verification.json']:
  assert json.loads((P/n).read_text())['status']=='passed'
 assert json.loads((P/'execution_authorization.json').read_text())['approved']
 assert not (P/'trial').exists() and not (P/'time').exists(), 'Existing results require resume review'
 assert not (P/'run.lock').exists()
 lock=P/'execution.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 protected=json.loads((P/'preparation_verification.json').read_text())['original_source_hashes']
 assert all(r.sha(p)==h for p,h in protected.items())
 try:
  r.save(P/'launch_record.json',dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),specification_sha256=r.sha(P/'specification.json'),launcher_sha256=r.sha(__file__),workers=1,targets=169,dyads=15,surrogates=100,protected_sources=protected))
  r.log('Starting interaction controls: 169 targets, 15 dyads, 100 surrogates; one worker. Direction fixed by parent cluster; gated time shifts and separate geometry tiers.')
  runpy.run_path(str(P/'trial_generation.py'),run_name='__main__')
  runpy.run_path(str(P/'finish_controls.py'),run_name='__main__')
  assert all(r.sha(p)==h for p,h in protected.items())
  r.save(P/'execution_complete.json',dict(status='complete',finished=time.strftime('%Y-%m-%d %H:%M:%S'),protected_sources_unchanged=True,completion_sha256=r.sha(P/'completion_verification.json')))
 except BaseException as e:
  r.status(dict(state='failed',error=repr(e)));r.log('Interaction execution stopped: '+repr(e));raise
 finally:lock.unlink()
