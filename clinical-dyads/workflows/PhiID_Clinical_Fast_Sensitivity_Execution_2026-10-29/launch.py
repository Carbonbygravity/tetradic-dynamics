import json,hashlib,subprocess,sys,os,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;W=O.parent
folders=['PhiID_Clinical_RoleSwap_Preflight_2026-10-29','PhiID_Clinical_Diagnostics_Preflight_2026-10-29']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def log(s):
 msg=time.strftime('%Y-%m-%d %H:%M:%S')+' '+s;print(msg,flush=True)
 with (O/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(msg+'\n')
def verify_inputs(p):
 s=read(p/'specification.json')
 assert all(sha(q)==h for q,h in s['hashes'].items())
 assert read(p/'preflight.json')['status']==read(p/'fresh_process_resume.json')['status']=='passed'
 assert read(p/'execution_authorization.json')['approved']
 assert not (p/'run.lock').exists()
 return s
def verify_completed(p,s,role):
 complete=read(p/'completion_verification.json');assert complete['state']=='complete' and complete['targets']==len(s['targets'])
 sig=sha(p/'specification.json') if role else hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
 markers=list((p/'checkpoints').glob('*.complete.json'))
 expected=15*len(s['targets'])*101 if role else 15*4*sum(sum(v!=0 for v in read(W/'PhiID_Clinical_Sensitivity_Frozen_2026-10-29/role_swap_specification.json')['coefficients'][t['contrast']]) for t in s['targets'])
 assert len(markers)==expected,(len(markers),expected)
 for m in markers:
  v=read(m);q=m.with_name(m.name.replace('.complete.json','.npy'));assert sha(q)==v['sha256'] and v['signature']==sig
  a=np.load(q,allow_pickle=False);assert list(a.shape)==v['shape'] and not np.isinf(a).any()
  if role:assert a.shape==(2,4) and np.isfinite(a).all()
 files=['four_contrast_results.json','genuine_interaction_results.json'] if role else ['four_contrast_diagnostics.json','genuine_interaction_diagnostics.json']
 rows=[read(p/f) for f in files];assert [len(x) for x in rows]==([10,1] if role else [148,12])
 if role:
  results=rows[0]+rows[1];order=np.argsort([x['p'] for x in results]);adj=0
  signs=1-2*((np.arange(32768)[:,None]>>np.arange(15))&1)
  for rank,j in enumerate(order):
   x=results[j];d=np.array(x['dyad_changes']);pv=float(np.mean(np.abs(signs@d/15)>=abs(d.mean())-1e-14));assert pv==x['p']
   adj=max(adj,min(1,pv*(11-rank)));assert adj==x['holm_p']
 assert all(sha(q)==h for q,h in s['hashes'].items())
 save(p/'postrun_checkpoint_verification.json',dict(status='passed',checkpoints=expected,output_counts=[len(x) for x in rows],scope='All checkpoint hashes, signatures and shapes checked; finite role values, no infinite diagnostic values; separate output counts; role p/Holm values recomputed from saved dyad changes. No second full scientific run.',source_hashes_unchanged=True))
if __name__=='__main__':
 lock=O/'queue.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  save(O/'launch_record.json',dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),stages=folders,workers=1,launcher_sha256=sha(__file__)))
  for i,name in enumerate(folders):
   p=W/name;s=verify_inputs(p)
   assert not (p/'completion_verification.json').exists(),'Already complete; do not duplicate'
   log('Starting '+name)
   with (p/'production_stdout.log').open('x',encoding='utf-8') as out,(p/'production_stderr.log').open('x',encoding='utf-8') as err:
    child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(p/'run.py'),'run'],stdout=out,stderr=err,cwd=str(W))
    save(p/'production_launch.json',dict(pid=child.pid,started=time.strftime('%Y-%m-%d %H:%M:%S'),specification_sha256=sha(p/'specification.json')))
    code=child.wait()
   if code:raise RuntimeError(f'{name} exit {code}; inspect production_stderr.log')
   verify_completed(p,s,i==0);log('Completed and verified '+name)
  save(O/'completion.json',dict(status='passed',finished=time.strftime('%Y-%m-%d %H:%M:%S')))
 except BaseException as e:
  save(O/'failure.json',dict(error=repr(e),time=time.strftime('%Y-%m-%d %H:%M:%S')));log('Stopped: '+repr(e));raise
 finally:lock.unlink()
