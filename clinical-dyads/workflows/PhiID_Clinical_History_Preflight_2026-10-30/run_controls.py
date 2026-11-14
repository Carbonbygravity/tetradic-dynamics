import sys,os,time,json,importlib.util,subprocess
from pathlib import Path
sp=importlib.util.spec_from_file_location('clinical_history_controls_runner',Path(__file__).with_name('runner.py'))
h=importlib.util.module_from_spec(sp);sys.modules[sp.name]=h;sp.loader.exec_module(h)
import numpy as np
R=h.R
read=h.read;save=h.save;sha=h.sha;checkpoint=h.checkpoint;path_for=h.path_for;GRID=h.GRID

def main():
 assert sys.argv[1:]==['--approved-full-run'],'Expected --approved-full-run'
 auth=read(R/'controls_execution_authorization.json');assert auth['full_run_approved'] is True
 for p,digest in auth['hashes'].items():assert sha(p)==digest
 lock=R/'controls.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  for kind in ['trial','time']:
   h.log('Clinical history controls stage starting/resuming: '+kind)
   for script,args in [('execute_controls_stage.py',[kind,'--approved-full-run']),('verify_controls_stage.py',[kind])]:
    with (R/f'{kind}_stdout.log').open('a') as out,(R/f'{kind}_stderr.log').open('a') as err:
     subprocess.run([sys.executable,str(R/script),*args],stdout=out,stderr=err,check=True)
   h.log('Clinical history controls independently verified: '+kind)
 finally:lock.unlink()
if __name__=='__main__':main()
