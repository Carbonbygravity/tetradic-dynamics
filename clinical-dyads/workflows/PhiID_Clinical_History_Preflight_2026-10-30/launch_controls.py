"""Launch the frozen clinical history controls."""
import json,subprocess,sys,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
def record(name,value):
 with (R/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
if __name__=='__main__':
 with (R/'controls_stdout.log').open('a') as out,(R/'controls_stderr.log').open('a') as err:
  process=subprocess.Popen([sys.executable,str(R/'run_controls.py'),'--approved-full-run'],stdout=out,stderr=err)
  record('controls_process.json',dict(pid=process.pid,started=datetime.datetime.now().astimezone().isoformat(),workers=4))
  print('Clinical history orchestrator PID '+str(process.pid),flush=True)
  code=process.wait()
 record('controls_exit.json',dict(returncode=code,ended=datetime.datetime.now().astimezone().isoformat()))
 sys.exit(code)
