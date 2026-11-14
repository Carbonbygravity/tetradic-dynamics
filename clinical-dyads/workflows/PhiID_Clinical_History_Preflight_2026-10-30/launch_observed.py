import subprocess,sys,json,datetime
from pathlib import Path
p=Path(__file__).resolve().parent
with (p/'observed_stdout.log').open('x',encoding='utf-8') as out,(p/'observed_stderr.log').open('x',encoding='utf-8') as err:
    child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(p/'execute_observed.py')],stdout=out,stderr=err,cwd=p.parent)
    with (p/'observed_process.json').open('x') as f:json.dump({'pid':child.pid,'started_at':datetime.datetime.now().astimezone().isoformat()},f,indent=2)
    print('Observed history orchestrator PID',child.pid,flush=True)
    rc=child.wait()
    with (p/'observed_exit.json').open('x') as f:json.dump({'exit_code':rc,'ended_at':datetime.datetime.now().astimezone().isoformat()},f,indent=2)
    sys.exit(rc)
