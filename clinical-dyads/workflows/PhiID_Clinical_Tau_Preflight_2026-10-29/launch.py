import subprocess, sys, json, datetime
from pathlib import Path
p=Path(__file__).resolve().parent
with (p/'stdout.log').open('x',encoding='utf-8') as out, (p/'stderr.log').open('x',encoding='utf-8') as err:
    child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(p/'run.py'),'run'],stdout=out,stderr=err,cwd=p.parent)
    with (p/'launch_record.json').open('x') as f:
        json.dump({'pid':child.pid,'started_at':datetime.datetime.now().astimezone().isoformat(),'workers':1},f,indent=2)
    print('Clinical tau worker PID',child.pid,flush=True)
    result=child.wait()
    with (p/'execution_exit.json').open('x') as f:
        json.dump({'exit_code':result,'ended_at':datetime.datetime.now().astimezone().isoformat()},f,indent=2)
    sys.exit(result)
