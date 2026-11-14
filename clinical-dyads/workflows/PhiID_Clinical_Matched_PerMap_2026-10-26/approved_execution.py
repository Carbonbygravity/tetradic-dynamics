"""Verify completed clinical estimates and run the four matched contrasts."""
import json, os, time, traceback
from pathlib import Path
import runner as r

ROOT = Path(__file__).resolve().parent
def record(state, **kwargs):
    r.save(ROOT/'execution_status.json', dict(state=state, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs))

if __name__ == '__main__':
    lock = ROOT/'approved_execution.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        plan=json.loads((ROOT/'plan.json').read_text())
        r.verify_plan(plan)
        assert json.loads((ROOT/'preparation_tests.json').read_text())['status']=='passed'
        assert json.loads((r.SOURCE/'parallel_status.json').read_text())['state']=='complete'
        assert not (r.SOURCE/'scheduler.lock').exists()
        assert not (ROOT/'assembly.json').exists(), 'Existing assembly requires explicit resume review'
        assert not list(ROOT.glob('*.partial.npy')), 'Preserve partial assembly for review'
        r.completion_gate(plan)
        r.save(ROOT/'execution_authorization.json', dict(date=time.strftime('%Y-%m-%d'), authorization='Clinical tile verification and four matched per-map tests.', plan_sha256=r.engine.digest(ROOT/'plan.json'), runner_sha256=r.engine.digest(ROOT/'runner.py'), execution_script_sha256=r.engine.digest(Path(__file__))))
        record('verifying_and_assembling')
        r.log('Verifying completed clinical tiles and assembling the four matched contrasts.')
        r.prepare(plan)
        record('running_matched_tests', assembly_sha256=r.engine.digest(ROOT/'assembly.json'))
        r.run(plan)
        results={n:r.engine.digest(ROOT/n/'results.json') for n in r.NAMES}
        record('complete', results_sha256=results, genuine_interaction='excluded')
    except BaseException as error:
        record('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink()
