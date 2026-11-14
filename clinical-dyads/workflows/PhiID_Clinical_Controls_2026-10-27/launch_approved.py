"""Run the four-contrast surrogate controls with the frozen settings."""
import json,os,runpy,sys,time
from pathlib import Path
import runner as r
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    spec=json.loads((ROOT/'specification.json').read_text());r.verify(spec)
    assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'adaptation_tests.json').read_text())['status']=='passed'
    assert not (ROOT/'run.lock').exists()
    assert not (ROOT/'trial').exists() and not (ROOT/'time').exists(), 'Existing output requires resume review'
    normative={}
    for folder in ['PhiID_Normative_Controls_2026-10-26','PhiID_Normative_Interaction_Controls_2026-10-26']:
        for name in ['runner.py','control_core.py','specification.json','targets.json']:
            p=Path(os.environ['TETRADIC_NORMATIVE_REFERENCE_ROOT']).resolve()/folder/name;normative[str(p)]=r.sha(p)
    lock=ROOT/'execution.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        r.save(ROOT/'launch_record.json',dict(authorized='Four-contrast clinical surrogate controls and geometry-specific recurrence tiers.',started=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),specification_sha256=r.sha(ROOT/'specification.json'),launcher_sha256=r.sha(Path(__file__)),normative_code_and_spec_hashes=normative,workers=1,stages=['trial shuffles','direction-aware gate','time shifts','separate geometry recurrence tiers']))
        r.log('Starting surrogate controls: 783 clinical targets, 100 surrogates, seed 1337; fixed-direction tests and separate geometry tiers. Genuine interaction excluded.')
        sys.argv=[str(ROOT/'prepare.py'),'run']
        runpy.run_path(str(ROOT/'prepare.py'),run_name='__main__')
        assert all(r.sha(p)==h for p,h in normative.items())
        r.save(ROOT/'execution_complete.json',dict(state='complete',normative_code_and_specs_unchanged=True,completion_sha256=r.sha(ROOT/'completion_verification.json'),finished=time.strftime('%Y-%m-%d %H:%M:%S')))
    except BaseException as e:
        r.log('Execution stopped: '+repr(e));raise
    finally:lock.unlink()
