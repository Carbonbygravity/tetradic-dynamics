"""Run the observed clinical history grid."""
import sys, os, time, json, importlib.util, datetime
from pathlib import Path
sp=importlib.util.spec_from_file_location('clinical_history_observed_runner',Path(__file__).with_name('runner.py'))
h=importlib.util.module_from_spec(sp);sys.modules[sp.name]=h;sp.loader.exec_module(h)
import numpy as np

def main():
    p=h.R;spec=h.read(p/'specification.json');auth=h.read(p/'observed_authorization.json')
    assert auth['stage']=='observed' and auth['workers']==4
    assert auth['specification_sha256']==h.sha(p/'specification.json')
    assert h.read(p/'benchmark_verification.json')['status']=='passed'
    for f,digest in spec['hashes'].items():assert h.sha(f)==digest
    for f,digest in auth['protected_hashes'].items():assert h.sha(f)==digest
    lock=p/'observed.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    start=time.perf_counter()
    try:
        folder='observed_checkpoints';(p/folder).mkdir(exist_ok=True)
        h.log('Starting the observed clinical history grid with four workers')
        h.save(p/'observed_launch.json',dict(pid=os.getpid(),started_at=datetime.datetime.now().astimezone().isoformat(),targets=54,dyads=15,expected_checkpoints=810,workers=4))
        result=h.pool([(d,list(range(len(spec['targets']))),['observed'],[0],folder,float('inf')) for d in spec['dyads']])
        h.save(p/'observed_execution.json',result)
        sig=h.sha(p/'specification.json');rows=[]
        for ti,t in enumerate(spec['targets']):
            arrays=[h.checkpoint(h.path_for(folder,d,ti,'observed',0),sig) for d in spec['dyads']]
            assert all(a is not None for a in arrays)
            values=np.stack(arrays);scores=values[:,:,0].mean(0);best=int(np.argmin(scores))
            rows.append(dict(target_index=ti,target=t,selections=[dict(replicate=0,selected_grid_index=best,H=h.GRID[best][0],tau=h.GRID[best][1],mean_scores=scores.tolist(),dyad_contrasts=values[:,best,1].tolist())]))
        for name,flag in [('observed_four_contrast.json',False),('observed_genuine_interaction.json',True)]:
            h.save(p/name,[r for r in rows if (r['target']['contrast']=='Role_by_Condition')==flag])
        # Independently reread persisted checkpoint arrays and summary selections.
        paths=list((p/folder).glob('*.npy'));assert len(paths)==810
        for path in paths:
            marker=h.read(path.with_suffix('.complete.json'));a=np.load(path,allow_pickle=False)
            assert marker['sha256']==h.sha(path) and marker['signature']==sig and marker['shape']==[21,2]
            assert a.shape==(21,2) and np.isfinite(a).all()
        counts={}
        for name in ['observed_four_contrast.json','observed_genuine_interaction.json']:
            saved=h.read(p/name);counts[name]=len(saved)
            for row in saved:
                v=np.stack([np.load(h.path_for(folder,d,row['target_index'],'observed',0),allow_pickle=False) for d in spec['dyads']])
                mean=np.mean(v[:,:,0],axis=0);best=min(range(21),key=lambda j:(mean[j],*h.GRID[j]));s=row['selections'][0]
                assert best==s['selected_grid_index'] and list(h.GRID[best])==[s['H'],s['tau']]
                np.testing.assert_array_equal(mean,s['mean_scores']);np.testing.assert_array_equal(v[:,best,1],s['dyad_contrasts'])
        assert list(counts.values())==[48,6]
        for f,digest in auth['protected_hashes'].items():assert h.sha(f)==digest
        compute=sum(r['compute_seconds'] for w in result['workers'] for r in w['records'])
        h.save(p/'observed_independent_verification.json',dict(status='passed',checkpoints=810,grid_configurations=17010,output_counts=counts,checks='All hashes/signatures/shapes/finite values; selections independently recomputed from saved grids; protected hashes unchanged',scope='Observed grid and selection only; no history-specific p values or tiers',elapsed_seconds=time.perf_counter()-start,total_compute_seconds=compute,ideal_trial_compute_hours=compute*100/4/3600,ideal_all_time_compute_hours=compute*100/4/3600,output_hashes={name:h.sha(p/name) for name in counts}))
        h.log('Observed history grid complete and verified: 48 main and six interaction selections; shuffle controls not launched')
    finally:
        lock.unlink()

if __name__=='__main__':main()
