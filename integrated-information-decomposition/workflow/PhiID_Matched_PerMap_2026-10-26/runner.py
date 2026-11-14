"""Approved matched normative per-map tests. Historical inputs/outputs are read-only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,shutil,hashlib
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
OLD=WORK/'PhiID_Normative_Inference_2026-10-26'
sys.path.insert(0,str(OLD))
import pipeline as engine
core=engine.core
NAMES=('Leader_minus_Follower_Mirroring','Leader_minus_Follower_Matching')
REUSED=('Matching_minus_Mirroring','Leader_period_minus_Follower_period')
SEPARATE='Role_by_Condition'

def log(message):
    line=time.strftime('%Y-%m-%d %H:%M:%S')+' — '+message
    print(line,flush=True)
    with (ROOT/'CHANGE_LOG.txt').open('a',encoding='utf-8') as f:f.write(line+'\n')

def save(path,value):engine.save(path,value)
def export(name,result,provenance,separate=False):
    rows=[]
    for c in result['clusters']:
        row={k:v for k,v in c.items() if k not in ('p_primary_family','p_historical_per_map')}
        row['p_within_map']=c['p_historical_per_map'];rows.append(row)
    out=ROOT/('separate_interaction' if separate else 'matched_results');out.mkdir(exist_ok=True)
    save(out/(name+'.json'),dict(contrast=name,cohort='normative',correction='Temporal cluster maximum within each ROI/geometry/band/atom map',source=provenance,clusters=rows,significant_count=sum(c['p_within_map']<=.05 for c in rows)))

def prepare():
    config=json.loads((OLD/'specification.json').read_text());engine.verify(config)
    oldsig=engine.digest(OLD/'specification.json');maps=json.loads((OLD/'map_inventory.json').read_text())
    checks=[]
    for name in REUSED+(SEPARATE,):
        result=json.loads((OLD/name/'results.json').read_text());assert result['signature']==oldsig
        item=config['array_files'][name];assert engine.digest(OLD/item['path'])==item['sha256']
        x=np.load(OLD/item['path'],mmap_mode='r');null=[]
        for lo in range(0,5000,25):
            v=engine.checkpoint(OLD/name/f'null_{lo:04d}_{lo+25:04d}.npy',oldsig,(25,3264));assert v is not None;null.append(v)
        null=np.concatenate(null);original={(c['map'],c['start'],c['stop'],c['sign']):c for c in result['clusters']};nchecked=0
        for lo in range(0,3264,96):
            z=np.asarray(x[:,lo:lo+96]);t=stats.ttest_1samp(z,0,axis=0).statistic
            for c in core.cluster_masses(t,stats.t.ppf(.975,14)):
                c['map']+=lo;s=original[(c['map'],c['start'],c['stop'],c['sign'])]
                np.testing.assert_allclose(c['mass'],s['mass'],rtol=1e-10,atol=1e-9)
                cutoff=s['mass']-1e-12*max(1.,abs(s['mass']))
                assert s['p_historical_per_map']==float(np.mean(null[:,c['map']]>=cutoff));assert s['coordinates']==maps[c['map']];nchecked+=1
        assert nchecked==len(original)
        export(name,result,dict(path=str(OLD/name/'results.json'),sha256=engine.digest(OLD/name/'results.json'),verified_null_batches=200,observed_clusters_checked=nchecked),name==SEPARATE)
        checks.append(dict(contrast=name,clusters=nchecked,verified_batches=200));log('Verified and exported existing per-map results: '+name)
    del null
    # Only the two missing within-condition contrasts require new permutations.
    role=np.load(OLD/config['array_files'][REUSED[1]]['path'],mmap_mode='r')
    interaction=np.load(OLD/config['array_files'][SEPARATE]['path'],mmap_mode='r')
    arrays={name:np.lib.format.open_memmap(ROOT/(name+'.partial.npy'),mode='w+',dtype='float64',shape=role.shape) for name in NAMES}
    source=json.loads((OLD/'source_inventory.json').read_text());assert len(source)==1020
    import h5py
    for item in source:
        path=Path(item['path']);assert engine.digest(path)==item['sha256']
        di=config['dyads'].index(item['dyad']);lo=item['roi']*48;hi=lo+48
        with h5py.File(path,'r') as f:v=f['values'][:]
        assert v.shape==(2,4,6,1999,4) and np.isfinite(v).all()
        for name,sgn,cells in [(NAMES[0],-1,(0,3)),(NAMES[1],1,(1,2))]:
            derived=role[di,lo:hi]+sgn*interaction[di,lo:hi]/2
            direct=(v[:,cells[0]]-v[:,cells[1]]).transpose(0,1,3,2).reshape(48,1999)
            np.testing.assert_allclose(derived,direct,rtol=1e-9,atol=1e-12)
            arrays[name][di,lo:hi]=direct
    files={}
    for name,x in arrays.items():
        x.flush();x._mmap.close();partial=ROOT/(name+'.partial.npy');final=ROOT/(name+'.npy');partial.rename(final)
        files[name]=dict(path=final.name,sha256=engine.digest(final),shape=list(role.shape))
    for filename in ['map_inventory.json','source_inventory.json','sign_assignments.npy']:
        assert not (ROOT/filename).exists();shutil.copyfile(OLD/filename,ROOT/filename)
    config.update(contrasts={n:core.CONTRASTS[n].tolist() for n in NAMES},family='Temporal cluster correction separately within each map',comparisons='Four approved matched contrasts; pooled condition and pooled Role reused after verification. Genuine interaction exported separately.',array_files=files,code_hashes={str(Path(__file__)):engine.digest(__file__),str(Path(engine.__file__)):engine.digest(engine.__file__),str(Path(core.__file__)):engine.digest(core.__file__)},inference_scope='Normative within-condition Role tests only. Clinical estimates continue independently. No overlap, surrogate, or cohort-comparison inference.')
    save(ROOT/'specification.json',config);save(ROOT/'assembly_validation.json',dict(status='passed',source_tiles_verified=1020,arithmetic_checked='Both within-condition contrasts checked against all source tile cells',reused=checks))
    log('Prepared both missing contrasts; all 1,020 source hashes and direct cell calculations checked.')
    return config

def run(config):
    engine.ROOT=ROOT;engine.verify(config);signs=np.load(ROOT/'sign_assignments.npy');maps=json.loads((ROOT/'map_inventory.json').read_text());sig=engine.digest(ROOT/'specification.json')
    lock=ROOT/'inference.lock';lock.open('x').write(str(os.getpid()));started=time.time()
    try:
        for name in NAMES:
            x=np.load(ROOT/config['array_files'][name]['path'],mmap_mode='r');assert engine.digest(ROOT/config['array_files'][name]['path'])==config['array_files'][name]['sha256']
            threshold=stats.t.ppf(.975,len(x)-1);out=ROOT/name;out.mkdir(exist_ok=True);null=[]
            for lo in range(0,5000,25):
                path=out/f'null_{lo:04d}_{lo+25:04d}.npy';v=engine.checkpoint(path,sig,(25,3264))
                if v is None:v=engine.null_batch(x,signs[lo:lo+25],threshold);engine.publish(path,v,sig)
                null.append(v);engine.status(dict(state='running',cohort='normative',contrast=name,completed_assignments=lo+25,total_assignments=5000,elapsed_seconds=time.time()-started));print(name,lo+25,flush=True)
            null=np.concatenate(null);clusters=[]
            for lo in range(0,3264,96):
                z=np.asarray(x[:,lo:lo+96]);t=core.t_from_sums(z.sum(0),(z*z).sum(0),len(x))
                np.testing.assert_allclose(t,stats.ttest_1samp(z,0,axis=0).statistic,rtol=1e-9,atol=1e-10)
                for c in core.cluster_masses(t,threshold):
                    c['map']+=lo;cutoff=c['mass']-1e-12*max(1.,abs(c['mass']))
                    c.update(p_historical_per_map=float(np.mean(null[:,c['map']]>=cutoff)),start_ms=config['times_ms'][c['start']],end_ms=config['times_ms'][c['stop']-1],coordinates=maps[c['map']]);clusters.append(c)
            result=dict(contrast=name,signature=sig,clusters=clusters)
            save(out/'results.json',result);export(name,result,dict(path=str(out/'results.json'),sha256=engine.digest(out/'results.json')))
            log('Completed '+name+'; within-map significant clusters='+str(sum(c['p_historical_per_map']<=.05 for c in clusters)))
        engine.status(dict(state='complete',cohort='normative',matched_contrasts=4,separate_interaction=True,elapsed_seconds=time.time()-started));log('Normative matched per-map stage complete. Clinical stage pending complete estimate verification.')
    except BaseException as e:engine.status(dict(state='failed',error=repr(e)));log('Failure: '+repr(e));raise
    finally:lock.unlink()

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare':prepare()
    else:
        config=json.loads((ROOT/'specification.json').read_text());engine.ROOT=ROOT
        if stage=='preflight':
            # Reuse the frozen numerical/checkpoint test, selecting the first new contrast.
            core.PRIMARY=NAMES;engine.preflight(config);log('Preflight passed; bounded permutation agreement and checkpoint checks passed.')
        elif stage=='run':
            assert json.loads((ROOT/'preflight_validation.json').read_text())['status']=='passed';run(config)
        else:raise ValueError(stage)
