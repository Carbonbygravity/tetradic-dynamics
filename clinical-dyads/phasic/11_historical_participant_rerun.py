"""Historical participant-level rerun; never averages participants into dyads.

Separate original-input benchmark and corrected-input four-group replication.
Existing project files are read only. All outputs belong to a new run.
"""
import os, sys, json, pickle, hashlib, argparse, traceback, importlib.util
from pathlib import Path
from datetime import datetime, timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import h5py
from scipy import stats, sparse

ROOT=Path(os.environ['TETRADIC_PHASIC_ROOT']).resolve()
NAME='Historical_Participant_Replication_Run_01'
RUN=ROOT/'03_Results'/NAME
QA=ROOT/'05_Quality_Checks'/NAME
FIG=ROOT/'04_Figures'/NAME
REPORT=ROOT/'06_Manuscript_Text'/NAME
OLD=ROOT/'03_Results/Normative_Cell_Order_Corrected_Run_01'
INPUT=ROOT/'03_Results/Harmonized_Phasic_Run_01'
ORDER=['LMA','LMR','FMA','FMR']
WEIGHTS={'Condition':[.5,-.5,.5,-.5],'Role':[.5,.5,-.5,-.5],'Interaction':[-.5,.5,.5,-.5]}
GROUPS=['HistoricalNormHC','NormHC','ClinicalParticipants','PT','HC']
SEED=20261010

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8');os.replace(t,p)
def status(stage,**kw):
    d=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw)
    write(RUN/'run_status.json',d);print(json.dumps(d),flush=True)
def helper():
    p=(Path(__file__).resolve().parent/'09_within_cohort_identity.py')
    s=importlib.util.spec_from_file_location('identity_helpers',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m
def load_old():
    with (OLD/'log_transformed_tensor.pkl').open('rb') as f:d=pickle.load(f)
    assert d['cell_order']==ORDER and d['tensor'].shape==(30,4,68,6,2001)
    return d
def contrast(z,kind,band):
    # Deliberately match historical A/B cell averaging, including SF operation order.
    ix={'Condition':([0,2],[1,3]),'Role':([0,1],[2,3]),'Interaction':([1,2],[0,3])}[kind]
    a=z[:,ix[0]].mean(axis=1);b=z[:,ix[1]].mean(axis=1)
    if band is None:return (a.mean(axis=-1)-b.mean(axis=-1)).reshape(len(z),-1)
    return (a[:,:,band,:]-b[:,:,band,:]).transpose(0,2,1)
def selected(meta,g):
    if g=='NormHC':return meta['normative'],list(range(30))
    m=meta['clinical'];ix=[i for i,x in enumerate(m['groups']) if g=='ClinicalParticipants' or x==g]
    assert len(ix)==(30 if g=='ClinicalParticipants' else 15)
    return m,ix
def fields(meta,g,kind,band):
    if g=='HistoricalNormHC':return contrast(load_old()['tensor'],kind,band)
    m,ix=selected(meta,g);out=[]
    with h5py.File(m['path'],'r') as f:
        for i in ix:
            z=f['tensor'][i:i+1];assert np.isfinite(z).all()
            out.append(contrast(z,kind,band)[0])
    return np.stack(out)
def adjacency(rois):
    with (OLD/'source_cluster_results_theta_Condition.pkl').open('rb') as f:d=pickle.load(f)
    assert list(d['roi_labels'])==rois
    a=sparse.csr_matrix(d['adjacency']);assert a.shape==(68,68) and (a!=a.T).nnz==0
    return a
def test(x,a,n):
    import mne
    fn=mne.stats.spatio_temporal_cluster_1samp_test if x.ndim==3 else mne.stats.permutation_cluster_1samp_test
    return fn(x,adjacency=a,n_permutations=n,tail=0,threshold=None,n_jobs=1,
              seed=SEED,out_type='indices',buffer_size=1000,verbose=False)
def prepare():
    assert not (RUN/'prepared.json').exists(),'Do not overwrite an existing preparation'
    for p in [RUN,QA,FIG,REPORT]:p.mkdir(parents=True,exist_ok=True)
    status('preparing');h=helper();meta,generation=h.inspect();old=load_old()
    assert old['roi_labels']==meta['normative']['rois'] and old['band_labels']==meta['normative']['bands']
    np.testing.assert_allclose(np.asarray(old['time_vector'])/1000,meta['normative']['times'])
    adj=adjacency(meta['normative']['rois']);sparse.save_npz(RUN/'historical_adjacency.npz',adj)
    audit=[]
    for bi,b in enumerate(old['band_labels']):
        for k in WEIGHTS:
            path=OLD/f'source_cluster_results_{b}_{k}.pkl'
            with path.open('rb') as f:d=pickle.load(f)
            x=contrast(old['tensor'],k,bi);t=stats.ttest_1samp(x,0,axis=0).statistic
            np.testing.assert_allclose(t,d['t_obs'],rtol=2e-6,atol=2e-6)
            assert d['n_subjects']==30 and d['cell_order']==ORDER
            assert (sparse.csr_matrix(d['adjacency'])!=adj).nnz==0
            audit.append(dict(path=str(path),sha256=sha(path),max_abs_t_difference=float(np.max(abs(t-d['t_obs']))),passed=True))
    NOTEBOOK=Path(os.environ['TETRADIC_HISTORICAL_NOTEBOOK']).resolve()
    nb=read(NOTEBOOK);excerpts={str(i):''.join(nb['cells'][i]['source']) for i in [198,243]}
    write(RUN/'historical_notebook_reference.json',dict(path=str(NOTEBOOK),sha256=sha(NOTEBOOK),cells=excerpts,
        warning='Notebook cell-order comments are stale. Corrected saved tensor and all 18 saved t maps verify LMA,LMR,FMA,FMR. No notebook code is executed.'))
    mapping={}
    for g in GROUPS:
        if g=='HistoricalNormHC':mapping[g]=dict(subjects=old['subjects'],n=30,input=str(OLD/'log_transformed_tensor.pkl'))
        else:
            m,ix=selected(meta,g);mapping[g]=dict(subjects=[m['subjects'][i] for i in ix],indices=ix,n=len(ix),input=m['path'],identities=[m['groups'][i] for i in ix])
    spec=dict(name=NAME,groups=mapping,inputs=meta,historical_input_sha256=sha(OLD/'log_transformed_tensor.pkl'),
        cell_order=ORDER,contrast_weights=WEIGHTS,unit='Individual participant throughout; no dyad averaging, no reused inferential maps.',
        n_permutations=5000,tail=0,threshold=None,seed=SEED,n_jobs=1,window_seconds=[-1,1],
        adjacency='Historical projected ROI centroids, exact saved graph',adjacency_sha256=sha(RUN/'historical_adjacency.npz'),
        space_frequency='Build participant A/B cell means, average each across full time window, subtract; ROI x band-identity adjacency; no cross-band edges.',
        spatiotemporal='Six bands separately, participant time x ROI contrasts; temporal nearest neighbors plus historical ROI adjacency.',
        primary_reporting='Historical within-map cluster p < .05. Across-map Bonferroni 18 ST / 3 SF provided separately, not substituted for historical criterion.',
        corrected_frequency_bins=generation['frequency_bins'],
        differences=['HistoricalNormHC uses the original corrected saved log tensor and original bins as a benchmark.',
          'Other four groups use already approved harmonized source inputs and corrected bins. These are method-matched updated-input analyses, not identical-input historical reproductions.',
          'The historical notebook did not set a seed. New fixed seed makes reruns reproducible; exact old Monte Carlo p values are not guaranteed.',
          'One compute job and index-format cluster storage replace historical parallel jobs/masks without changing the statistical test.'],
        limitations=['Historical participant sign flips reproduce the historical independence assumption; partners may be dependent. The 30-row analyses are historical-method reproductions, not newly dyad-adjusted inference.',
          'PT and HC subsets each retain one participant per dyad. Combined clinical rows are not a patient-only sample.',
          'Response-event role is not recorded participant task assignment. No imposed initiation/feedback phases.',
          'Cluster extent does not establish pointwise significance or precise onset. Descriptive plots do not establish cross-group effects.',
          'Source scale and cohort processing differences limit absolute-power comparisons.'],
        script_sha256=sha(Path(__file__)),helper_sha256=sha((Path(__file__).resolve().parent/'09_within_cohort_identity.py')),
        engine_sha256=sha((Path(__file__).resolve().parent/'06_anatomical_and_paired_clusters.py')),historical_maps=audit)
    write(RUN/'identity_mapping.json',mapping);write(RUN/'frozen_specification.json',spec)
    sh=sha(RUN/'frozen_specification.json')
    write(QA/'historical_t_map_audit.json',dict(passed=True,specification_sha256=sh,checks=audit))
    write(REPORT/'correction_record.json',dict(reason='Previous 15-dyad averaging did not reproduce the requested historical participant analysis. User requested all within-cohort analyses redone.',
        supersedes_claim='Within_Cohort_Identity_Run_01 is a historical-equivalent participant analysis',
        preserves='All prior files and manuscript tracking README remain unchanged.',specification=spec))
    make_figures(meta,old,h)
    write(RUN/'prepared.json',dict(passed=True,specification_sha256=sh));status('prepared',maps_complete=0,maps_total=105)
def make_figures(meta,old,h):
    hist=read(h.HIST);raw={};reg={}
    def summarize(z,m):
        pp=[]
        for p in hist['panels']:
            ri=[m['rois'].index(r) for r in p['roi_labels']];bi=m['bands'].index(p['band']);ci=[0,1] if p['role']=='Leader' else [2,3]
            pp.append(z[ci][:,ri,bi,:].mean(axis=1))
        return z.mean(axis=1),np.stack(pp)
    for g in GROUPS:
        aa=[];bb=[]
        if g=='HistoricalNormHC':
            for z in old['tensor']:
                a,b=summarize(z,meta['normative']);aa.append(a);bb.append(b)
        else:
            m,ix=selected(meta,g)
            with h5py.File(m['path'],'r') as f:
                for i in ix:
                    a,b=summarize(f['tensor'][i],m);aa.append(a);bb.append(b)
        raw[g]=np.stack(aa);reg[g]=np.stack(bb)
    np.savez_compressed(RUN/'descriptive_curves.npz',times=meta['normative']['times'],**raw,**{k+'_historical_roi':v for k,v in reg.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.asarray(meta['normative']['times']);files=[]
    # Historical plot convention: participant mean +/- SEM, no transferred significance.
    for regional,data in [(False,raw),(True,reg)]:
        for g,a in data.items():
            fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True) if regional else plt.subplots(6,2,figsize=(13,16),sharex=True)
            for j,ax in enumerate(axes.ravel()):
                b=j if regional else j//2;role=hist['panels'][j]['role'] if regional else ['Leader','Follower'][j%2]
                for c,label in enumerate(['Matching','Mirroring']):
                    z=a[:,j,c] if regional else a[:,2*(j%2)+c,b]
                    mean=z.mean(0);sem=stats.sem(z,axis=0)
                    ax.plot(t,mean,label=label,color=['#c43944','#2472ad'][c]);ax.fill_between(t,mean-sem,mean+sem,alpha=.15,color=['#c43944','#2472ad'][c])
                band=hist['panels'][j]['band'] if regional else meta['normative']['bands'][b]
                ax.set_title(f'{role} response event | {band}');ax.set_xlim(-.8,.8);ax.axvline(0,color='gray',ls=':');ax.grid(alpha=.15)
                ax.set_ylabel('Mean log1p power')
            axes.flat[0].legend();fig.suptitle(f'{g} | {len(a)} participant rows | '+('fixed historical ROI panels' if regional else 'all 68 ROIs'))
            for ax in axes[-1]:ax.set_xlabel('Time from response event (s)')
            fig.text(.5,.01,'Participant mean +/- SEM; descriptive. Historical significance not transferred. Event role is not participant task assignment.',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.025,1,.96]);suffix='historical_roi_panels' if regional else 'role_condition_all_rois'
            for ext in ['png','pdf','svg']:
                p=FIG/f'{g}_{suffix}.{ext}';fig.savefig(p,dpi=140);files.append(dict(path=str(p),sha256=sha(p)))
            plt.close(fig)
    write(FIG/'figure_manifest.json',dict(outputs=files,uncertainty='Participant SEM, historical convention'))
def run(stage):
    spec=read(RUN/'frozen_specification.json');sh=sha(RUN/'frozen_specification.json')
    assert spec['script_sha256']==sha(Path(__file__))
    assert spec['helper_sha256']==sha((Path(__file__).resolve().parent/'09_within_cohort_identity.py'))
    assert spec['engine_sha256']==sha((Path(__file__).resolve().parent/'06_anatomical_and_paired_clusters.py'))
    assert spec['historical_input_sha256']==sha(OLD/'log_transformed_tensor.pkl')
    h=helper();meta,_=h.inspect();h.RUN=RUN;m=h.engine();a=sparse.load_npz(RUN/'historical_adjacency.npz')
    assert sha(RUN/'historical_adjacency.npz')==spec['adjacency_sha256']
    sf=sparse.kron(a,sparse.eye(6),format='csr');checks=[]
    if stage=='pilot':
        for g in GROUPS:
            for bi,k in [(None,'Condition'),(0,'Role')]:
                x=fields(meta,g,k,bi);res=test(x,sf if bi is None else a,16);rep=test(x,sf if bi is None else a,16)
                np.testing.assert_allclose(res[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=max(1e-10,20*np.finfo(x.dtype).eps),atol=max(1e-10,20*np.finfo(x.dtype).eps))
                np.testing.assert_array_equal(res[2],rep[2]);np.testing.assert_array_equal(res[3],rep[3])
                assert x.shape[0]==spec['groups'][g]['n']
                checks.append(dict(group=g,band=bi,contrast=k,n=len(x),passed=True))
        write(QA/'pilot.json',dict(passed=True,specification_sha256=sh,checks=checks));status('pilot_passed');return
    assert read(QA/'pilot.json')['passed'] and read(QA/'pilot.json')['specification_sha256']==sh
    rows=[];tasks=[(b,k) for b in range(6) for k in WEIGHTS]+[(None,k) for k in WEIGHTS]
    for g in GROUPS:
        for bi,k in tasks:
            name=('space_frequency_' if bi is None else 'spatiotemporal_'+meta['normative']['bands'][bi]+'_')+k
            dest=RUN/g/(name+'.pkl');marker=dest.with_suffix('.json');dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                r=read(marker);assert r['sha256']==sha(dest) and r['specification_sha256']==sh;rows.append(r);continue
            assert not marker.exists();status('running',group=g,analysis=name,maps_complete=len(rows),maps_total=105)
            x=fields(meta,g,k,bi);res=test(x,sf if bi is None else a,5000)
            np.testing.assert_allclose(res[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=max(1e-10,20*np.finfo(x.dtype).eps),atol=max(1e-10,20*np.finfo(x.dtype).eps))
            assert len(res[3]) in [0,5000]
            summ=m.summaries(res,x.shape[1:],meta['normative']['bands'],meta['normative']['rois'],meta['normative']['times'],3 if bi is None else 18,bi is None)
            data=dict(t_obs=res[0],clusters=res[1],cluster_p_values=res[2],H0=res[3],n_subjects=len(x),cell_order=ORDER,
                      weights=WEIGHTS[k],group=g,analysis=name,specification_sha256=sh)
            with dest.open('xb') as f:pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
            r=dict(group=g,analysis=name,n_participants=len(x),path=str(dest),sha256=sha(dest),specification_sha256=sh,clusters=summ,
                   significant_within_map=sum(c['p_within_map']<.05 for c in summ),significant_across_maps=sum(c['p_across_maps_bonferroni']<.05 for c in summ))
            write(marker,r);rows.append(r);status('map_complete',group=g,analysis=name,maps_complete=len(rows),maps_total=105)
    assert len(rows)==105
    for r in rows:assert r['sha256']==sha(Path(r['path']))
    write(RUN/'cluster_summary.json',dict(results=rows,specification_sha256=sh));report(rows,spec,sh)
    write(QA/'validation.json',dict(passed=True,maps_complete=105,specification_sha256=sh,checks=['18 historical observed t maps reproduced','participant identities and row counts','all input and result hashes','independent t statistics','fixed-seed pilot','historical adjacency']))
    write(RUN/'analysis_complete.json',dict(passed=True,maps_complete=105,specification_sha256=sh,utc=datetime.now(timezone.utc).isoformat()))
    status('complete',maps_complete=105,maps_total=105)
def report(rows,spec,sh):
    import html
    parts=['<!doctype html><meta charset="utf-8"><title>Historical participant replication</title><style>body{font:16px Arial;max-width:1200px;margin:40px auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}</style><h1>Historical participant-level replication</h1>',
        '<p>HistoricalNormHC is the original-input benchmark. NormHC and clinical groups use corrected harmonized inputs. Individual rows throughout; no dyad averaging. Primary significance follows historical within-map correction.</p>']
    for x in spec['differences']+spec['limitations']:parts.append('<p>'+html.escape(x)+'</p>')
    for g in GROUPS:
        parts.append('<h2>'+g+'</h2><table><tr><th>Test</th><th>Participants</th><th>Within-map clusters</th><th>Across-map clusters</th></tr>')
        for r in rows:
            if r['group']==g:parts.append(f'<tr><td>{r["analysis"]}</td><td>{r["n_participants"]}</td><td>{r["significant_within_map"]}</td><td>{r["significant_across_maps"]}</td></tr>')
        parts.append('</table>')
        for suffix in ['role_condition_all_rois','historical_roi_panels']:parts.append(f'<img src="{(FIG/(g+"_"+suffix+".png")).as_uri()}">')
        for r in rows:
            if r['group']==g:
                for c in r['clusters']:
                    if c['p_within_map']<.05:parts.append('<pre>'+html.escape(json.dumps(dict(test=r['analysis'],**c),indent=2))+'</pre>')
    p=REPORT/'historical_participant_results.html';p.write_text('\n'.join(parts),encoding='utf-8')
    write(REPORT/'report_provenance.json',dict(specification_sha256=sh,outputs=[dict(path=str(p),sha256=sha(p)) for p in REPORT.iterdir() if p.is_file() and p.name!='report_provenance.json']))
if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--stage',choices=['prepare','pilot','full'],required=True);stage=args.parse_args().stage
    RUN.mkdir(parents=True,exist_ok=True);lock=RUN/'worker.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if stage=='prepare':prepare()
        else:run(stage)
    except Exception as e:status('failed',error=repr(e));traceback.print_exc();raise
    finally:lock.unlink()
