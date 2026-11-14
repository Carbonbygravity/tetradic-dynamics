"""Identity-preserving within-cohort phasic analysis and reproducible figures.

Uses approved harmonized inputs without regenerating or overwriting them.
Normative/clinical dyad tests are verified and reused. New tests are PT-only
and HC-only, 15 independent dyads in each, using the existing MNE engine.
All Role labels denote RESPONSE-EVENT roles, not participant task assignment.
"""
import os, sys, json, hashlib, pickle, argparse, importlib.util, traceback
from pathlib import Path
from datetime import datetime, timezone
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import h5py
from scipy import sparse, stats
ROOT=Path(os.environ['TETRADIC_PHASIC_ROOT']).resolve()
NAME='Within_Cohort_Identity_Run_01'
RUN=ROOT/'03_Results'/NAME
QA=ROOT/'05_Quality_Checks'/NAME
FIG=ROOT/'04_Figures'/NAME
REPORT=ROOT/'06_Manuscript_Text'/NAME
INPUT=ROOT/'03_Results/Harmonized_Phasic_Run_01'
HIST=Path(os.environ['TETRADIC_FIGURE_REFERENCE']).resolve()
ORDER=['LMA','LMR','FMA','FMR']
WEIGHTS={'Condition':[.5,-.5,.5,-.5],'Role':[.5,.5,-.5,-.5],'Interaction':[-.5,.5,.5,-.5]}
SEED=20261008
REUSED={'historical':'Harmonized_Dyad_Clusters_Run_01','anatomical':'Anatomical_Dyad_Clusters_Run_01'}

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(t,p)
def status(stage,**kw):
    d=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid());d.update(kw)
    write(RUN/'run_status.json',d);print(json.dumps(d),flush=True)
def inspect():
    complete=read(INPUT/'generation_complete.json');frozen=read(INPUT/'frozen_specification.json')
    manifest=read(Path(frozen['clinical_input_manifest']))
    identities={r['participant']:r['group'] for r in manifest['inputs']}
    meta={}
    for cohort in ['normative','clinical']:
        path=INPUT/f'{cohort}_tensor.h5'
        assert sha(path)==complete['cohorts'][cohort]['sha256']
        with h5py.File(path,'r') as f:
            assert f['tensor'].shape==(30,4,68,6,2001)
            assert f['cell_order'].asstr()[:].tolist()==ORDER
            assert f.attrs['complete'] and f.attrs['specification_sha256']==complete['specification_sha256']
            ids=f['subjects'].asstr()[:].tolist();dyads=f['dyads'][:]
            groups=f['groups'].asstr()[:].tolist() if cohort=='clinical' else ['NormHC']*30
            if cohort=='clinical':assert groups==[identities[i] for i in ids]
            pairs=[]
            for d in sorted(set(dyads.tolist())):
                ix=np.flatnonzero(dyads==d).tolist();assert len(ix)==2
                if cohort=='clinical':assert sorted(groups[i] for i in ix)==['HC','PT']
                pairs.append(dict(dyad=int(d),indices=ix,subjects=[ids[i] for i in ix],groups=[groups[i] for i in ix]))
            assert len(pairs)==15 and len(set(ids))==30
            times=f['times_seconds'][:];np.testing.assert_allclose(times,np.arange(-1000,1001)/1000,atol=1e-12,rtol=0)
            meta[cohort]=dict(path=str(path),sha256=sha(path),subjects=ids,groups=groups,pairs=pairs,
                rois=f['roi_labels'].asstr()[:].tolist(),bands=f['band_labels'].asstr()[:].tolist(),times=times.tolist())
    for k in ['rois','bands','times']:assert meta['normative'][k]==meta['clinical'][k]
    return meta,frozen

def verify_reuse(meta):
    out=[]
    for graph,name in REUSED.items():
        base=ROOT/'03_Results'/name;sh=sha(base/'frozen_specification.json')
        c=read(base/'analysis_complete.json');v=read(ROOT/'05_Quality_Checks'/name/'validation.json')
        assert c['maps_complete']==v['maps_complete']==42 and v['passed']
        assert c['specification_sha256']==v['specification_sha256']==sh
        spec=read(base/'frozen_specification.json')
        assert spec['cell_order']==ORDER and spec['contrast_weights']==WEIGHTS
        assert spec['n_permutations']==5000 and spec['tail']==0
        for cohort in meta:assert spec['input_cohorts'][cohort]['sha256']==meta[cohort]['sha256']
        rows=read(base/'cluster_summary.json')['results'];assert len(rows)==42
        for r in rows:
            path=base/r['cohort']/(r['analysis']+'.pkl')
            assert sha(path)==r['sha256'] and r['specification_sha256']==sh
        out.append(dict(adjacency=graph,path=str(base),specification_sha256=sh,verified_maps=42,results=rows))
    return out

def prepare():
    for p in [RUN,QA,FIG,REPORT]:p.mkdir(parents=True,exist_ok=True)
    assert not (RUN/'prepared.json').exists(),'Prepared run exists; do not overwrite'
    status('preparing')
    meta,frozen=inspect();reuse=verify_reuse(meta)
    hist=read(HIST);assert hist['saved_cell_order']==ORDER
    spec=dict(inputs=meta,cell_order=ORDER,contrast_weights=WEIGHTS,frequency_bins=frozen['frequency_bins'],
        seed=SEED,n_permutations=5000,tail=0,threshold='automatic t(.975,14)',
        analyses='21 maps per PT/HC group and adjacency; 84 new maps. Reuse 84 normative/clinical-dyad maps.',
        correction='Within-map max cluster; Bonferroni 18 ST or 3 SF maps per group, separately per adjacency.',
        groups='NormHC = 15 equal-partner dyad means; ClinicalDyad = 15 equal PT/HC means; PT and HC each one person per dyad.',
        role='Response-event role. Not recorded participant task role; no initiation/feedback timing established.',
        figure_selection='All 68 ROIs plus separately labeled fixed historical Figure 6 ROI sets, applied identically to all groups. No transferred historical significance.',
        figure_uncertainty='Pointwise t-based 95% intervals across 15 independent dyad units; not simultaneous or inferential cluster bands.',
        historical_manifest=str(HIST),historical_manifest_sha256=sha(HIST),script_sha256=sha(Path(__file__)),
        cluster_engine_sha256=sha((Path(__file__).resolve().parent/'06_anatomical_and_paired_clusters.py')),
        reuse=[{k:v for k,v in r.items() if k!='results'} for r in reuse],
        limitations=['Updated bins, cleaned epoch populations and source reconstruction differ from historical manuscript inputs.',
          'Normative partner averaging changes the historical participant-level inferential unit.',
          'Full epoch retains Morlet edge effects; cluster extent does not identify precise onset or pointwise effects.',
          'Source scale/cohort confounding remains; descriptive levels do not establish diagnosis effects.',
          'Historical ROI sets were selected in overlapping normative data; these panels are descriptive, not independent replication.'])
    write(RUN/'frozen_specification.json',spec);sh=sha(RUN/'frozen_specification.json')
    write(RUN/'reused_results.json',dict(verified=True,branches=reuse))
    write(RUN/'identity_mapping.json',dict(clinical=meta['clinical']['pairs'],normative=meta['normative']['pairs'],cell_order=ORDER,role=spec['role']))
    # Preserve participant curves and fixed historical ROI curves, before aggregation.
    raw={};regional={}
    for cohort in ['normative','clinical']:
        curves=[];panels=[]
        with h5py.File(meta[cohort]['path'],'r') as f:
            for i in range(30):
                a=f['tensor'][i];assert np.isfinite(a).all()
                curves.append(a.mean(axis=1))
                pp=[]
                for panel in hist['panels']:
                    ri=[meta[cohort]['rois'].index(r) for r in panel['roi_labels']]
                    bi=meta[cohort]['bands'].index(panel['band']);ci=[0,1] if panel['role']=='Leader' else [2,3]
                    pp.append(a[ci][:,ri,bi,:].mean(axis=1))
                panels.append(np.stack(pp))
        raw[cohort]=np.stack(curves);regional[cohort]=np.stack(panels)
    def groups(values):
        n=np.stack([values['normative'][p['indices']].mean(axis=0) for p in meta['normative']['pairs']])
        pt=np.stack([values['clinical'][p['indices'][p['groups'].index('PT')]] for p in meta['clinical']['pairs']])
        hc=np.stack([values['clinical'][p['indices'][p['groups'].index('HC')]] for p in meta['clinical']['pairs']])
        return dict(NormHC=n,ClinicalDyad=(pt+hc)/2,PT=pt,HC=hc)
    g=groups(raw);r=groups(regional)
    np.savez_compressed(RUN/'descriptive_curves.npz',times=np.asarray(meta['clinical']['times']),
        **g,**{k+'_historical_roi':v for k,v in r.items()},
        normative_participants=raw['normative'],clinical_participants=raw['clinical'])
    np.testing.assert_allclose(g['ClinicalDyad'],(g['PT']+g['HC'])/2,rtol=0,atol=0)
    write(RUN/'historical_panel_reference.json',hist)
    figures(g,r,meta['clinical'],hist)
    write(QA/'preparation_validation.json',dict(passed=True,specification_sha256=sh,reused_maps=84,
        checks=['tensor hashes and axes','PT/HC identities from source manifest','one PT and one HC per dyad','equal partner means','all reused result hashes','fixed identical historical ROI sets']))
    write(RUN/'prepared.json',dict(passed=True,specification_sha256=sh,utc=datetime.now(timezone.utc).isoformat()))
    write(REPORT/'run_record.json',dict(status='prepared; new PT/HC cluster tests not yet complete',specification=spec,
        script=str(Path(__file__)),execution_log=str(RUN/'analysis.log'),outputs=str(RUN),figures=str(FIG)))
    status('prepared',reused_maps=84,new_maps_complete=0,new_maps_total=84)

def figures(groups,regional,meta,hist):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.asarray(meta['times']);cut=abs(t)<=.8
    colors=['#c43944','#2472ad'];limits=[]
    def ci(a):return a.mean(axis=0),stats.t.ppf(.975,14)*a.std(axis=0,ddof=1)/np.sqrt(15)
    for b in range(6):
        bounds=[ci(g[:,:,b,:]) for g in groups.values()]
        low=min((m-e)[:,cut].min() for m,e in bounds);high=max((m+e)[:,cut].max() for m,e in bounds)
        pad=.05*(high-low);limits.append((low-pad,high+pad))
    files=[]
    def save(fig,name):
        for ext in ['png','pdf','svg']:
            path=FIG/(name+'.'+ext);fig.savefig(path,dpi=160);files.append(dict(path=str(path),sha256=sha(path)))
        plt.close(fig)
    for name,a in groups.items():
        fig,axs=plt.subplots(6,2,figsize=(12,16),sharex=True)
        for b,band in enumerate(meta['bands']):
            for role in range(2):
                ax=axs[b,role]
                for c,label in enumerate(['Matching','Mirroring']):
                    m,e=ci(a[:,role*2+c,b,:]);ax.plot(t,m,color=colors[c],label=label)
                    ax.fill_between(t,m-e,m+e,color=colors[c],alpha=.13)
                ax.axvline(0,color='gray',ls=':',lw=1);ax.set_xlim(-.8,.8);ax.set_ylim(limits[b]);ax.grid(alpha=.15)
                ax.set_title(f'{band.replace("_"," ")} | {"Leader" if role==0 else "Follower"} response event')
                if role==0:ax.set_ylabel('Mean log1p power')
        axs[0,0].legend();axs[-1,0].set_xlabel('Time from response event (s)');axs[-1,1].set_xlabel('Time from response event (s)')
        fig.suptitle(name+' | all 68 ROIs | updated frequency bins',fontsize=17)
        fig.text(.5,.014,'15 dyad units; pointwise 95% intervals. Event role is not participant task role. No significance shading.',ha='center',fontsize=10)
        fig.tight_layout(rect=[0,.035,1,.96]);save(fig,name+'_role_condition_all_rois')
    fig,axs=plt.subplots(6,2,figsize=(12,16),sharex=True)
    for b,band in enumerate(meta['bands']):
        for role in range(2):
            ax=axs[b,role]
            for group,color in [('PT','#b93e48'),('HC','#198578')]:
                for c,condition in enumerate(['Matching','Mirroring']):
                    m,e=ci(groups[group][:,2*role+c,b,:]);ax.plot(t,m,color=color,ls='-' if c==0 else '--',label=group+' '+condition)
            ax.axvline(0,color='gray',ls=':',lw=1);ax.set_xlim(-.8,.8);ax.set_ylim(limits[b]);ax.grid(alpha=.15)
            ax.set_title(f'{band.replace("_"," ")} | {"Leader" if role==0 else "Follower"} response event')
            if role==0:ax.set_ylabel('Mean log1p power')
    axs[0,0].legend(ncol=2,fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Time from response event (s)')
    fig.suptitle('Clinical dyads | PT and HC identities retained',fontsize=17)
    fig.text(.5,.015,'Solid = Matching; dashed = Mirroring. Event role is not participant task role. Descriptive means.',ha='center')
    fig.tight_layout(rect=[0,.035,1,.96]);save(fig,'Clinical_PT_HC_overlay')
    # Same six historical panels and ROI sets for each group; no old p values/windows overlaid.
    lim=[]
    for j in range(6):
        z=[ci(a[:,j]) for a in regional.values()];lo=min((m-e)[:,cut].min() for m,e in z);hi=max((m+e)[:,cut].max() for m,e in z)
        lim.append((lo-.05*(hi-lo),hi+.05*(hi-lo)))
    for name,a in regional.items():
        fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True)
        for j,ax in enumerate(axs.ravel()):
            panel=hist['panels'][j]
            for c,condition in enumerate(['Matching','Mirroring']):
                m,e=ci(a[:,j,c]);ax.plot(t,m,color=colors[c],label=condition);ax.fill_between(t,m-e,m+e,color=colors[c],alpha=.13)
            ax.axvline(0,color='gray',ls=':',lw=1);ax.set_xlim(-.8,.8);ax.set_ylim(lim[j]);ax.grid(alpha=.15)
            ax.set_title(f'{panel["role"]} response event | {panel["band"].replace("_"," ")} | {len(panel["roi_labels"])} ROIs')
            ax.set_ylabel('Mean log1p power')
        axs[0,0].legend()
        for ax in axs[-1]:ax.set_xlabel('Time from response event (s)')
        fig.suptitle(name+' | historical Figure 6 ROI panels, updated inputs',fontsize=16)
        fig.text(.5,.015,'Fixed historical ROI selections; descriptive only. Pointwise 95% intervals. Historical p values do not apply.',ha='center')
        fig.tight_layout(rect=[0,.04,1,.95]);save(fig,name+'_historical_roi_panels')
    write(FIG/'figure_manifest.json',dict(outputs=files,full_saved_epoch=[-1,1],display_epoch=[-.8,.8],
        axes='Same y limits for corresponding panels across groups',phase_mapping='Response event at zero only; initiation and feedback not independently established'))

def engine():
    sys.path.insert(0,str(ROOT/'02_Scripts/_runtime'))
    os.environ['NUMBA_CACHE_DIR']=str(RUN/'runtime_cache')
    s=importlib.util.spec_from_file_location('existing_engine',(Path(__file__).resolve().parent/'06_anatomical_and_paired_clusters.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fields(meta,group,kind,band):
    out=[]
    with h5py.File(meta['path'],'r') as f:
        for pair in meta['pairs']:
            i=pair['indices'][pair['groups'].index(group)]
            a=f['tensor'][i] if band is None else f['tensor'][i,:,:,band,:]
            assert np.isfinite(a).all()
            if band is None:a=a.mean(axis=-1)
            out.append(np.tensordot(WEIGHTS[kind],a,axes=(0,0)))
    x=np.stack(out)
    return x.reshape(15,-1) if band is None else x.transpose(0,2,1)

def graphs(meta):
    a=ROOT/'01_Source_Data/Cortical_Boundary_Adjacency_Run_01/roi_adjacency.npz'
    v=read(ROOT/'05_Quality_Checks/Cortical_Boundary_Adjacency_Run_01/validation.json')
    assert v['passed'] and v['adjacency_sha256']==sha(a)
    with (ROOT/'03_Results/Normative_Cell_Order_Corrected_Run_01/source_cluster_results_theta_Condition.pkl').open('rb') as f:h=pickle.load(f)
    assert list(h['roi_labels'])==meta['rois']
    return {'anatomical':sparse.load_npz(a).tocsr(),'historical':sparse.csr_matrix(h['adjacency'])}

def analyse(stage):
    spec=read(RUN/'frozen_specification.json');sh=sha(RUN/'frozen_specification.json')
    assert spec['script_sha256']==sha(Path(__file__))
    assert spec['cluster_engine_sha256']==sha((Path(__file__).resolve().parent/'06_anatomical_and_paired_clusters.py'))
    assert read(RUN/'prepared.json')['specification_sha256']==sh
    meta,_=inspect();meta=meta['clinical'];adjs=graphs(meta);m=engine()
    tasks=[(None,k) for k in WEIGHTS]+[(b,k) for b in range(6) for k in WEIGHTS]
    if stage=='pilot':
        checks=[]
        for group in ['PT','HC']:
            for graph,adj in adjs.items():
                for band,kind in [(None,'Condition'),(0,'Role')]:
                    x=fields(meta,group,kind,band);a=sparse.kron(adj,sparse.eye(6),format='csr') if band is None else adj
                    r=m.cluster_test(x,a,16,SEED);r2=m.cluster_test(x,a,16,SEED)
                    np.testing.assert_allclose(r[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-11,atol=1e-11)
                    np.testing.assert_array_equal(r[2],r2[2]);np.testing.assert_array_equal(r[3],r2[3])
                    with h5py.File(meta['path'],'r') as f:
                        pair=meta['pairs'][0];z=f['tensor'][pair['indices'][pair['groups'].index(group)]]
                    explicit=((z[0]+z[2]-z[1]-z[3])/2).mean(axis=-1).ravel() if band is None else ((z[0,:,0]+z[1,:,0]-z[2,:,0]-z[3,:,0])/2).T
                    np.testing.assert_allclose(x[0],explicit,rtol=1e-10,atol=1e-12)
                    checks.append(dict(group=group,graph=graph,band=band,passed=True))
        write(QA/'pilot.json',dict(passed=True,specification_sha256=sh,checks=checks));status('pilot_passed');return
    assert read(QA/'pilot.json')['passed'] and read(QA/'pilot.json')['specification_sha256']==sh
    results=[]
    for group in ['PT','HC']:
        for band,kind in tasks:
            name=('space_frequency_' if band is None else 'spatiotemporal_'+meta['bands'][band]+'_')+kind
            x=fields(meta,group,kind,band)
            for graph,adj in adjs.items():
                folder=RUN/graph/group;folder.mkdir(parents=True,exist_ok=True);dest=folder/(name+'.pkl');marker=folder/(name+'.json')
                if dest.exists():
                    saved=read(marker);assert saved['sha256']==sha(dest) and saved['specification_sha256']==sh
                    results.append(saved);continue
                assert not marker.exists()
                status('running',group=group,adjacency=graph,analysis=name,new_maps_complete=len(results),new_maps_total=84)
                a=sparse.kron(adj,sparse.eye(6),format='csr') if band is None else adj
                result=m.cluster_test(x,a,5000,SEED)
                np.testing.assert_allclose(result[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-11,atol=1e-11)
                summaries=m.summaries(result,x.shape[1:],meta['bands'],meta['rois'],meta['times'],3 if band is None else 18,band is None)
                t,clusters,p,h0=result;assert len(h0) in [0,5000]
                data=dict(t_obs=t,clusters=clusters,cluster_p_values=p,H0=h0,group=group,analysis=name,adjacency=graph,
                    n_independent_dyads=15,role=spec['role'],cell_order=ORDER,weights=WEIGHTS[kind],roi_labels=meta['rois'],band_labels=meta['bands'],times_seconds=meta['times'],specification_sha256=sh)
                with dest.open('xb') as f:pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
                saved=dict(group=group,adjacency=graph,analysis=name,path=str(dest),sha256=sha(dest),specification_sha256=sh,
                    clusters=summaries,significant_within_map=sum(c['p_within_map']<.05 for c in summaries),
                    significant_across_maps=sum(c['p_across_maps_bonferroni']<.05 for c in summaries))
                write(marker,saved);results.append(saved)
                status('map_complete',new_maps_complete=len(results),new_maps_total=84,group=group,analysis=name,adjacency=graph)
    assert len(results)==84
    for r in results:assert sha(Path(r['path']))==r['sha256']
    for group in ['PT','HC']:
        for f in (RUN/'anatomical'/group).glob('*.pkl'):
            with f.open('rb') as h:a=pickle.load(h)
            with (RUN/'historical'/group/f.name).open('rb') as h:b=pickle.load(h)
            np.testing.assert_array_equal(a['t_obs'],b['t_obs'])
    write(RUN/'cluster_summary.json',dict(results=results,specification_sha256=sh))
    build_report(results,spec)
    write(QA/'validation.json',dict(passed=True,new_maps=84,reused_maps=84,specification_sha256=sh,
        checks=['input hashes','group identity and response-event alignment','independent contrast arithmetic and t statistics','fixed-seed pilot','all result hashes','identical t maps across graphs']))
    write(RUN/'analysis_complete.json',dict(passed=True,new_maps=84,reused_maps=84,specification_sha256=sh,utc=datetime.now(timezone.utc).isoformat()))
    status('complete',new_maps_complete=84,new_maps_total=84)

def build_report(results,spec):
    import html
    allrows=[]
    for branch in read(RUN/'reused_results.json')['branches']:
        for r in branch['results']:
            allrows.append(dict(r,group='NormHC' if r['cohort']=='normative' else 'ClinicalDyad',adjacency=branch['adjacency']))
    allrows+=results
    write(REPORT/'all_results.json',dict(results=allrows,specification_sha256=sha(RUN/'frozen_specification.json')))
    parts=['<!doctype html><meta charset="utf-8"><title>Within-cohort phasic results</title><style>body{font:16px Arial;max-width:1300px;margin:40px auto}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}img{max-width:100%}</style><h1>Within-cohort phasic results</h1>',
      '<p>Order: normative, clinical dyad mean, PT, HC. Response-event roles are not participant task assignments. New PT/HC results and verified reused dyad results use 5,000 two-sided sign flips. No direct group-difference inference from comparing within-group significance.</p>']
    parts+=['<p>'+html.escape(s)+'</p>' for s in spec['limitations']]
    for group in ['NormHC','ClinicalDyad','PT','HC']:
        parts.append('<h2>'+group+'</h2><table><tr><th>Adjacency</th><th>Test</th><th>Within-map clusters p&lt;.05</th><th>Across-map clusters p&lt;.05</th></tr>')
        for r in allrows:
            if r['group']==group:parts.append(f'<tr><td>{r["adjacency"]}</td><td>{r["analysis"]}</td><td>{r["significant_within_map"]}</td><td>{r["significant_across_maps"]}</td></tr>')
        parts.append('</table>')
        for suffix in ['role_condition_all_rois','historical_roi_panels']:
            parts.append(f'<img src="{(FIG/(group+"_"+suffix+".png")).as_uri()}" alt="{group} {suffix}">')
    parts.append('<h2>Significant within-map clusters</h2><p>Across-map p uses 18 ST or 3 SF tests per group and adjacency. Extent is not pointwise significance.</p>')
    for r in allrows:
        for c in r['clusters']:
            if c['p_within_map']<.05:parts.append('<pre>'+html.escape(json.dumps(dict(group=r['group'],adjacency=r['adjacency'],analysis=r['analysis'],**c),indent=2))+'</pre>')
    (REPORT/'within_cohort_results.html').write_text('\n'.join(parts),encoding='utf-8')
    outputs=[p for p in REPORT.iterdir() if p.is_file() and p.name!='report_provenance.json']
    write(REPORT/'report_provenance.json',dict(specification_sha256=sha(RUN/'frozen_specification.json'),outputs=[dict(path=str(p),sha256=sha(p)) for p in outputs]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','pilot','full'],required=True);args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);lock=RUN/'worker.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if args.stage=='prepare':prepare()
        else:analyse(args.stage)
    except Exception as e:status('failed',error=repr(e));traceback.print_exc();raise
    finally:lock.unlink()
