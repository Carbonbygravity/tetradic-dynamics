"""Anatomical-primary dyad reruns and paired PT-HC comparisons with historical sensitivity.

Each observation is the equal-weight mean of the two partners' log-power
contrasts. The test therefore estimates the mean dyad response contrast,
not a patient-control difference. Historical products remain untouched.
"""
import os,sys,json,pickle,hashlib,argparse,traceback,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(os.environ['TETRADIC_PHASIC_ROOT']).resolve()
RUN=ROOT/'03_Results/Anatomical_Dyad_Clusters_Run_01'
QA=ROOT/'05_Quality_Checks/Anatomical_Dyad_Clusters_Run_01'
INPUT=ROOT/'03_Results/Harmonized_Phasic_Run_01'
sys.path.insert(0,str(ROOT/'02_Scripts/_runtime'))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']=str(RUN/'runtime_cache')
import numpy as np,h5py,mne
from scipy import sparse,stats
from threadpoolctl import threadpool_limits
ORDER=['LMA','LMR','FMA','FMR']
WEIGHTS={'Condition':[.5,-.5,.5,-.5],'Role':[.5,.5,-.5,-.5],'Interaction':[-.5,.5,.5,-.5]}
SEED=20261008
MODE='dyad'
ADJACENCY='anatomical'
TOTAL=42
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf8');os.replace(tmp,p)
def status(stage,**kw):
    d=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw)
    write(RUN/'run_status.json',d);print(json.dumps(d),flush=True)
def inspect(cohort):
    path=INPUT/f'{cohort}_tensor.h5'
    complete=json.loads((INPUT/'generation_complete.json').read_text())
    assert sha(path)==complete['cohorts'][cohort]['sha256'],'Tensor hash changed'
    with h5py.File(path,'r') as f:
        assert f.attrs['complete'] and f.attrs['specification_sha256']==complete['specification_sha256']
        assert f['tensor'].shape==(30,4,68,6,2001)
        assert f['cell_order'].asstr()[:].tolist()==ORDER
        ids=f['subjects'].asstr()[:].tolist();dyads=f['dyads'][:];times=f['times_seconds'][:]
        assert len(set(ids))==30 and (f['cell_counts'][:]>0).all()
        np.testing.assert_allclose(times,np.arange(-1000,1001)/1000,atol=1e-12,rtol=0)
        pairs=[]
        for d in sorted(set(dyads.tolist())):
            ix=np.flatnonzero(dyads==d).tolist();assert len(ix)==2
            assert set(ids[i] for i in ix)=={f'dyad{d}_P1',f'dyad{d}_P2'}
            pairs.append(dict(dyad=int(d),indices=ix,subjects=[ids[i] for i in ix]))
        assert len(pairs)==15
        return dict(path=str(path),sha256=sha(path),pairs=pairs,roi_labels=f['roi_labels'].asstr()[:].tolist(),band_labels=f['band_labels'].asstr()[:].tolist(),times_seconds=times.tolist(),groups=f['groups'].asstr()[:].tolist() if 'groups' in f else None)
def dyad_contrast_fields(meta,kind,band=None):
    """Apply labeled participant contrast first, then equal-weight partner mean."""
    w=np.asarray(WEIGHTS[kind]);rows=[]
    with h5py.File(meta['path'],'r') as f:
        for pair in meta['pairs']:
            partner=[]
            for i in pair['indices']:
                a=f['tensor'][i] if band is None else f['tensor'][i,:,:,band,:]
                assert np.isfinite(a).all()
                # Space-frequency follows historical time averaging AFTER log1p.
                if band is None:a=a.mean(axis=-1)
                partner.append(np.tensordot(w,a,axes=(0,0)))
            rows.append((partner[0]+partner[1])/2)
    a=np.stack(rows)
    return a.reshape(15,-1) if band is None else a.transpose(0,2,1)
def group_indices(meta,pair):
    groups=meta['groups'];pt=[i for i in pair['indices'] if groups[i]=='PT'];hc=[i for i in pair['indices'] if groups[i]=='HC']
    assert len(pt)==len(hc)==1
    return pt[0],hc[0]
def validate_group_pairs(meta):
    spec=json.loads((INPUT/'frozen_specification.json').read_text())
    manifest=json.loads(Path(spec['clinical_input_manifest']).read_text())
    lookup={r['participant']:r['group'] for r in manifest['inputs']}
    with h5py.File(meta['path'],'r') as f:
        ids=f['subjects'].asstr()[:].tolist()
    assert meta['groups']==[lookup[pid] for pid in ids]
    for pair in meta['pairs']:group_indices(meta,pair)
def contrast_fields(meta,kind,band=None):
    if MODE=='dyad':return dyad_contrast_fields(meta,kind,band)
    rows=[];w=np.asarray(WEIGHTS[kind])
    with h5py.File(meta['path'],'r') as f:
        for pair in meta['pairs']:
            pt,hc=group_indices(meta,pair);values=[]
            for i in [pt,hc]:
                a=f['tensor'][i] if band is None else f['tensor'][i,:,:,band,:]
                assert np.isfinite(a).all()
                if band is None:a=a.mean(axis=-1)
                values.append(np.tensordot(w,a,axes=(0,0)))
            rows.append(values[0]-values[1])
    x=np.stack(rows)
    return x.reshape(15,-1) if band is None else x.transpose(0,2,1)
def cluster_test(x,adjacency,n,seed):
    fn=mne.stats.spatio_temporal_cluster_1samp_test if x.ndim==3 else mne.stats.permutation_cluster_1samp_test
    return fn(x,adjacency=adjacency,n_permutations=n,tail=0,threshold=None,
              seed=seed,n_jobs=1,out_type='indices',buffer_size=1000,verbose=False)
def summaries(result,shape,bands,rois,times,family_size,spacefreq):
    t,clusters,p,h0=result;rows=[]
    assert t.shape==shape and len(clusters)==len(p)
    assert np.isfinite(t).all() and np.isfinite(h0).all() and np.isfinite(p).all()
    assert ((p>=0)&(p<=1)).all()
    for j,(cl,prob) in enumerate(zip(clusters,p)):
        vals=t[cl];assert vals.size>0
        if spacefreq:
            flat=np.asarray(cl[0]);ri,bi=np.unravel_index(flat,(68,6));assert len(set(bi.tolist()))==1,'Unexpected cross-band edge'
            extra=dict(bands=[bands[k] for k in sorted(set(bi.tolist()))])
        else:
            ti,ri=cl;extra=dict(start_seconds=float(times[min(ti)]),end_seconds=float(times[max(ti)]))
        rows.append(dict(cluster_index=j,p_within_map=float(prob),p_across_maps_bonferroni=min(1.,float(prob)*family_size),
            n_points=int(vals.size),mean_t=float(vals.mean()),peak_abs_t=float(abs(vals).max()),
            direction='positive' if vals.sum()>0 else 'negative',rois=[rois[k] for k in sorted(set(np.asarray(ri).tolist()))],**extra))
    return rows
def main(stage):
    RUN.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    meta={c:inspect(c) for c in (['clinical','normative'] if MODE=='dyad' else ['clinical'])}
    if MODE=='paired':validate_group_pairs(meta['clinical'])
    if MODE=='dyad':
        for key in ['roi_labels','band_labels','times_seconds']:assert meta['clinical'][key]==meta['normative'][key]
    adjacency_source=ROOT/'01_Source_Data/Cortical_Boundary_Adjacency_Run_01/roi_adjacency.npz'
    anatomical=json.loads((adjacency_source.parent/'frozen_specification.json').read_text())
    assert anatomical['roi_labels']==meta['clinical']['roi_labels']
    assert sha(adjacency_source)==anatomical['adjacency_sha256']
    qa=json.loads((ROOT/'05_Quality_Checks/Cortical_Boundary_Adjacency_Run_01/validation.json').read_text())
    assert qa['passed'] and qa['adjacency_sha256']==sha(adjacency_source)
    adj=sparse.load_npz(adjacency_source).tocsr()
    assert adj.shape==(68,68) and (adj!=adj.T).nnz==0
    if MODE=='paired' and ADJACENCY=='historical':
        adjacency_source=ROOT/'03_Results/Normative_Cell_Order_Corrected_Run_01/source_cluster_results_theta_Condition.pkl'
        with adjacency_source.open('rb') as f:old=pickle.load(f)
        assert list(old['roi_labels'])==meta['clinical']['roi_labels']
        adj=sparse.csr_matrix(old['adjacency'])
    sfadj=sparse.kron(adj,sparse.eye(6,format='csr'),format='csr')
    rows,cols=sfadj.nonzero();assert np.all(rows%6==cols%6)
    spec=dict(input_cohorts=meta,cell_order=ORDER,contrast_weights=WEIGHTS,
        mode=MODE,adjacency_branch=ADJACENCY,inferential_unit=('15 paired patient-minus-HC contrast differences' if MODE=='paired' else '15 dyad averages per cohort'),
        justification=('Direct paired patient-minus-HC contrast within each clinical dyad, preserving pairing and keeping group labels separate.' if MODE=='paired' else 'Both partners averaged per dyad to avoid treating paired participants as independent. Anatomical-primary rerun with unchanged data and contrasts.'),
        assumptions='Dyads are independent; the joint null distribution of each dyad contrast field is symmetric for sign flipping.',
        user_approval='User approved anatomical-primary and historical-sensitivity analyses; paired group comparisons follow the agreed sequence.',
        n_permutations=5000,tail=0,threshold=None,automatic_cluster_forming_t=float(stats.t.ppf(.975,14)),seed=SEED,n_jobs=1,
        window_seconds=[-1,1],spatiotemporal='Six bands separately, three contrasts per cohort; consecutive time samples plus frozen selected ROI adjacency',
        space_frequency='Average logged cell values over all 2001 samples; ROI adjacency x band identity; no cross-band edges',
        adjacency_source=str(adjacency_source),adjacency_source_sha256=sha(adjacency_source),
        corrections='Within-map max-cluster correction; additionally Bonferroni across 18 spatiotemporal maps or three space-frequency contrasts within each cohort, reported as separate analysis families.',
        script_sha256=sha(Path(__file__)),versions=dict(mne=mne.__version__,numpy=np.__version__),
        limitations=['Full historical window retained, including Morlet edge effects; cluster extents do not establish precise effect onset or ROI-specific significance.','Anatomical primary connects full-mesh parcel boundaries within hemispheres; historical sensitivity retains projected centroid edges.','No RT adjustment or between-cohort test; patient-control differences are tested only in paired mode.','Matching methods does not establish comparable absolute dSPM scales.','Separate analysis families do not provide a single experiment-wide error guarantee.'])
    sp=RUN/'frozen_specification.json'
    if sp.exists():assert json.loads(sp.read_text())==spec,'Frozen specification changed'
    else:write(sp,spec)
    sh=sha(sp)
    if stage=='pilot':
        assert [float(np.dot([11,3,5,2],w)) for w in WEIGHTS.values()]==[5.5,3.5,-2.5]
        checks=[]
        for c in meta:
            x=contrast_fields(meta[c],'Condition');a=cluster_test(x,sfadj,32,SEED);b=cluster_test(x,sfadj,32,SEED)
            np.testing.assert_allclose(a[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(a[3],b[3])
            # Independently reconstruct the first dyad's condition contrast.
            with h5py.File(meta[c]['path'],'r') as f:
                pair=meta[c]['pairs'][0]['indices'];z=np.stack([f['tensor'][i] for i in pair]).mean(axis=0)
                if MODE=='paired':
                    pt,hc=group_indices(meta[c],meta[c]['pairs'][0]);z=f['tensor'][pt]-f['tensor'][hc]
                explicit=((z[0]+z[2]-z[1]-z[3])/2).mean(axis=-1).ravel()
            np.testing.assert_allclose(x[0],explicit,rtol=1e-10,atol=1e-12)
            y=contrast_fields(meta[c],'Role',0);st=cluster_test(y,adj,8,SEED)
            np.testing.assert_allclose(st[0],stats.ttest_1samp(y,0,axis=0).statistic,rtol=1e-12,atol=1e-12)
            checks.append(dict(cohort=c,dyads=15,pairing_passed=True,independent_contrast_passed=True,t_statistics_passed=True,seed_reproducibility_passed=True))
        write(QA/'pilot.json',dict(passed=True,specification_sha256=sh,checks=checks,band_disconnection_passed=True))
        status('pilot_passed');return
    pilot=json.loads((QA/'pilot.json').read_text());assert pilot['passed'] and pilot['specification_sha256']==sh
    results=[]
    for c in meta:
        dest=RUN/c;dest.mkdir(exist_ok=True)
        tasks=[('space_frequency',None,k) for k in WEIGHTS]+[('spatiotemporal',j,k) for j in range(6) for k in WEIGHTS]
        for family,band,kind in tasks:
            name=f'{family}_{meta[c]["band_labels"][band]+"_" if band is not None else ""}{kind}'
            out=dest/f'{name}.pkl';mark=dest/f'{name}.json'
            if out.exists():
                assert mark.exists();saved=json.loads(mark.read_text());assert saved['specification_sha256']==sh and saved['sha256']==sha(out)
                results.append(saved);continue
            assert not mark.exists()
            status('running',cohort=c,analysis=name,maps_complete=len(results),maps_total=TOTAL)
            x=contrast_fields(meta[c],kind,band);start=time.monotonic();result=cluster_test(x,sfadj if band is None else adj,5000,SEED)
            np.testing.assert_allclose(result[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-11,atol=1e-11)
            summary=summaries(result,x.shape[1:],meta[c]['band_labels'],meta[c]['roi_labels'],meta[c]['times_seconds'],3 if band is None else 18,band is None)
            t,clusters,p,h0=result;assert len(h0) in [0,5000]
            data=dict(t_obs=t,clusters=clusters,cluster_p_values=p,H0=h0,n_dyads=15,dyad_pairs=meta[c]['pairs'],
                roi_labels=meta[c]['roi_labels'],band_labels=meta[c]['band_labels'],times_seconds=meta[c]['times_seconds'],
                cell_order=ORDER,contrast=kind,weights=WEIGHTS[kind],cohort=c,analysis_family=family,band_index=band,
                adjacency=sfadj if band is None else adj,specification_sha256=sh)
            with out.open('xb') as f:pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
            saved=dict(cohort=c,analysis=name,specification_sha256=sh,sha256=sha(out),seconds=time.monotonic()-start,
                permutations_in_null=len(h0),no_suprathreshold_clusters=len(clusters)==0,clusters=summary,
                significant_within_map=sum(r['p_within_map']<.05 for r in summary),significant_across_maps=sum(r['p_across_maps_bonferroni']<.05 for r in summary))
            write(mark,saved);results.append(saved)
            status('map_complete',cohort=c,analysis=name,maps_complete=len(results),maps_total=TOTAL)
    assert len(results)==TOTAL
    write(RUN/'cluster_summary.json',dict(results=results,specification_sha256=sh))
    write(QA/'validation.json',dict(passed=True,maps_complete=TOTAL,specification_sha256=sh,dyads_per_cohort=15,
        checks=['input hashes','pair identities','ROI/band/time/cell alignment','independent t statistic agreement','result checksums'],statistics_run=True))
    write(RUN/'analysis_complete.json',dict(maps_complete=TOTAL,specification_sha256=sh,utc=datetime.now(timezone.utc).isoformat()))
    status('complete',maps_complete=TOTAL)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full'],required=True)
    p.add_argument('--mode',choices=['dyad','paired'],required=True)
    p.add_argument('--adjacency',choices=['anatomical','historical'],default='anatomical');args=p.parse_args()
    MODE=args.mode;ADJACENCY=args.adjacency;TOTAL=42 if MODE=='dyad' else 21
    assert not (MODE=='dyad' and ADJACENCY=='historical'),'Existing completed historical dyad results are reused'
    name='Anatomical_Dyad_Clusters_Run_01' if MODE=='dyad' else ('Paired_PT_HC_Anatomical_Run_01' if ADJACENCY=='anatomical' else 'Paired_PT_HC_Historical_Sensitivity_Run_01')
    RUN=ROOT/'03_Results'/name;QA=ROOT/'05_Quality_Checks'/name
    os.environ['NUMBA_CACHE_DIR']=str(RUN/'runtime_cache')
    try:
        with threadpool_limits(limits=1):main(args.stage)
    except Exception as e:
        status('failed',error=repr(e));traceback.print_exc();raise
