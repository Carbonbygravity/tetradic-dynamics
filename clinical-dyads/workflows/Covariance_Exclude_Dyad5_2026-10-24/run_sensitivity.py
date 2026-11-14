"""Isolated covariance membership sensitivity. All source project files read-only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import json, hashlib, importlib.util, traceback, time, pickle, shutil
from types import SimpleNamespace
import numpy as np
import pandas as pd
import h5py, mne
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parent
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'runtime_cache')
PROJECT=Path(os.environ['TETRADIC_CLINICAL_ROOT']).resolve()
T=PROJECT/'Tonic Analysis'; P=PROJECT/'Phasic Analysis'
OLD=T/'03_Results/Normative_Source_Repair_Run_01/Shared_Inverse_Two_Branch_Run_01'
PH=P/'03_Results/Harmonized_Phasic_Run_01'
HIST=P/'03_Results/Historical_Participant_Replication_Run_01'
COMB=T/'03_Results/Clinical_Combined_Tonic_Model_Run_01'
mne.set_log_level('ERROR')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2),encoding='utf8');os.replace(q,p)
def status(stage,**kw):
 d=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw);write(ROOT/'status.json',d);print(json.dumps(d),flush=True)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
gen=module('gen',(Path(__file__).resolve().parents[2]/'phasic/01_generate_clinical_phasic_mean.py'))
hist=module('hist',(Path(__file__).resolve().parents[2]/'phasic/11_historical_participant_rerun.py'))
eng=module('cluster_engine',(Path(__file__).resolve().parents[2]/'phasic/06_anatomical_and_paired_clusters.py'))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'runtime_cache')
ORDER=['LMA','LMR','FMA','FMR']
def source(ep,ix,inv,labels):
 chunk=ep[ix].copy().load_data().pick(gen.CHANNELS).reorder_channels(gen.CHANNELS)
 chunk.set_eeg_reference('average',projection=True,verbose=False)
 stcs=mne.minimum_norm.apply_inverse_epochs(chunk,inv,lambda2=1/9,method='dSPM',pick_ori='normal',return_generator=True,verbose=False)
 return np.asarray([mne.extract_label_time_course(s,labels,inv['src'],mode='mean',allow_empty=False,verbose=False) for s in stcs])
def welch(ts):
 psd,f=mne.time_frequency.psd_array_welch(ts,sfreq=1000,fmin=2,fmax=50,n_fft=1000,n_per_seg=1000,n_overlap=0,window='hamming',average='mean',remove_dc=True,verbose=False)
 return np.stack([psd[:,:,(f>=lo)&(f<=hi)].mean(-1) for _,lo,hi in gen.BANDS],axis=1).astype('f4')
def main():
 frozen=json.loads((OLD/'frozen_run_specification.json').read_text()); recs=[r for r in frozen['inputs'] if r['dyad']!=5]
 assert len(recs)==30 and sum(r['n_epochs'] for r in recs)==24436
 spec=dict(only_scientific_change='Exclude dyad5_P1 and dyad5_P2 from equal-participant normative covariance average',source_manifest_sha256=sha(OLD/'frozen_run_specification.json'),script_sha256=sha(Path(__file__)),participants=[r['participant'] for r in recs],preserved='EEG inputs, events, RT membership and covariates, source parameters, power definitions, statistical models, adjacency, seed and multiplicity',source_scripts={str(p):sha(p) for p in [(Path(__file__).resolve().parents[2]/'phasic/01_generate_clinical_phasic_mean.py'),(Path(__file__).resolve().parents[2]/'phasic/11_historical_participant_rerun.py'),(Path(__file__).resolve().parents[2]/'phasic/06_anatomical_and_paired_clusters.py'),(Path(__file__).resolve().parents[2]/'tonic/83_fit_clinical_combined_tonic_models.py'),(Path(__file__).resolve().parents[2]/'tonic/80_fit_shared_inverse_normative_models.py')]})
 sp=ROOT/'specification.json'
 if sp.exists():assert json.loads(sp.read_text())==spec
 else:write(sp,spec)
 for r in recs:
  st=Path(r['path']).stat();assert st.st_size==r['bytes'] and st.st_mtime_ns==r['mtime_ns']
 covdir=OLD/'04_Covariance_Inverse/participant_covariances'
 cs=[mne.read_cov(covdir/(r['participant']+'-cov.fif'),verbose=False) for r in recs];first=cs[0]
 assert all(c.ch_names==first.ch_names for c in cs)
 cov=mne.Covariance(np.mean([c.data for c in cs],0),names=first.ch_names,bads=first['bads'],projs=first['projs'],nfree=first['nfree'])
 cp=ROOT/'exclude5-cov.fif';ip=ROOT/'exclude5-inv.fif'
 if not cp.exists():cov.save(cp,overwrite=False)
 else:np.testing.assert_allclose(mne.read_cov(cp,verbose=False).data,cov.data,rtol=1e-12,atol=1e-25)
 status('inverse')
 if not ip.exists():
  assert sha(Path(frozen['forward']['path']))==frozen['forward']['sha256']
  sample=mne.read_epochs(recs[0]['path'],preload=True,verbose=False).pick(gen.CHANNELS).reorder_channels(gen.CHANNELS)
  forward=mne.read_forward_solution(frozen['forward']['path'],verbose=False)
  inv=mne.minimum_norm.make_inverse_operator(sample.info,forward=forward,noise_cov=cov,loose=.2,depth=.8,verbose=False)
  mne.minimum_norm.write_inverse_operator(ip,inv,overwrite=False,verbose=False)
 inv=mne.minimum_norm.read_inverse_operator(ip,verbose=False)
 labels=[l for l in mne.read_labels_from_annot('fsaverage',parc='aparc',subjects_dir=str(gen.SUBJECTS),verbose=False) if 'unknown' not in l.name]
 assert len(labels)==68
 # Reproduce saved old-inverse Welch and float32 ROI signals before regeneration.
 if not (ROOT/'pilot.json').exists():
  oldinv=mne.minimum_norm.read_inverse_operator(OLD/'04_Covariance_Inverse/shared_full_epochs-inv.fif',verbose=False);checks=[]
  for pid in ['dyad1_P1','dyad6_P2']:
   r=next(r for r in recs if r['participant']==pid);ep=mne.read_epochs(r['path'],preload=False,verbose=False)
   ix=[next(i for i,v in enumerate(r['event_labels']) if v.startswith(c)) for c in ORDER]
   ts=source(ep,ix,oldinv,labels)
   with np.load(OLD/'05_Source_Power/complete_epochs/participant_power'/f'{pid}_power.npz') as z:
    assert z['roi_labels'].tolist()==[l.name for l in labels];np.testing.assert_allclose(welch(ts),z['raw_power'][ix],rtol=1e-6,atol=1e-12)
   with h5py.File(PH/'normative'/f'{pid}_roi.h5','r') as z:
    reference=np.stack([z['roi_time_series'][i] for i in ix]);np.testing.assert_allclose(ts.astype('f4'),reference,rtol=1e-6,atol=1e-10)
   checks.append(dict(participant=pid,passed=True))
  write(ROOT/'pilot.json',dict(passed=True,checks=checks))
 out=ROOT/'participants';out.mkdir(exist_ok=True)
 for ri,r in enumerate(recs):
  pid=r['participant'];dest=out/f'{pid}.h5';mark=dest.with_suffix('.json')
  if mark.exists():assert sha(dest)==json.loads(mark.read_text())['sha256'];continue
  assert not dest.exists();partial=dest.with_suffix('.partial.h5');assert not partial.exists(),'Interrupted participant needs explicit review'
  ep=mne.read_epochs(r['path'],preload=False,verbose=False)
  assert np.array_equal(ep.events,r['events']) and np.array_equal(ep.selection,r['selection'])
  ix=np.array([ORDER.index(v[:3]) for v in r['event_labels']]);counts=np.bincount(ix,minlength=4);assert (counts>0).all()
  sums=np.zeros((4,68,6,2001),dtype='f8');power=np.zeros((len(ep),6,68),dtype='f4')
  with h5py.File(partial,'x') as z:
   d=z.create_dataset('roi_time_series',(len(ep),68,2001),dtype='f4',chunks=(1,68,2001),compression='lzf')
   for start in range(0,len(ep),8):
    stop=min(start+8,len(ep));ts=source(ep,list(range(start,stop)),inv,labels);power[start:stop]=welch(ts);ts=ts.astype('f4');d[start:stop]=ts;raw=gen.power(ts)
    for j,row in enumerate(raw):sums[ix[start+j]]+=row
    if start%64==0:status('source_and_power',participant=pid,participants_complete=ri,epochs_done=stop,epochs_total=len(ep))
   z.create_dataset('raw_power',data=power);z.create_dataset('tensor',data=np.log1p(sums/counts[:,None,None,None]),compression='lzf');z.create_dataset('cell_counts',data=counts);z.create_dataset('events',data=ep.events);z.create_dataset('selection',data=ep.selection)
   z.attrs['inverse_sha256']=sha(ip);z.attrs['complete']=True
  os.replace(partial,dest);write(mark,dict(sha256=sha(dest),participant=pid,epochs=len(ep)))
 status('generation_complete')
 # Copy the exact existing joint-model metadata and replace normative power rows only.
 inp=ROOT/'tonic_inputs';base=COMB/'01_Event_Linked_Inputs_82';folder=inp/'combined_primary';folder.mkdir(parents=True,exist_ok=True)
 mp=base/'combined_primary/model_metadata.csv';meta=pd.read_csv(mp,low_memory=False)
 with np.load(base/'combined_primary/model_power.npz') as z:Y=z['raw_power'].copy();features=z['features'].copy()
 clinical=~meta.cohort.eq('normative').to_numpy();oldclinical=Y[clinical].copy()
 for pid in meta.loc[~clinical,'original_participant_id'].unique():
  positions=np.flatnonzero(meta.original_participant_id.eq(pid)&meta.cohort.eq('normative'));rows=meta.iloc[positions].power_row.astype(int).to_numpy()
  with h5py.File(out/f'{pid}.h5','r') as z:
   np.testing.assert_array_equal(z['events'][:][rows,0],meta.iloc[positions].event_sample.to_numpy());Y[positions]=z['raw_power'][:].reshape(-1,408)[rows]
 np.testing.assert_array_equal(Y[clinical],oldclinical);shutil.copyfile(mp,folder/'model_metadata.csv')
 np.savez_compressed(folder/'model_power.npz',raw_power=Y,features=features)
 write(inp/'input_validation_complete.json',dict(status='clinical_combined_inputs_verified',clinical_power_unchanged=True,metadata_unchanged=sha(mp)==sha(folder/'model_metadata.csv'),source_validation_sha256=sha(base/'input_validation_complete.json')))
 if not (ROOT/'tonic_fits/model_run_complete.json').exists():
  status('tonic_models');fit=module('fit83',(Path(__file__).resolve().parents[2]/'tonic/83_fit_clinical_combined_tonic_models.py'));fit.MODELS=['combined_primary'];fit.run(SimpleNamespace(inputs=inp,output=ROOT/'tonic_fits',pilot=False))
 # Exact published participant-level phasic procedure, normative maps only.
 status('phasic_maps');adj=sparse.load_npz(HIST/'historical_adjacency.npz');sf=sparse.kron(adj,sparse.eye(6),format='csr');summaries=[];times=np.arange(-1000,1001)/1000
 maps=ROOT/'phasic';maps.mkdir(exist_ok=True)
 for bi in list(range(6))+[None]:
  for kind in hist.WEIGHTS:
   name=('space_frequency_' if bi is None else 'spatiotemporal_'+gen.BANDS[bi][0]+'_')+kind;dest=maps/(name+'.pkl');marker=dest.with_suffix('.json')
   if marker.exists():summaries.append(json.loads(marker.read_text()));continue
   status('phasic_map',analysis=name,maps_complete=len(summaries));rows=[]
   for r in recs:
    with h5py.File(out/(r['participant']+'.h5'),'r') as z:rows.append(hist.contrast(z['tensor'][:][None],kind,bi)[0])
   x=np.stack(rows);res=hist.test(x,sf if bi is None else adj,5000)
   np.testing.assert_allclose(res[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-9,atol=1e-9)
   summary=eng.summaries(res,x.shape[1:],[b[0] for b in gen.BANDS],[l.name for l in labels],times,3 if bi is None else 18,bi is None)
   with dest.open('xb') as f:pickle.dump(dict(t_obs=res[0],clusters=res[1],cluster_p_values=res[2],H0=res[3]),f)
   row=dict(analysis=name,clusters=summary,sha256=sha(dest));write(marker,row);summaries.append(row)
 write(ROOT/'phasic_summary.json',summaries)
 comparison=[]
 for row in summaries:
  old=json.loads((HIST/'NormHC'/(row['analysis']+'.json')).read_text())
  comparison.append(dict(analysis=row['analysis'],old_significant=[c for c in old['clusters'] if c['p_within_map']<.05],new_significant=[c for c in row['clusters'] if c['p_within_map']<.05]))
 write(ROOT/'phasic_comparison.json',comparison)
 # Reproduce the role-averaged 1224-test family from saved joint model checkpoints.
 def condition_rows(folder):
  rr=[]
  for p in sorted((folder/'checkpoints').glob('*.json')):
   ck=json.loads(p.read_text());f=ck['fit'];beta=np.asarray(f['coef']);cov=np.asarray(f['cov_beta'])
   for group,offset in [('normative_HC',None),('mixed_dyad_HC',8),('PT',9)]:
    v=np.zeros(13);v[4]=-1
    if offset is not None:v[offset]=-1
    est=float(v@beta);se=float(np.sqrt(v@cov@v));rr.append(dict(feature=ck['feature'],group=group,estimate=est,se=se,p=2*stats.norm.sf(abs(est/se)),valid=f['valid']))
  assert len(rr)==1224 and all(r['valid'] for r in rr)
  df=pd.DataFrame(rr);df['q']=multipletests(df.p,method='fdr_bh')[1];return df
 a=condition_rows(COMB/'03_Model_Fits_83/combined_primary');b=condition_rows(ROOT/'tonic_fits/combined_primary')
 a.merge(b,on=['feature','group'],suffixes=('_old','_new'),validate='one_to_one').to_csv(ROOT/'tonic_condition_comparison.csv',index=False)
 # Participant theta window summaries, paired unchanged clinical values remain at source.
 rows=[]
 for r in recs:
  with h5py.File(out/(r['participant']+'.h5'),'r') as z:a=z['tensor'][:]
  curve=((a[0,:,0]+a[1,:,0]-a[2,:,0]-a[3,:,0])/2).mean(0)
  with h5py.File(PH/'normative'/(r['participant']+'_cells.h5'),'r') as z:b=z['tensor'][:]
  oldcurve=((b[0,:,0]+b[1,:,0]-b[2,:,0]-b[3,:,0])/2).mean(0)
  rows.append(dict(participant=r['participant'],old_pre=oldcurve[times<0].mean(),new_pre=curve[times<0].mean(),old_post=oldcurve[times>=0].mean(),new_post=curve[times>=0].mean(),curve_correlation=np.corrcoef(curve,oldcurve)[0,1]))
 pd.DataFrame(rows).to_csv(ROOT/'theta_comparison.csv',index=False)
 status('analysis_complete_comparison_review_pending',manuscript_changed=False)
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e));traceback.print_exc();raise
