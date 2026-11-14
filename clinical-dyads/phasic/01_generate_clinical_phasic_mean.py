"""Clinical phasic mean-ROI generation; original inputs are read-only.

Use --stage pilot before --stage full. Results are participant HDF5 files,
not legacy pickle files. No cluster statistics or RT residualization are run.
"""
import argparse, hashlib, json, os, time, traceback, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ['TETRADIC_PHASIC_ROOT']).resolve()
TONIC = ROOT.parent / 'Tonic Analysis/03_Results/Clinical_Cleaned_Reconstruction_Run_01'
RUN = ROOT / '03_Results/Clinical_Cleaned_Mean_Morlet_Run_01'
QA = ROOT / '05_Quality_Checks/Clinical_Cleaned_Mean_Morlet_Run_01'
os.environ.setdefault('NUMBA_CACHE_DIR', str(RUN / 'runtime_cache'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent / '_runtime'))
import mne
import numpy as np
import h5py
from threadpoolctl import threadpool_limits

INV = TONIC / 'primary/inverse_operator-inv.fif'
FROZEN = TONIC / 'frozen_run_specification.json'
SUBJECTS = Path(os.environ['TETRADIC_SUBJECTS_DIR']).resolve()
CHANNELS = [f'E{i}' for i in range(1,129)]
FREQS = np.arange(4.,46.)
BANDS = [('theta',4,7),('low_alpha',8,9),('high_alpha',10,12),('low_beta',13,19),('high_beta',20,29),('gamma',30,45)]
MASKS = [(FREQS>=lo)&(FREQS<=hi) for _,lo,hi in BANDS]
BATCH = 8

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''): h.update(x)
    return h.hexdigest()
def write(p, data):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(data,indent=2),encoding='utf8'); os.replace(q,p)
def status(state, **kw):
    d=dict(utc=now(),state=state,**kw);write(RUN/'run_status.json',d);print(json.dumps(d),flush=True)
def load_epochs(r):
    p=Path(r['input']['path']);st=p.stat()
    if st.st_size!=r['input']['size_bytes'] or st.st_mtime_ns!=r['input']['mtime_ns']: raise ValueError('Input fingerprint changed: '+str(p))
    ep=mne.read_epochs(p,preload=False,verbose=False)
    if not np.array_equal(ep.events,r['events']) or not np.array_equal(ep.selection,r['selection']): raise ValueError('Event/selection mismatch')
    if len(ep)!=r['n_epochs'] or ep.info['sfreq']!=1000 or len(ep.times)!=2001 or not np.allclose([ep.tmin,ep.tmax],[-1,1]): raise ValueError('Epoch grid mismatch')
    names={v:k for k,v in ep.event_id.items()}
    if [names[int(x[2])] for x in ep.events]!=r['event_labels']: raise ValueError('Event labels changed')
    return ep
def source_roi(ep, indices, inverse, labels):
    chunk=ep[indices].copy().load_data().pick(CHANNELS).reorder_channels(CHANNELS)
    chunk.set_eeg_reference('average',projection=True,verbose=False)
    stcs=mne.minimum_norm.apply_inverse_epochs(chunk,inverse,lambda2=1/9,method='dSPM',pick_ori='normal',return_generator=True,verbose=False)
    ts=np.asarray([mne.extract_label_time_course(s,labels,inverse['src'],mode='mean',allow_empty=False,verbose=False) for s in stcs],dtype=np.float32)
    if not np.isfinite(ts).all(): raise ValueError('Invalid source time series')
    return ts
def power(ts):
    out=np.empty((len(ts),68,6,2001),np.float32)
    for start in range(0,68,4):
        tf=mne.time_frequency.tfr_array_morlet(ts[:,start:start+4],sfreq=1000,freqs=FREQS,n_cycles=FREQS/2,zero_mean=False,output='power',use_fft=True,n_jobs=1,verbose=False)
        for b,mask in enumerate(MASKS):out[:,start:start+4,b,:]=tf[:,:,mask,:].mean(axis=2)
    if not np.isfinite(out).all() or (out<0).any(): raise ValueError('Invalid band power')
    return out
def main(stage):
    RUN.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(FROZEN.read_text());records=frozen['inputs']
    inverse=mne.minimum_norm.read_inverse_operator(INV,verbose=False)
    labels=[l for l in mne.read_labels_from_annot('fsaverage',parc='aparc',subjects_dir=str(SUBJECTS),verbose=False) if 'unknown' not in l.name]
    if len(labels)!=68:raise ValueError('ROI count')
    # Preserve corrected tonic ROI order; original normative phasic extraction also uses mean.
    spec=dict(input_manifest=str(FROZEN),input_manifest_sha256=sha(FROZEN),inverse=str(INV),inverse_sha256=sha(INV),
        source=dict(method='dSPM',lambda2=1/9,pick_ori='normal',roi_extraction='mean',allow_empty=False),
        morlet=dict(freqs=FREQS.tolist(),n_cycles=(FREQS/2).tolist(),zero_mean=False,use_fft=True),
        freq_bins_by_band={b:FREQS[m].tolist() for (b,_,_),m in zip(BANDS,MASKS)},
        power_order='mean_frequency_power_then_log10_then_downstream_cell_average',log_eps=1e-12,
        time_units='seconds',time_range=[-1,1],sfreq=1000,batch_epochs=BATCH,roi_batch=4,
        role='response_event_role_not_recorded_actor_task_role',roi_labels=[l.name for l in labels],
        versions=dict(mne=mne.__version__,numpy=np.__version__,h5py=h5py.__version__),
        primary_participants=30,primary_dyads=15,primary_epochs=25332,
        caveat='Whole saved epochs retained. Morlet edge effects and statistical analysis windows require downstream treatment. No artifact clearance or normative equivalence claim.')
    path=RUN/'frozen_specification.json'
    if path.exists() and json.loads(path.read_text())!=spec: raise ValueError('Run specification changed')
    if not path.exists():write(path,spec)
    spec_hash=sha(path)
    if stage=='pilot':
        results=[];t0=time.monotonic()
        for pid in ['dyad1_P1','dyad15_P2']:
            r=next(r for r in records if r['participant']==pid);ep=load_epochs(r)
            ix=[next(i for i,e in enumerate(r['event_labels']) if e.startswith(c)) for c in ['LMR','LMA','FMR','FMA']]
            ts=source_roi(ep,ix,inverse,labels);raw=power(ts);log=np.log10(raw+1e-12).astype(np.float32)
            single=power(ts[:1]);np.testing.assert_allclose(raw[:1],single,rtol=1e-6,atol=1e-12)
            with np.load(TONIC/'primary/participant_power'/f'{pid}_power.npz') as old:
                if list(old['roi_labels'])!=spec['roi_labels']:raise ValueError('Tonic ROI order mismatch')
                psd,f=mne.time_frequency.psd_array_welch(ts,sfreq=1000,fmin=2,fmax=50,n_fft=1000,n_per_seg=1000,n_overlap=0,window='hamming',average='mean',remove_dc=True,verbose=False)
                welch=np.stack([psd[:,:,(f>=lo)&(f<=hi)].mean(axis=-1) for _,lo,hi in BANDS],axis=1)
                reference=old['raw_power'][ix]
                err=float(np.linalg.norm(welch-reference)/max(np.linalg.norm(reference),1e-30))
                if err>1e-5:raise ValueError('Tonic reconstruction reproduction discrepancy '+str(err))
            results.append(dict(participant=pid,indices=ix,events=ep.events[ix].tolist(),shape=list(raw.shape),tonic_relative_l2_error=err,finite=bool(np.isfinite(log).all()),batch_invariance_pass=True))
        write(QA/'pilot_validation.json',dict(state='passed',utc=now(),specification_sha256=spec_hash,seconds=time.monotonic()-t0,checks=results))
        status('pilot_passed',seconds=time.monotonic()-t0);return
    pilot=json.loads((QA/'pilot_validation.json').read_text())
    if pilot['state']!='passed' or pilot['specification_sha256']!=spec_hash:raise ValueError('Matching pilot required')
    dest=RUN/'participant_time_resolved';dest.mkdir(exist_ok=True)
    completed=[];start_all=time.monotonic()
    for r in records:
        pid=r['participant'];target=dest/f'{pid}_time_resolved.h5';marker=dest/f'{pid}_complete.json'
        if target.exists() and marker.exists():
            prior=json.loads(marker.read_text())
            if prior['specification_sha256']!=spec_hash or prior['output_sha256']!=sha(target):raise ValueError('Resume output mismatch')
            completed.append(prior);continue
        if target.exists():raise FileExistsError('Unmarked output '+str(target))
        ep=load_epochs(r);input_hash=sha(r['input']['path']);partial=dest/f'{pid}.partial.h5'
        if partial.exists():raise FileExistsError('Review interrupted partial file before retry: '+str(partial))
        status('running',participant=pid,participants_complete=len(completed),epochs_done=0,epochs_total=len(ep))
        t0=time.monotonic();minimum=float('inf');maximum=0.;zeros=0
        with h5py.File(partial,'x') as f:
            n=len(ep);f.attrs['specification_sha256']=spec_hash;f.attrs['input_sha256']=input_hash;f.attrs['participant']=pid
            for name,shape,chunks in [('roi_time_series',(n,68,2001),(1,68,2001)),('data_raw',(n,68,6,2001),(1,68,1,2001)),('data_log',(n,68,6,2001),(1,68,1,2001))]:
                f.create_dataset(name,shape=shape,dtype='f4',chunks=chunks,compression='lzf',shuffle=True)
            f.create_dataset('events',data=ep.events);f.create_dataset('selection',data=ep.selection);f.create_dataset('times_seconds',data=ep.times)
            for name,values in [('event_labels',r['event_labels']),('roi_labels',spec['roi_labels']),('band_labels',[b[0] for b in BANDS])]:f.create_dataset(name,data=np.asarray(values,dtype=h5py.string_dtype('utf-8')))
            for start in range(0,n,BATCH):
                stop=min(n,start+BATCH);ts=source_roi(ep,list(range(start,stop)),inverse,labels);raw=power(ts);log=np.log10(raw+1e-12).astype(np.float32)
                if not np.isfinite(log).all():raise ValueError('Invalid log power')
                f['roi_time_series'][start:stop]=ts;f['data_raw'][start:stop]=raw;f['data_log'][start:stop]=log
                minimum=min(minimum,float(raw.min()));maximum=max(maximum,float(raw.max()));zeros+=int(np.count_nonzero(raw==0))
                if start%64==0 or stop==n: f.flush();status('running',participant=pid,participants_complete=len(completed),epochs_done=stop,epochs_total=n,participant_seconds=time.monotonic()-t0)
            # Read every saved batch back: verify serialization and log relation before completion.
            for start in range(0,n,BATCH):
                a=f['data_raw'][start:start+BATCH];b=f['data_log'][start:start+BATCH];ts=f['roi_time_series'][start:start+BATCH]
                if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(ts).all() or (a<0).any():raise ValueError('Saved values invalid')
                np.testing.assert_array_equal(b,np.log10(a+1e-12).astype(np.float32))
            np.testing.assert_array_equal(f['events'][:],r['events']);np.testing.assert_array_equal(f['selection'][:],r['selection'])
            f.attrs['complete']=True
        os.replace(partial,target)
        record=dict(participant=pid,n_epochs=len(ep),specification_sha256=spec_hash,input_sha256=input_hash,output_sha256=sha(target),size_bytes=target.stat().st_size,raw_min=minimum,raw_max=maximum,raw_zero_count=zeros,all_saved_values_verified=True,seconds=time.monotonic()-t0,completed_utc=now())
        write(marker,record);completed.append(record)
    summary=dict(state='complete',utc=now(),participants=len(completed),epochs=sum(x['n_epochs'] for x in completed),specification_sha256=spec_hash,seconds=time.monotonic()-start_all,outputs=completed)
    if summary['participants']!=30 or summary['epochs']!=25332:raise ValueError('Incomplete coverage')
    write(QA/'full_validation.json',summary);write(RUN/'generation_complete.json',summary);status('complete',participants=30,epochs=25332)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['pilot','full'],required=True);args=p.parse_args()
    try:
        with threadpool_limits(limits=1):main(args.stage)
    except Exception:
        status('failed',stage=args.stage,traceback=traceback.format_exc());raise

