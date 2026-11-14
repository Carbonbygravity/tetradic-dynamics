"""Approved focused sensitivity; separate local artifacts, originals read only."""
import os,sys,json,pickle,hashlib,traceback,time
from pathlib import Path
from datetime import datetime,timezone
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
P=Path(os.environ['TETRADIC_PHASIC_ROOT']).resolve()
R=Path(os.environ['TETRADIC_THETA_OUTPUT']).resolve()
sys.path.insert(0,str(P/'02_Scripts/_runtime'))
os.environ['NUMBA_CACHE_DIR']=str(R/'runtime_cache')
import numpy as np,h5py,mne
from scipy import sparse,stats
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(t,p)
def status(stage,**kw):
 d=dict(stage=stage,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw);write(R/'run_status.json',d);print(json.dumps(d),flush=True)
def test(x,a,n,seed):return mne.stats.spatio_temporal_cluster_1samp_test(x,adjacency=a,n_permutations=n,tail=0,threshold=None,n_jobs=1,seed=seed,out_type='indices',buffer_size=1000,verbose=False)
def main():
 assert not (R/'frozen_specification.json').exists(),'Refuse duplicate run or overwrite'
 R.mkdir(exist_ok=True);status('preparing')
 base=P/'03_Results/Historical_Participant_Replication_Run_01';orig=read(base/'frozen_specification.json')
 inp=Path(orig['inputs']['clinical']['path']);assert sha(inp)==orig['inputs']['clinical']['sha256']
 hist=base/'historical_adjacency.npz';assert sha(hist)==orig['adjacency_sha256'];ha=sparse.load_npz(hist)
 anat=P/'01_Source_Data/Cortical_Boundary_Adjacency_Run_01/roi_adjacency.npz';ast=read(anat.parent/'frozen_specification.json');assert sha(anat)==ast['adjacency_sha256'];aa=sparse.load_npz(anat)
 with h5py.File(inp) as f:
  assert f['cell_order'].asstr()[:].tolist()==['LMA','LMR','FMA','FMR'];assert f['band_labels'].asstr()[0]=='theta'
  z=f['tensor'][:,:,:,0,:];dyads=f['dyads'][:];groups=f['groups'].asstr()[:];ids=f['subjects'].asstr()[:].tolist();rois=f['roi_labels'].asstr()[:].tolist();times=f['times_seconds'][:]
 assert ast['roi_labels']==rois
 fields={};full={};subjects={}
 for g in ['PT','HC']:
  ix=np.flatnonzero(groups==g);assert dyads[ix].tolist()==list(range(1,16))
  a=z[ix];full[g]=((a[:,0]+a[:,1])/2-(a[:,2]+a[:,3])/2).transpose(0,2,1)
  fields[g]=full[g][1:];subjects[g]=[ids[i] for i in ix[1:]]
 full['PT_minus_HC']=full['PT']-full['HC'];fields['PT_minus_HC']=full['PT_minus_HC'][1:]
 tasks=[('PT','historical',20261010,base/'PT/spatiotemporal_theta_Role.pkl'),('HC','historical',20261010,base/'HC/spatiotemporal_theta_Role.pkl')]
 for branch in ['anatomical','historical']:
  name='Paired_PT_HC_'+('Anatomical' if branch=='anatomical' else 'Historical_Sensitivity')+'_Run_01'
  tasks.append(('PT_minus_HC',branch,20261008,P/'03_Results'/name/'clinical/spatiotemporal_theta_Role.pkl'))
 sources=[]
 for g,b,seed,path in tasks:
  marker=read(path.with_suffix('.json'));assert sha(path)==marker['sha256']
  with path.open('rb') as f:old=pickle.load(f)
  np.testing.assert_allclose(stats.ttest_1samp(full[g],0,axis=0).statistic,old['t_obs'],rtol=1e-10,atol=1e-10)
  sources.append(dict(group=g,branch=b,path=str(path),sha256=sha(path)))
 spec=dict(scope='Only theta Role: PT, HC historical adjacency; paired PT-minus-HC anatomical primary and historical sensitivity. Four maps.',omitted_dyads=[1],retained_dyads=list(range(2,16)),subjects=subjects,n=14,theta_hz=[4,5,6,7],window_seconds=[-1,1],n_permutations=5000,tail=0,threshold=None,automatic_t_threshold=float(stats.t.ppf(.975,13)),seeds=dict(within_group=20261010,paired=20261008),input_path=str(inp),input_sha256=sha(inp),historical_adjacency_sha256=sha(hist),anatomical_adjacency_sha256=sha(anat),source_results=sources,script_sha256=sha(Path(__file__)),versions=dict(mne=mne.__version__,numpy=np.__version__),correction='Within-map primary; original Bonferroni x18 separately for comparability, not a newly reduced family.',limitations='Post hoc diagnostic sensitivity. Reuses full-sample source covariance/inverse; does not remove upstream influence. No new cleaning or primary exclusions. No accommodation mechanism test.')
 write(R/'frozen_specification.json',spec);sh=sha(R/'frozen_specification.json')
 for g,b,seed,_ in tasks:
  a=ha if b=='historical' else aa;x=fields[g];r=test(x,a,16,seed);r2=test(x,a,16,seed)
  np.testing.assert_array_equal(r[2],r2[2]);np.testing.assert_array_equal(r[3],r2[3]);np.testing.assert_allclose(r[0],stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-10,atol=1e-10)
 write(R/'pilot_validation.json',dict(passed=True,specification_sha256=sh,checks=['Four full-sample t maps independently reproduced','Input and adjacency hashes verified','14 identities per group and matching dyads','Fixed-seed pilots reproduced']))
 rows=[]
 for g,b,seed,_ in tasks:
  status('running',group=g,adjacency=b,maps_complete=len(rows),maps_total=4)
  x=fields[g];t,cl,ps,h0=test(x,ha if b=='historical' else aa,5000,seed)
  assert len(h0)==5000 and np.isfinite(h0).all();np.testing.assert_allclose(t,stats.ttest_1samp(x,0,axis=0).statistic,rtol=1e-10,atol=1e-10)
  clusters=[]
  for i,(c,p) in enumerate(zip(cl,ps)):
   ti,ri=c;clusters.append(dict(cluster_index=i,direction='positive' if t[c].sum()>0 else 'negative',p_within_map=float(p),p_bonferroni18=min(1.,float(p)*18),start_seconds=float(times[min(ti)]),end_seconds=float(times[max(ti)]),rois=[rois[j] for j in sorted(set(ri))],n_points=len(ti)))
  dest=R/(g+'_'+b+'.pkl')
  with dest.open('xb') as f:pickle.dump(dict(t_obs=t,clusters=cl,cluster_p_values=ps,H0=h0,specification_sha256=sh,n=14),f)
  row=dict(group=g,adjacency=b,seed=seed,path=str(dest),sha256=sha(dest),clusters=clusters);rows.append(row);write(dest.with_suffix('.json'),row);status('map_complete',group=g,adjacency=b,maps_complete=len(rows),maps_total=4)
 write(R/'cluster_summary.json',dict(results=rows,specification_sha256=sh))
 for r in rows:assert sha(Path(r['path']))==r['sha256']
 write(R/'validation.json',dict(passed=True,maps_complete=4,specification_sha256=sh,result_hashes=[{k:r[k] for k in ['path','sha256']} for r in rows],pilot_passed=True,independent_t_checks_passed=True))
 write(R/'analysis_complete.json',dict(passed=True,maps_complete=4,specification_sha256=sh));status('complete',maps_complete=4,maps_total=4)
if __name__=='__main__':
 try:main()
 except Exception:
  if R.exists():status('failed',traceback=traceback.format_exc())
  raise
