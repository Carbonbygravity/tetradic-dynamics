"""Isolated correction of tonic sensitivity; original sources remain read-only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import importlib.util, json, hashlib, shutil, time, traceback, sys
import numpy as np
import pandas as pd
import h5py

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tonic_separate_cohort_sensitivity'
T=Path(os.environ['TETRADIC_TONIC_ROOT']).resolve()
OLD=T/'03_Results/Normative_Source_Repair_Run_01/Shared_Inverse_Two_Branch_Run_01/06_Models'
INPUT=OLD/'Event_Linked_Inputs_79/complete_epochs'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadmod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,v):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(temp,p)
def status(stage,**kw):
 v=dict(stage=stage,pid=os.getpid(),**kw);write(OUT/'status.json',v);print(json.dumps(v),flush=True)
def main():
 OUT.mkdir(exist_ok=True)
 assert not (OUT/'comparison_summary.json').exists(),'Preserve completed comparison'
 fit=loadmod('fit80',(Path(__file__).resolve().parents[2]/'tonic/80_fit_shared_inverse_normative_models.py'))
 cmp=loadmod('cmp90',(Path(__file__).resolve().parents[2]/'tonic/90_compare_separate_cohort_condition_effects.py'))
 sources=[Path(__file__),(Path(__file__).resolve().parents[2]/'tonic/80_fit_shared_inverse_normative_models.py'),(Path(__file__).resolve().parents[2]/'tonic/90_compare_separate_cohort_condition_effects.py'),INPUT/'model_metadata.csv',INPUT/'model_power.npz']
 sources+=list((cmp.NORM/'checkpoints').glob('*.json'))+list((cmp.CLIN/'checkpoints').glob('*.json'))
 fingerprints={str(p):sha(p) for p in sources}
 spec=dict(purpose='Covariance excludes normative dyad 5; manuscript separate-cohort models',fit_specification=fit.SPEC,
   scope='Refit 408 normative full models with stage80 engine; retain original clinical fits. Stage90 contrasts and two 1224-test BH families unchanged. Reduced-model LRTs not used by stage90 and not rerun.',
   original_sources_sha256=fingerprints,python=sys.version,clinical_fits_unchanged=True,existing_joint_sensitivity_preserved=True)
 if not (OUT/'specification.json').exists():write(OUT/'specification.json',spec)
 else:assert json.loads((OUT/'specification.json').read_text())==spec
 baseline=OUT/'baseline_comparison'
 if not baseline.exists():cmp.main(baseline)
 counts=pd.read_csv(baseline/'within_context_counts.csv')
 assert counts[['matching_higher','mirroring_higher']].values.tolist()==[[213,17],[33,24],[85,69]]
 meta=pd.read_csv(INPUT/'model_metadata.csv');assert len(meta)==23364 and meta.participant_id.nunique()==30 and meta.dyad.nunique()==15
 with np.load(INPUT/'model_power.npz') as z: original=z['raw_power'].copy();features=z['features'].tolist()
 Y=original.copy()
 for pid in meta.participant_id.unique():
  pos=np.flatnonzero(meta.participant_id.eq(pid));rows=meta.iloc[pos].power_row.astype(int).to_numpy()
  with h5py.File(ROOT/'participants'/f'{pid}.h5','r') as z:
   assert z.attrs['complete']
   np.testing.assert_array_equal(z['events'][:][rows,0],meta.iloc[pos].event_sample.to_numpy())
   np.testing.assert_array_equal(z['selection'][:][rows],meta.iloc[pos].selection.to_numpy())
   Y[pos]=z['raw_power'][:].reshape(-1,408)[rows]
 inp=OUT/'inputs';inp.mkdir(exist_ok=True)
 if not (inp/'model_power.npz').exists():
  shutil.copyfile(INPUT/'model_metadata.csv',inp/'model_metadata.csv');np.savez_compressed(inp/'model_power.npz',raw_power=Y,features=np.array(features))
 assert sha(inp/'model_metadata.csv')==sha(INPUT/'model_metadata.csv')
 X=np.column_stack([np.ones(len(meta)),meta.role_code,meta.cond_code,meta.role_code*meta.cond_code,meta.rt_z])
 pilot=[]
 for idx in [0,68,136,340]:
  y=np.log1p(original[:,idx].astype('f8'));mean=float(y.mean());unit=float(y.std())
  full,_=fit.NestedProfile(X,(y-mean)/unit,meta).fit(unit,mean,verify=True)
  old=json.loads((cmp.NORM/'checkpoints'/f'{idx:03d}.json').read_text())['full']
  np.testing.assert_allclose(full['coef'],old['coef'],rtol=1e-6,atol=1e-8)
  np.testing.assert_allclose(full['se'],old['se'],rtol=1e-5,atol=1e-8)
  assert full['valid'];pilot.append(dict(feature=features[idx],max_coef_error=float(np.max(np.abs(np.array(full['coef'])-old['coef']))),verification=full['verification']))
 write(OUT/'baseline_refit_pilot.json',pilot)
 folder=OUT/'normative_fits';(folder/'checkpoints').mkdir(parents=True,exist_ok=True)
 records=[]
 for idx in range(408):
  checkpoint=folder/'checkpoints'/f'{idx:03d}.json'
  if checkpoint.exists():full=json.loads(checkpoint.read_text())['full']
  else:
   y=np.log1p(Y[:,idx].astype('f8'));mean=float(y.mean());unit=float(y.std())
   full,_=fit.NestedProfile(X,(y-mean)/unit,meta).fit(unit,mean,verify=idx in [0,68,136,340])
   assert full['valid'],(idx,full)
   write(checkpoint,dict(feature=features[idx],full=full))
  assert full['valid']
  records.append(dict(feature=features[idx],condition_coef=full['coef'][2],condition_se=full['se'][2],full_valid=full['valid'],llf=full['llf']))
  if (idx+1)%34==0:status('normative_fitting',completed=idx+1,total=408)
 pd.DataFrame(records).to_csv(folder/'model_results.csv',index=False)
 cmp.NORM=folder;cmp.main(OUT/'sensitivity_comparison')
 summaries={}
 for name,key in [('within_context_condition_effects','context'),('direct_condition_differences','comparison')]:
  a=pd.read_csv(baseline/f'{name}.csv');b=pd.read_csv(OUT/'sensitivity_comparison'/f'{name}.csv')
  m=a.merge(b,on=['feature',key],suffixes=('_original','_sensitivity'),validate='one_to_one')
  m.to_csv(OUT/f'{name}_comparison.csv',index=False)
  stats=[]
  for group,g in m.groupby(key,sort=False):
   old=g.significant_q05_original;new=g.significant_q05_sensitivity
   stats.append(dict(group=group,original_significant=int(old.sum()),sensitivity_significant=int(new.sum()),retained=int((old&new).sum()),lost=int((old&~new).sum()),gained=int((~old&new).sum()),point_direction_changes=int(((g.estimate_original>0)!=(g.estimate_sensitivity>0)).sum()),estimate_correlation=float(g.estimate_original.corr(g.estimate_sensitivity)),max_estimate_difference=float((g.estimate_original-g.estimate_sensitivity).abs().max())))
   if group in ['mixed_dyad_HC','PT','PT-minus-mixed_dyad_HC']:
    for col in ['estimate','se','p_wald']:np.testing.assert_array_equal(g[col+'_original'],g[col+'_sensitivity'])
  summaries[name]=stats
 assert all(sha(Path(p))==h for p,h in fingerprints.items())
 write(OUT/'comparison_summary.json',dict(status='completed',baseline_table3_reproduced=True,full_models_valid=408,clinical_estimates_se_unadjusted_p_unchanged=True,original_sources_unchanged=True,comparisons=summaries))
 status('complete')
if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():status('failed',error=repr(e))
  traceback.print_exc();raise
