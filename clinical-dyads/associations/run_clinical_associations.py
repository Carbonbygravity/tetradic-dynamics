from pathlib import Path
import os,sys,json,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd,scipy
from scipy import stats
O=Path(os.environ['TETRADIC_ASSOC_OUTPUT']).resolve()
F=Path(os.environ['TETRADIC_FROZEN_INPUTS']).resolve()
B=Path(os.environ['TETRADIC_COMPONENT_INPUTS']).resolve()
O.mkdir(parents=True, exist_ok=False)
import association_core as c
from evaluate import evaluate
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def save(d,name):d.to_csv(O/name,index=False,float_format='%.17g')
spec=json.loads((F/'frozen_specification.json').read_text())
assert np.__version__==spec['runtime']['numpy'] and scipy.__version__==spec['runtime']['scipy']
for name in ['features.csv','clinical_scores.csv','association_core.py','test_inventory.csv','run_associations.py']:
 assert sha(F/name)==spec['frozen_hashes'][name]
source=Path(os.environ['TETRADIC_SCORE_WORKBOOK']).resolve()
sourcehash=sha(source)
d=pd.read_excel(source,sheet_name='Demographics_Emmanuel_CLEANED');pt=d[d.group.eq('PT')].set_index('participant_id')
reference_scores=read(F/'clinical_scores.csv').set_index('patient_id');x=read(F/'features.csv').set_index('patient_id')
assert pt.index.is_unique and len(pt)==15 and set(pt.index)==set(x.index)==set(reference_scores.index) and x.index.equals(reference_scores.index)
y=pt.loc[x.index,reference_scores.columns].apply(pd.to_numeric);assert np.isfinite(y).all().all()
items=['panss_'+k for k in 'p1 p2 p3 n1 n4 n6 g2 g5 g6 g9'.split()]
assert pt[items].apply(pd.to_numeric).sum(axis=1,min_count=10).eq(pt.panss_total).all()
save(y.reset_index(),'clinical_scores.csv')
inv=read(F/'test_inventory.csv');inv['n']=15;inv['included_dyads']=';'.join(map(str,range(1,16)))
save(inv,'test_inventory.csv')
rows=evaluate(x.to_numpy(),y.to_numpy(),list(x),list(y),inv.to_dict('records'),spec['permutations'],spec['bootstrap_draws'])
meta={r['feature']:r for r in json.loads((F/'feature_metadata.json').read_text())}
for r in rows:
 for k,v in meta[r['feature']].items():r.setdefault(k,v)
main=pd.DataFrame([r for r in rows if r['family']=='non_interaction']);inter=pd.DataFrame([r for r in rows if r['family']=='genuine_interaction'])
components=read(B/'component_inputs.csv').set_index('patient_id').loc[x.index];fs=['patient_Leader_mean_log_RT','partner_Leader_mean_log_RT'];scores=['cog_dsst_written_score','cog_dsst_oral_score']
assert sha(B/'component_inputs.csv')==json.loads((B/'specification.json').read_text())['inputs_sha256']
assert np.allclose(components[fs[0]]-components[fs[1]],x.beh_leader_slowing_vs_partner_dlog,rtol=0,atol=1e-12)
xx=components[fs].to_numpy();yy=y[scores].to_numpy();rho,pv,counts=c.permutation_counts(xx,yy,999999,c.generator(3,32767));cr=[]
for i,feature in enumerate(fs):
 for j,outcome in enumerate(scores):
  ci=c.bca_interval(xx[:,i],yy[:,j],20000,c.generator(4,0));loo=c.leave_one_out(xx[:,i],yy[:,j]);cr.append(dict(test_id=feature+'|'+outcome,feature=feature,outcome=outcome,n=15,rho=rho[i,j],permutation_p=pv[i,j],ci_low=ci['low'],ci_high=ci['high'],ci_status=ci['status'],loo_min=loo.min(),loo_max=loo.max(),permutation_extreme_count=counts[i,j]))
comp=pd.DataFrame(cr);ext=pd.concat([main,comp],ignore_index=True);ext['extended_BY_p']=c.by_adjust(ext.permutation_p);comp['extended_BY_p']=ext.extended_BY_p.tail(4).to_numpy()
assert (len(main),len(inter),len(comp),len(ext))==(305,35,4,309)
families = [
    ('main', main, F/'Results/main_associations.csv'),
    ('interaction', inter, F/'Results/genuine_interaction_associations.csv'),
    ('component', comp, B/'component_results.csv'),
]
checked_tests = 0
for family, estimates, reference_path in families:
    save(estimates, family + '_associations.csv')
    reference = read(reference_path).set_index('test_id')
    assert reference.index.is_unique and estimates.test_id.is_unique
    assert set(reference.index) == set(estimates.test_id)

    for _, row in estimates.iterrows():
        features = components if family == 'component' else x
        rho = stats.spearmanr(features[row.feature], y[row.outcome]).statistic
        assert abs(rho - row.rho) < 1e-12
        assert row.ci_status == 'available' and np.isfinite([row.ci_low, row.ci_high]).all()
        checked_tests += 1

    scale_rows = estimates.loc[estimates.outcome.isin(['panss_total', 'sofas_total_score'])]
    columns = ['rho', 'permutation_p', 'ci_low', 'ci_high']
    if not scale_rows.empty:
        assert np.allclose(reference.loc[scale_rows.test_id, columns].to_numpy(dtype=float),
                       scale_rows[columns].to_numpy(dtype=float), atol=1e-12, rtol=0)
    adjusted = 'extended_BY_p' if family == 'component' else 'by_adjusted_p'
    print(family, 'nominal:', (estimates.permutation_p <= .05).sum(),
          'minimum BY-adjusted p:', estimates[adjusted].min(), flush=True)

save(ext[['test_id','permutation_p','extended_BY_p']],'extended_309.csv')
assert sha(source)==sourcehash
(O/'provenance.json').write_text(json.dumps({'clinical_workbook':str(source),'sha256':sourcehash,'features_sha256':sha(F/'features.csv'),'component_inputs_sha256':sha(B/'component_inputs.csv'),'core_sha256':sha(F/'association_core.py'),'permutations':999999,'paired_BCa_draws':20000,'families':[305,35,309],'n':15,'unique_tests':checked_tests,'rhos_independently_checked':checked_tests},indent=2))
