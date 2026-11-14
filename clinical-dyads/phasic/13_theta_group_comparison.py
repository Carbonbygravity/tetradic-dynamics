"""Approved descriptive theta comparison, steps 1-3; no new inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,html,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(os.environ['TETRADIC_PHASIC_ROOT']).resolve()
NAME='Theta_Group_Comparison_Run_01'
R=P/'03_Results'/NAME;F=P/'04_Figures'/NAME;Q=P/'05_Quality_Checks'/NAME;D=P/'06_Manuscript_Text'/NAME
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 for p in (R,F,Q,D):assert not p.exists(),f'Refuse overwrite {p}'
 base=P/'03_Results/Historical_Participant_Replication_Run_01';s=read(base/'frozen_specification.json')
 assert sha(base/'frozen_specification.json')=='36e81485219dce84cd66da68276273ec8bfe8ea5c7f9473cabe3390314df5d69'
 rois=s['inputs']['clinical']['rois'];times=np.array(s['inputs']['clinical']['times'])
 stems=['caudalmiddlefrontal','rostralmiddlefrontal','superiorfrontal','frontalpole','medialorbitofrontal','lateralorbitofrontal','parsopercularis','parsorbitalis','parstriangularis','caudalanteriorcingulate','rostralanteriorcingulate','posteriorcingulate','isthmuscingulate']
 regions={'All_68_ROIs':rois,'Frontal_Cingulate_26_ROIs':[x for x in rois if x.rsplit('-',1)[0] in stems]}
 assert len(regions['Frontal_Cingulate_26_ROIs'])==26
 inputs={};data={};metadata={};checks=[]
 for cohort in ('normative','clinical'):
  m=s['inputs'][cohort];path=Path(m['path']);assert sha(path)==m['sha256'];inputs[cohort]=dict(path=str(path),sha256=m['sha256'])
  with h5py.File(path) as f:
   assert f['cell_order'].asstr()[:].tolist()==['LMA','LMR','FMA','FMR'];assert f['roi_labels'].asstr()[:].tolist()==rois
   np.testing.assert_allclose(f['times_seconds'][:],times)
   bi=f['band_labels'].asstr()[:].tolist().index('theta');z=f['tensor'][:,:,:,bi,:]
   subjects=f['subjects'].asstr()[:].tolist();dyads=f['dyads'][:];groups=f['groups'].asstr()[:] if cohort=='clinical' else np.array(['NormHC']*30)
   assert subjects==m['subjects'] and groups.tolist()==m['groups']
   for g in (['NormHC'] if cohort=='normative' else ['PT','HC']):
    ix=np.flatnonzero(groups==g);metadata[g]=dict(subjects=[subjects[i] for i in ix],dyads=dyads[ix].tolist(),n=len(ix))
    data[g]={key:z[ix][:,:,[rois.index(x) for x in rr],:].mean(axis=2) for key,rr in regions.items()}
   assert len(set(dyads))==15
 assert metadata['PT']['dyads']==metadata['HC']['dyads']
 assert all(metadata['NormHC']['dyads'].count(d)==2 for d in set(metadata['NormHC']['dyads']))
 checks.append('Input hashes, four-cell order, ROI/time axes, participant identities and 15 dyads per cohort verified')
 # Consolidate all theta ST maps, not only significant clusters.
 names=['Historical_Participant_Replication_Run_01','Paired_PT_HC_Anatomical_Run_01','Paired_PT_HC_Historical_Sensitivity_Run_01','Cross_Cohort_Anatomical_Run_01','Cross_Cohort_Historical_Sensitivity_Run_01']
 maps=[];sources=[]
 for name in names:
  root=P/'03_Results'/name;v=read(P/'05_Quality_Checks'/name/'validation.json');assert v['passed']
  sh=sha(root/'frozen_specification.json');assert v['specification_sha256']==sh
  sources.append(dict(path=str(root/'cluster_summary.json'),sha256=sha(root/'cluster_summary.json'),specification_sha256=sh))
  for a in read(root/'cluster_summary.json')['results']:
   if not a['analysis'].startswith('spatiotemporal_theta_'):continue
   if a.get('group')=='ClinicalParticipants':continue
   subgroup=a.get('group',a.get('comparison',a.get('cohort')))
   path=Path(a['path']) if 'path' in a else root/subgroup/(a['analysis']+'.pkl')
   assert sha(path)==a['sha256'] and a['specification_sha256']==sh
   row=dict(a);row.update(source_run=name,label=subgroup,result_path=str(path));maps.append(row)
 assert len(maps)==30,len(maps)
 checks.append('All 30 source theta-map checksums and branch validation/specification hashes verified')
 # Cross-cohort branches inherit a queue specification; verify its hash and input identities.
 queue=P/'03_Results/Cross_Cohort_Cluster_Queue_Run_01/frozen_specification.json'
 for name in names[-2:]:assert read(P/'03_Results'/name/'frozen_specification.json')['queue_specification_sha256']==sha(queue)
 sources.append(dict(path=str(queue),sha256=sha(queue)))
 for p in (R,F,Q,D):p.mkdir(parents=True)
 spec=dict(name=NAME,scope='Approved steps 1-3 only; descriptive curves and existing inference. No new statistical tests.',inputs=inputs,regions=regions,
  region_note='All-ROI and anatomically named frontal/cingulate views chosen for this exploratory follow-up; no cluster selection or ROI optimization. Not preregistered independent confirmation.',
  cell_order=['LMA','LMR','FMA','FMR'],theta_hz=[4,5,6,7],window_seconds=[-1,1],units='Mean log1p source power; equal ROI and participant weights',
  participants=metadata,source_results=sources,script_sha256=sha(__file__),versions=dict(numpy=np.__version__,h5py=h5py.__version__,matplotlib=matplotlib.__version__),
  inference_note='Within-group historical-method tests use participant rows. Paired clinical tests use 15 PT-minus-HC dyad contrasts. Cross-cohort tests use 15 normative partner means and 15 clinical participants per group, null-centred whole-dyad bootstrap Welch, not permutations. No new averaging replaces the saved participant analyses.',
  interpretation='Accommodation and self/other monitoring are hypotheses, not assigned labels or tested mechanisms. RT and tonic analyses are completed study context. Source-scale/cohort confounding remains. Cluster extent is not precise pointwise onset; full-window Morlet edge effects remain.')
 write(R/'frozen_specification.json',spec);sh=sha(R/'frozen_specification.json')
 np.savez_compressed(R/'theta_curves.npz',times_seconds=times,**{g+'__'+k:a for g,d in data.items() for k,a in d.items()})
 write(R/'identity_mapping.json',metadata);write(R/'existing_theta_results.json',dict(specification_sha256=sh,maps=maps))
 figures=[];descriptive={}
 colors={'NormHC':'#333333','PT':'#c43f42','HC':'#2166ac'}
 def save(fig,name,caption):
  fig.savefig(F/(name+'.png'),dpi=150);fig.savefig(F/(name+'.pdf'));plt.close(fig)
  figures.append(dict(name=name,caption=caption))
 def role(a):return (a[:,0]+a[:,1]-a[:,2]-a[:,3])/2
 def style(ax):ax.axvline(0,color='.45',ls=':',lw=.9);ax.grid(alpha=.16);ax.set_xlim(-1,1)
 for region in regions:
  # Same axes across group rows, within each column.
  fig,axs=plt.subplots(3,3,figsize=(15,10),sharex=True,sharey='col')
  for i,g in enumerate(['NormHC','PT','HC']):
   a=data[g][region];mean=a.mean(0)
   for y,label,col in [((mean[0]+mean[1])/2,'Leader','#7843a8'),((mean[2]+mean[3])/2,'Follower','#148575')]:axs[i,0].plot(times,y,label=label,color=col)
   for j,label in enumerate(['Leader Matching','Leader Mirroring','Follower Matching','Follower Mirroring']):axs[i,1].plot(times,mean[j],label=label,color=['#7843a8','#7843a8','#148575','#148575'][j],ls=['-','--','-','--'][j])
   for y,label,col,ls in [(mean[0]-mean[2],'Matching','#c43f42','-'),(mean[1]-mean[3],'Mirroring','#2166ac','-'),(role(a).mean(0),'Across conditions','#222222','--')]:
    axs[i,2].plot(times,y,label=label,color=col,ls=ls)
   axs[i,2].axhline(0,color='.6',lw=.8)
   axs[i,0].set_ylabel(g+'\nMean log1p power')
   for ax in axs[i]:style(ax)
  for ax,title in zip(axs[0],['Leader and Follower means','Four condition/role cells','Leader minus Follower']):ax.set_title(title);ax.legend(fontsize=8)
  for ax in axs[-1]:ax.set_xlabel('Time from respective response event (s)')
  fig.suptitle(region.replace('_',' ')+' | corrected inputs | participant means, descriptive',fontsize=15);fig.tight_layout(rect=[0,0,1,.95]);save(fig,region+'_group_curves','Same ROI set and column-wise y limits across groups. All participants retained; no significance inferred from curve separation.')
  fig,axs=plt.subplots(1,4,figsize=(17,9),sharex=True)
  roledata=[role(data[g][region]) for g in ['NormHC','PT','HC']];delta=roledata[1]-roledata[2];m=max(np.max(abs(a)) for a in roledata+[delta])
  for ax,a,g in zip(axs,roledata+[delta],['NormHC','PT','HC','PT minus HC, paired']):
   im=ax.imshow(a,aspect='auto',extent=[-1,1,len(a)-.5,-.5],cmap='RdBu_r',vmin=-m,vmax=m,interpolation='nearest')
   ids=metadata[g]['subjects'] if g in metadata else ['dyad '+str(d) for d in metadata['PT']['dyads']]
   ax.set_yticks(range(len(a)),ids,fontsize=7);ax.set_title(g);ax.axvline(0,color='k',ls=':',lw=.8);ax.set_xlabel('Time (s)')
  fig.suptitle(region.replace('_',' ')+' | individual Leader-minus-Follower curves');fig.tight_layout(rect=[0,0,.94,.95]);cax=fig.add_axes([.95,.15,.012,.65]);fig.colorbar(im,cax=cax,label='Log1p contrast (red positive, blue negative)');save(fig,region+'_individual_heatmaps','Every participant shown; clinical difference pairs the patient with their own HC partner. Normative rows retain both partner identities. One shared color scale.')
  for cohort in ['clinical','normative']:
   fig,axs=plt.subplots(5,3,figsize=(15,14),sharex=True,sharey=True)
   ds=metadata['PT']['dyads'] if cohort=='clinical' else sorted(set(metadata['NormHC']['dyads']))
   for ax,d in zip(axs.flat,ds):
    if cohort=='clinical':
     i=metadata['PT']['dyads'].index(d);curves=[(roledata[1][i],'PT','#c43f42'),(roledata[2][i],'HC','#2166ac')]
    else:
     ix=[i for i,x in enumerate(metadata['NormHC']['dyads']) if x==d];curves=[(roledata[0][i],metadata['NormHC']['subjects'][i],col) for i,col in zip(ix,['#7843a8','#148575'])]
    for y,label,col in curves:ax.plot(times,y,label=label,color=col,lw=1)
    style(ax);ax.axhline(0,color='.6',lw=.7);ax.set_title('Dyad '+str(d),fontsize=9);ax.legend(fontsize=6)
   for ax in axs[-1]:ax.set_xlabel('Time (s)')
   fig.suptitle(region.replace('_',' ')+' | '+cohort+' dyads | Leader minus Follower');fig.tight_layout(rect=[0,0,1,.96]);save(fig,region+'_'+cohort+'_dyads','Individual curves with common limits within this figure; no new tests or exclusions.')
  fig,axs=plt.subplots(1,4,figsize=(17,4),sharex=True,sharey=True);ds={g:np.array(metadata[g]['dyads']) for g in metadata};arrays={g:a for g,a in zip(['NormHC','PT','HC'],roledata)};arrays['PT_minus_HC']=delta;ds['PT_minus_HC']=ds['PT']
  descriptive[region]={}
  for ax,(g,a) in zip(axs,arrays.items()):
   mean=a.mean(0);loo=np.stack([a[ds[g]!=d].mean(0) for d in sorted(set(ds[g]))]);lo=loo.min(0);hi=loo.max(0)
   ax.fill_between(times,lo,hi,color='.8',label='Leave-one-dyad-out range');ax.plot(times,mean,color=colors.get(g,'#222222'),label='Full mean');ax.axhline(0,color='.4',lw=.8);style(ax);ax.set_title(g);ax.set_xlabel('Time (s)')
   descriptive[region][g]=dict(max_absolute_leave_one_dyad_out_shift=float(np.max(abs(loo-mean))),full_mean_max_absolute=float(np.max(abs(mean))))
  axs[0].legend(fontsize=7);fig.suptitle(region.replace('_',' ')+' | sensitivity to omitting one dyad (not confidence intervals)');fig.tight_layout(rect=[0,0,1,.92]);save(fig,region+'_dyad_sensitivity','Each omission removes both normative partners or the paired clinical dyad. Shading is an omission range, not a statistical confidence interval.')
 write(R/'descriptive_sensitivity.json',descriptive)
 # Read-back equality verifies saved plotting inputs.
 with np.load(R/'theta_curves.npz') as f:
  for g,d in data.items():
   for k,a in d.items():np.testing.assert_array_equal(f[g+'__'+k],a)
 checks.append('Saved curve readback exact; all individual participants retained; leave-one-out omissions are whole dyads')
 esc=html.escape
 parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Theta group comparison</title><style>body{font:16px system-ui;max-width:1300px;margin:35px auto;padding:0 20px;line-height:1.5}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px}img{max-width:100%}summary{cursor:pointer}</style><h1>Theta group comparison â€” approved steps 1â€“3</h1>', '<p>'+esc(spec['scope'])+'</p><p>'+esc(spec['inference_note'])+'</p><p>'+esc(spec['interpretation'])+'</p>',
 '<p>All figures use corrected inputs and theta 4â€“7 Hz. NormHC: 30 participants in 15 dyads; PT: 15; clinical HC: 15. Figures are descriptive, with no confidence bands or new p values. Original-input benchmark results appear only in the existing-test inventory.</p><h2>Common regions</h2><p>'+esc(spec['region_note'])+'</p>']
 for k,rr in regions.items():parts.append('<details><summary>'+k+' ('+str(len(rr))+')</summary><p>'+esc(', '.join(rr))+'</p></details>')
 parts.append('<h2>Existing theta spatiotemporal tests</h2><p>Min p is the smallest cluster p in each map, not an additional omnibus test. Full candidate lists, including nonsignificant clusters, follow. The original multiple-map families remain: within-group/paired Ã—18; cross-cohort Ã—36.</p><table><tr><th>Run / group</th><th>Contrast</th><th>Minimum within-map p</th><th>Clusters p&lt;.05</th><th>Clusters across-map p&lt;.05</th></tr>')
 for a in maps:
  cs=a['clusters'];parts.append('<tr><td>'+esc(a['source_run']+' / '+a['label'])+'</td><td>'+a['analysis'].split('_')[-1]+'</td><td>'+str(min([c['p_within_map'] for c in cs],default=1))+'</td><td>'+str(sum(c['p_within_map']<.05 for c in cs))+'</td><td>'+str(sum(c['p_across_maps_bonferroni']<.05 for c in cs))+'</td></tr>')
 parts.append('</table><h2>Figures</h2>')
 for f in figures:
  rel='../../04_Figures/'+NAME+'/'+f['name'];parts.append('<h3>'+f['name'].replace('_',' ')+'</h3><p>'+esc(f['caption'])+'</p><a href="'+rel+'.pdf">PDF</a><img src="'+rel+'.png" alt="'+f['name']+'">')
 parts.append('<h2>All source cluster records</h2>')
 for a in maps:parts.append('<details><summary>'+esc(a['source_run']+' / '+a['label']+' / '+a['analysis'])+'</summary><pre>'+esc(json.dumps(a['clusters'],indent=2))+'</pre></details>')
 parts.append('<h2>Reproduction and audit</h2><ul>')
 for label,path in [('Script',Path(__file__).resolve()),('Frozen settings',R/'frozen_specification.json'),('Saved curves',R/'theta_curves.npz'),('Identity mapping',R/'identity_mapping.json'),('Validation',Q/'validation.json')]:parts.append('<li><a href="'+path.as_uri()+'">'+label+'</a></li>')
 parts.append('</ul></html>');(D/'theta_comparison_report.html').write_text(''.join(parts),encoding='utf-8')
 write(F/'figure_manifest.json',dict(figures=figures,specification_sha256=sh))
 outputs=[dict(path=str(f),sha256=sha(f)) for folder in [R,F,D] for f in folder.iterdir() if f.is_file()]
 write(Q/'validation.json',dict(passed=True,specification_sha256=sh,checks=checks,source_maps_verified=len(maps),figure_count=len(figures),outputs=outputs))
 write(D/'report_provenance.json',dict(specification_sha256=sh,sources=sources,outputs=outputs,validation_sha256=sha(Q/'validation.json')))
 write(R/'analysis_complete.json',dict(passed=True,stage='descriptive_steps_1_to_3_complete',new_inferential_tests=0,specification_sha256=sh,utc=datetime.now(timezone.utc).isoformat()))
 print(json.dumps(dict(report=str(D/'theta_comparison_report.html'),figures=len(figures),maps=len(maps),sensitivity=descriptive),indent=2))
if __name__=='__main__':main()
