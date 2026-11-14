import json,csv,html,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import os
W=Path(os.environ['TETRADIC_RESULTS_ROOT']).resolve()
O=Path(os.environ['TETRADIC_TABLE_OUTPUT']).resolve()
O.mkdir(parents=True,exist_ok=False)
sources={}
def read(p):
 p=Path(p);raw=p.read_bytes();sources[str(p.resolve())]=hashlib.sha256(raw).hexdigest();return json.loads(raw.decode('utf-8-sig'))
def f(v):return 'NT' if v is None else f'{v:.4f}'
def window(a,b):return f'{a} to {b}'
def uid(t,g):return t.get('uid',f'clinical:{g}:{t["id"]}')
def meta(t,g):return [uid(t,g),t['atom'],t['contrast'].replace('_',' '),t['geometry'],t['roi']+' / '+t.get('follower_roi',t['roi']),'-'.join(map(str,t['band'])),window(t['start_ms'],t['end_ms'])]
base=['Target ID','Atom','Contrast','Geometry','ROI pair (Leader / Follower)','Band (Hz)','Primary window (ms)']
tables=[]
def table(code,title,headers,rows,note):
 assert all(len(x)==len(headers) for x in rows),code
 with (O/(code+'.csv')).open('w',newline='',encoding='utf-8-sig') as h:
  wr=csv.writer(h);wr.writerow(headers);wr.writerows(rows)
 tables.append(dict(code=code,title=title,headers=headers,rows=rows,note=note))
groups=[('main','Four main contrasts','four_contrast'),('interaction','Genuine response-period × condition interaction','genuine_interaction')]
primary={}
for g,_,stem in groups:
 folder='PhiID_Clinical_Controls_2026-10-27' if g=='main' else 'PhiID_Clinical_Interaction_Overlap_Controls_2026-10-28'
 primary[g]=[x for x in read(W/folder/'directional_tiers.json') if x['tier'] in ['A','B']]
 rows=[meta(t,g)+[t['tier'],t['trial']['direction'],f(t['trial']['group_p']),t['trial']['n_pass'],f(t['time_shift']['group_p']) if t['time_shift'] else 'NT',t['time_shift']['n_pass'] if t['time_shift'] else 'NT'] for t in primary[g]]
 table('ST12'+('a' if g=='main' else 'b'),'Primary ΦID controls: '+groups[0 if g=='main' else 1][1],base+['Tier','Direction (+1 / −1)','Trial p','Dyads passing trial','Time-shift p','Dyads passing time shifts'],rows,'Tier A required trial-shuffle significance in at least four dyads and time-shift significance in at least three dyads. Tier B met the trial-shuffle criterion but did not meet the Tier A criterion. Dyad-level significance was defined as p ≤ .05. Time shifts were tested when the group trial-shuffle p value was ≤ .10; NT denotes not tested. Direction follows the signed parent contrast. ROI pairs follow Leader/Follower order. Leader and Follower designate task roles. Homologous and cross-hemispheric pairings follow the tetradic geometry defined in the main text.')
 R=W/'PhiID_Clinical_RoleSwap_Preflight_2026-10-29'
 rr=read(R/(stem+'_results.json'))
 table('ST13'+('a' if g=='main' else 'b'),'Early and late chronological task halves: '+g,base+['Early mean','Late mean','Late − early','dz','95% bootstrap CI','Raw p','Holm p','Matched-trial change, 2.5–97.5 percentiles'],[meta(x['target'],g)+[f(x['early_mean']),f(x['late_mean']),f(x['change']),f(x['dz']),' to '.join(map(f,x['bootstrap_95'])),f(x['p']),f(x['holm_p']),' to '.join(map(f,np.percentile(x['subsample_mean_changes'],[2.5,97.5])))] for x in rr],'The paired change is late minus early. Confidence intervals use 10,000 dyad-bootstrap samples. Exact paired sign-flip p values were Holm-corrected across all 11 unique-information targets, comprising ten main targets and one interaction target. The matched-trial interval describes variation across 100 subsamples and is not a confidence interval.')
 R=W/'PhiID_Clinical_Tau_Preflight_2026-10-29';rr=read(R/(stem+'_tau_results.json'));wi={x['uid']:x for x in read(R/'window_inventory.json')}
 rows=[]
 for t in primary[g]:
  u=uid(t,g);v=wi[u];cells=[]
  for tau in [1,2,4,8,16,32]:
   x=next(x for x in rr if x['target']['uid']==u and x['tau']==tau)
   cells.append(f"{x['tier']} ({x['trial']['n_pass']}, {x['time']['n_pass'] if x['time'] else 'NT'})")
  rows.append(meta(t,g)+[t['tier'],window(v['common_start_ms'],v['common_end_ms'])]+cells)
 table('ST14'+('a' if g=='main' else 'b'),'Lag sensitivity: '+g,base+['Primary tier','Common window (ms)']+['τ = '+str(n) for n in [1,2,4,8,16,32]],rows,'Cells show sensitivity tier (dyads passing trial shuffles, dyads passing time shifts). NT denotes a time-shift test not performed under the group trial gate. At 1 kHz, τ is the offset in samples on either side of the centre; past–future separation is 2τ ms. All offsets use the listed common window. Primary tiers retain the original windows. Tier C denotes fewer than four trial-shuffle passing dyads.')
 R=W/'PhiID_Clinical_History_Preflight_2026-10-30';rr={x['target']['uid']:x for x in read(R/(stem+'_history_tiers.json'))};ei={x['uid']:x for x in read(R/'target_inventory.json')['eligibility']}
 rows=[]
 for t in primary[g]:
  u=uid(t,g);e=ei[u];x=rr.get(u)
  rows.append(meta(t,g)+[t['tier'],window(e['history_start_ms'],e['history_end_ms']) if e['history_eligible'] else 'Insufficient common support',f"H={x['observed_selection']['H']}, τ={x['observed_selection']['tau']}" if x else 'Not tested',x['history_sensitivity_tier'] if x else 'Excluded',f"{f(x['trial']['group_p'])} ({x['trial']['n_pass']})" if x else 'NT',f"{f(x['time_shift']['group_p'])} ({x['time_shift']['n_pass']})" if x and x['time_shift'] else 'NT'])
 table('ST15'+('a' if g=='main' else 'b'),'History sensitivity: '+g,base+['Primary tier','History window (ms)','Observed selection','History tier','Trial p (dyads)','Time-shift p (dyads)'],rows,'The selected history was determined independently for the observed data and each surrogate. The grid comprised H = 1, 2, 3 and τ = 1, 2, 4, 8, 16, 32, 64 samples. Trials were evaluated on common support. Targets with fewer than three common centres were excluded, rather than assigned Tier C. Primary classifications are unchanged.')
 R=W/'PhiID_Clinical_Diagnostics_Preflight_2026-10-29';rr=read(R/(stem+'_diagnostics.json'))
 def diag(m):return 'Not estimable' if m['median_ms'] is None else f"{m['median_ms']:g} [{m['q25_ms']:g}, {m['q75_ms']:g}]; n={m['n']}"
 table('ST16'+('a' if g=='main' else 'b'),'Signal-timescale diagnostics: '+g,base+['Actor','Span','ACF first zero (ms)','ACF first minimum (ms)','MI first minimum (ms)'],[meta(x['target'],g)+[x['actor'],x['span'].replace('_',' ')]+[diag(x['metrics'][k]) for k in ['acf_zero','acf_minimum','mi_minimum']] for x in rr],'Diagnostics were evaluated for primary Tier A targets. Values are medians [first quartile, third quartile] across dyads, following within-cell trial and across-cell aggregation. n is the number of dyads with an estimable landmark. Missing landmarks were retained as missing. Feature-window and full-epoch diagnostics are reported separately.')
