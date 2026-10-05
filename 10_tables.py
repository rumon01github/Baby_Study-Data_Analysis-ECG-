"""Step 10. Paper tables: generic vs belt detectors vs expert; bSQI threshold sensitivity; per-infant summary."""
import numpy as np, pandas as pd, pickle, sys
from common import *
d=cache(); mb=pd.read_csv(WORK + 'manual_beats_clean.csv')
R=pd.read_csv(WORK + 'handoff_eval.csv'); M=pd.read_csv(WORK + 's3_strip_results.csv')[['Trial','start_s','F1_sc']]
R=R.merge(M,on=['Trial','start_s'],how='left'); R['P']=R.Trial.str[:3]
# generic detectors (NeuroKit, Hamilton) on belt from cache (42 trials)
rows=[]
for tr,g in R.groupby('Trial'):
    if tr not in d: continue
    dev=d[tr]['belt']; fs=dev['fs']; t0=dev['t0']
    det={k:dev[k]/fs+t0 for k in ('nk','ham')}
    hm=mb[mb.Trial==tr]
    for _,s in g.iterrows():
        a=s.start_s; h=np.sort(hm.loc[hm.strip_start_s==a,'peak_time_s'].values)
        if len(h)==0: continue
        r=dict(Trial=tr,start_s=a)
        for k in det: x=det[k][(det[k]>=a)&(det[k]<a+8)]; r['F1_'+k]=f1(x,h,0.075)
        rows.append(r)
G=pd.DataFrame(rows); R=R.merge(G,on=['Trial','start_s'],how='left')
R.to_csv(WORK + 'detector_table_strips.csv',index=False)
cols=['F1_nk','F1_ham','F1_v1','F1_v2','F1_sc']
print('BIOPAC-paired trials only (42), strips with human beats:')
for q in ['readable','partly_readable','uncertain']:
    s=R[(R.quality==q)&R.F1_nk.notna()]; print(q,len(s),s[cols].mean().round(3).to_dict())
s=R[(R.quality=='readable')&R.F1_nk.notna()]
print('\nper infant readable:'); print(s.groupby('P')[cols].mean().round(3).to_string())
print('\nstrips F1<0.9 readable:', {c:int((s[c]<0.9).sum()) for c in cols})
# unreadable false detections per strip
u=R[(R.quality=='unreadable')]; print('unreadable n per strip: v1 %.1f v2 %.1f'%(u.n_v1.mean(),u.n_v2.mean()))
# --- bSQI threshold sensitivity (clean BIOPAC windows, in comparison)
q=pd.read_csv(WORK + 'quadrants.csv'); c=q[q.In_comparison & q.BIOPAC_good].copy(); c['err']=(c.Belt_HR-c.BIOPAC_HR).abs()
print('\nbSQI threshold sensitivity (n clean BIOPAC windows =',len(c),')')
rows=[]
for th in [0.5,0.6,0.7,0.8,0.9,0.95,1.0]:
    g=(c.Belt_bSQI>=th)&(c.Belt_HR.between(80,220))&(c.Belt_RRcv<0.25)
    allq=q[q.In_comparison]; passall=((allq.Belt_bSQI>=th)&(allq.Belt_HR.between(80,220))&(allq.Belt_RRcv<0.25)).mean()*100
    rows.append(dict(bSQI=th,belt_pass_pct=round(passall,1),pass_MAE=round(c[g].err.mean(),2),pass_within5=round(100*(c[g].err<=5).mean(),1),fail_MAE=round(c[~g].err.mean(),2),fail_within5=round(100*(c[~g].err<=5).mean(),1),n_pass=int(g.sum())))
T=pd.DataFrame(rows); print(T.to_string()); T.to_csv(WORK + 'tbl_bsqi_sens.csv',index=False)
# --- per infant table
q2=q[q.In_comparison]; tt=pd.read_csv(WORK + 'trial_table_v2.csv'); print(tt.columns.tolist())
rows=[]
for p,g in q2.groupby('Participant'):
    bb=g[g.Belt_good&g.BIOPAC_good]; e=(bb.Belt_HR-bb.BIOPAC_HR).abs()
    rows.append(dict(Infant=p,trials_compared=g.Trial.nunique(),windows=len(g),contact_good_trials=g.groupby('Trial').Belt_contact_class.first().eq('good').sum(),
        belt_reliable=round(100*g.Belt_good.mean(),1),biopac_reliable=round(100*g.BIOPAC_good.mean(),1),both=len(bb),MAE=round(e.mean(),2) if len(bb) else np.nan,within5=round(100*(e<=5).mean(),1) if len(bb) else np.nan,moving_pct=round(g['Moving_%'].mean(),1)))
PI=pd.DataFrame(rows); print(PI.to_string()); PI.to_csv(WORK + 'tbl_per_infant.csv',index=False)
