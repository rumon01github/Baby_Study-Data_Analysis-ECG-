"""Step 7e. Guard-parameter grid (clip, tau, dead time) chosen on odd-numbered infants, checked on even."""
import numpy as np, pandas as pd, json, sys, itertools
from common import *
import neotex_qrs_v2 as v2
P=json.load(open('neotex_qrs_params.json')); mb=pd.read_csv(WORK + 'manual_beats_clean.csv')
ss=pd.read_csv(WORK + 'strips_gate_vs_human.csv')[['Trial','start_s','quality','n_human']]
G=list(itertools.product([2.0,4.0,8.0],[0.5,1.0,2.0],[1.0,1.5,2.0]))
out=[]
sig={tr:belt_csv(tr) for tr in ss.Trial.unique()}
for clip,tau,dead in G:
    for tr,g in ss.groupby('Trial'):
        b=sig[tr]; tt=b.time_s.values; d=tt[v2.detect(b.ECG.values,100,clip=clip,tau_s=tau,dead_s=dead,**P)]
        hm=mb[mb.Trial==tr]
        for _,s in g.iterrows():
            a=s.start_s; h=np.sort(hm.loc[hm.strip_start_s==a,'peak_time_s'].values); dd=d[(d>=a)&(d<a+8)]
            rq=v2.rr_quality((dd*100).astype(int),100)['ok'] if len(dd)>=4 else False
            out.append((clip,tau,dead,tr,a,s.quality,f1(dd,h,0.075) if len(h) else np.nan,rq))
O=pd.DataFrame(out,columns=['clip','tau','dead','Trial','start_s','quality','F1','ok']); O['P']=O.Trial.str[:3]; O['fold']=np.where(O.P.str[-1].astype(int)%2==1,'odd','even')
O.to_csv(WORK + 'grid_v2.csv',index=False)
for fold in ['odd','even']:
    r=O[(O.quality=='readable')&(O.fold==fold)]; u=O[(O.quality=='unreadable')&(O.fold==fold)]
    t=r.groupby(['clip','tau','dead']).F1.mean().round(4).rename('F1').to_frame(); t['FP_unread']=u.groupby(['clip','tau','dead']).ok.sum()
    print(fold); print(t.sort_values('F1',ascending=False).head(8).to_string())
