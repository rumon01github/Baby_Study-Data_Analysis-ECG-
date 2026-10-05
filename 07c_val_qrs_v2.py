"""Step 7c. Window-level validation of detector v2 (artifact guards) + run-time gate against clean BIOPAC windows."""
import numpy as np, pandas as pd, glob, json, pickle
from neotex_qrs_v2 import detect, rr_quality
from common import *
P=json.load(open('neotex_qrs_params_v2.json')); from common import *
q=pd.read_csv(WORK + 'quadrants.csv'); d=cache()
rows=[]
for tr,g in q.groupby('Trial'):
    df=belt_csv(tr)
    tt=df.time_s.values; r=detect(df.ECG.values,fs=100,**P); rt=tt[r]
    # second, cheaper detector for agreement (same pipeline, conservative threshold) -> app-side bSQI
    r2=detect(df.ECG.values,fs=100,lo=P['lo'],hi=P['hi'],W_ms=P['W_ms'],beta=0.5,searchback=False,clip=P['clip'],tau_s=P['tau_s'],dead_s=P['dead_s']); rt2=tt[r2]
    nk=d[tr]['belt']['nk']/d[tr]['belt']['fs']+d[tr]['belt']['t0']
    for _,w in g.iterrows():
        a,b=w.Start_s,w.Start_s+10; rw=rt[(rt>=a)&(rt<b)]; qq=rr_quality((rw*100).astype(int),100) if len(rw)>=4 else dict(hr=np.nan,rrcv=np.nan,ok=False)
        ag=f1(rw,rt2[(rt2>=a)&(rt2<b)],0.075); agnk=f1(rw,nk[(nk>=a)&(nk<b)],0.075)
        rows.append(dict(Trial=tr,Start_s=a,hr_new=qq['hr'],rrcv_new=qq['rrcv'],ok_rr=qq['ok'],agree2=ag,agree_nk=agnk))
R=q.merge(pd.DataFrame(rows),on=['Trial','Start_s']); R=R[R.In_comparison]
ref=R.BIOPAC_good    # clean BIOPAC under HR criterion
R['err_new']=(R.hr_new-R.BIOPAC_HR).abs(); R['err_nk']=(R.Belt_HR-R.BIOPAC_HR).abs()
print('Clean-BIOPAC windows:',ref.sum())
print('Agreement new detector vs NeuroKit (median F1 per window): %.3f'%R.agree_nk.median())
gate_app=(R.ok_rr)&(R.agree2>=0.9)
for name,gate in [('NeuroKit + bSQI gate (paper)',R.Belt_good),('New detector, RR gate only',R.ok_rr),('New detector, RR + 2-threshold agreement gate',gate_app)]:
    e=R.err_new if 'New' in name else R.err_nk
    a=R[ref&gate]; b=R[ref&~gate]
    print(f'{name:48s} pass {100*gate.mean():4.1f}% | pass: MAE {a[e.name].mean():5.2f} w5 {100*(a[e.name]<=5).mean():4.1f}% n={len(a)} | fail: MAE {b[e.name].mean():5.2f} w5 {100*(b[e.name]<=5).mean():4.1f}%')
for m in ['Still','Light','Active']:
    s=R[R.Motion==m]; print(m,'new-detector app gate pass %.2f | paper belt %.2f | BIOPAC %.2f'%(gate_app[s.index].mean(),s.Belt_good.mean(),s.BIOPAC_good.mean()))
R.to_csv(WORK + 'val_qrs_windows_v2.csv',index=False)
