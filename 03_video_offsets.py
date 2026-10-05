"""Step 3. Video-to-data offset per annotated trial: cross-correlate the annotated movement signal with gyroscope activity.
EXP = expected offsets from clock/pixel landmarks (used only to bound the search). Output: work/video_offsets_12.csv (status OK/CHECK)."""
import numpy as np, pandas as pd, glob, json
from config import *
from common import belt_path
V = VIDEO
MOV={'limb_movement','head_movement','caregiver_handling','crying','wire_contact','belt_contact','contact_target_uncertain','caregiver_contact_candidate','trunk_repositioning'}
def gyro_motion(p,t,fs=10):
    b=pd.read_csv(belt_path(f'{p} {t}'),usecols=['time_s','GyroX','GyroY','GyroZ'])
    g=np.linalg.norm(b[['GyroX','GyroY','GyroZ']].values.astype(float),axis=1)
    tg=np.arange(0,b.time_s.iloc[-1],1/fs); gi=np.interp(tg,b.time_s.values,g)
    return tg,pd.Series(gi).rolling(fs,center=True,min_periods=1).mean().values
def ann_signal(a,dur,fs=10):
    tv=np.arange(0,dur,1/fs); s=np.zeros(len(tv)); still=np.zeros(len(tv))
    for r in a.itertuples():
        m=(tv>=r.video_start_s)&(tv<r.video_end_s)
        if r.label in MOV: s[m]+=2 if r.label=='limb_movement' else 1
        if r.label=='minimal_visible_movement': still[m]=1
    return tv,np.where(still>0,0,s)
z=lambda v:(v-v.mean())/(v.std()+1e-9)
EXP={'P01 T01':25,'P01 T02':21.5,'P02 T01':52.8,'P03 T01':32.6,'P03 T02':6.3,'P04 T01':-2.3,'P04 T02':-53.6,'P09 T03':0}
out=[]
files = sorted(glob.glob(V + 'P*/*/P*_annotations.csv') + glob.glob(V + 'P*/P*_annotations.csv'))
if not files: raise FileNotFoundError(f'No annotation CSVs found under {V}. Check NEOTEX_DATA or NEOTEX_VIDEO.')
for f in files:
    a=pd.read_csv(f); p=a.participant.iloc[0]; t=a.trial.iloc[0]; tr=f'{p} {t}'
    tv,s=ann_signal(a,a.video_end_s.max()); tg,g=gyro_motion(p,t); T=tg[-1]
    lo,hi=(EXP[tr]-90,EXP[tr]+90) if tr in EXP else (-max(tv),T)
    res=[]
    for lag in np.arange(lo,hi+0.01,0.2):
        gi=np.interp(tv+lag,tg,g,left=np.nan,right=np.nan); m=np.isfinite(gi)
        if m.sum()<600 or m.mean()<0.3: continue
        res.append((np.corrcoef(z(s[m]),z(gi[m]))[0,1],round(lag,1),round(m.mean(),2)))
    res.sort(reverse=True); best=res[0]; second=max([r for r in res if abs(r[1]-best[1])>5],default=(np.nan,np.nan,0))
    # refine ±3 s at 0.1
    fine=[]
    for lag in np.arange(best[1]-3,best[1]+3,0.1):
        gi=np.interp(tv+lag,tg,g,left=np.nan,right=np.nan); m=np.isfinite(gi); fine.append((np.corrcoef(z(s[m]),z(gi[m]))[0,1],round(lag,1)))
    fb=max(fine)
    # consistency: halves
    halves=[]
    for k in range(2):
        seg=(tv>=k*len(tv)/20)&(tv<(k+1)*len(tv)/20) if False else (tv>=k*tv[-1]/2)&(tv<(k+1)*tv[-1]/2)
        bb=None
        for lag in np.arange(fb[1]-15,fb[1]+15,0.2):
            gi=np.interp(tv[seg]+lag,tg,g,left=np.nan,right=np.nan); m=np.isfinite(gi)
            if m.sum()<300: continue
            r=np.corrcoef(z(s[seg][m]),z(gi[m]))[0,1]
            if bb is None or r>bb[0]: bb=(r,round(lag,1))
        halves.append(bb)
    ok = fb[0]>=0.2 and (fb[0]-second[0])>=0.05 and all(h and abs(h[1]-fb[1])<=2.5 for h in halves)
    out.append(dict(Trial=tr,n_events=len(a),video_dur=round(tv[-1],1),data_dur=round(T,1),lag=fb[1],r=round(fb[0],3),second_r=round(second[0],3),second_lag=second[1],half1=halves[0],half2=halves[1],expected=EXP.get(tr),status='OK' if ok else 'CHECK'))
    print(out[-1])
pd.DataFrame(out).to_csv(WORK + 'video_offsets_12.csv', index=False)
