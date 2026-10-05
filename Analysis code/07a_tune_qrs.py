"""Step 7a. Tune belt detector filter parameters on the odd-numbered strips of the first 23 labelled strips; test on even strips."""
import numpy as np, pandas as pd, glob, json, itertools, pickle
from neotex_qrs import detect
from common import *
k=pd.read_csv('segment_key.csv'); LAB=json.load(open('round1_labels.json')); strips=[]
for _,r in k.iterrows():
    if r.id not in LAB: continue
    df=belt_csv(r.trial)
    a=r.start_s; lab=np.array(sorted(LAB[r.id]['peaks_s']),float); man=lab+a if lab.max()<=9 else lab
    seg=df[(df.time_s>=a-10)&(df.time_s<a+8)]           # 10 s run-in for the adaptive threshold
    strips.append(dict(id=r.id,cat=r.category,t=seg.time_s.values,x=seg.ECG.values,man=man,a=a))
def score(params, ids):
    fs=[]
    for s in strips:
        if s['id'] not in ids: continue
        fs_=1/np.median(np.diff(s['t']))
        r=detect(s['x'],fs=100,**params); tr=s['t'][r]; tr=tr[(tr>=s['a'])&(tr<s['a']+8)]
        fs.append(f1(tr,s['man'],0.075))
    return np.mean(fs)
ids=[s['id'] for s in strips]; A=ids[0::2]; B=ids[1::2]
grid=list(itertools.product([5,8,10],[25,30,35],[50,70,90],[0.25,0.35,0.5]))
res=[]
for lo,hi,W,beta in grid:
    p=dict(lo=lo,hi=hi,W_ms=W,beta=beta); res.append((score(p,A),score(p,B),p))
res.sort(key=lambda z:-z[0]); print('top on A (tune) -> tested on B:')
for a_,b_,p in res[:6]: print(round(a_,3),round(b_,3),p)
best=res[0][2]; print('\nchosen',best,'F1 all strips',round(score(best,ids),4))
# per-strip report
rows=[]
for s in strips:
    r=detect(s['x'],fs=100,**best); tr=s['t'][r]; tr=tr[(tr>=s['a'])&(tr<s['a']+8)]; rows.append((s['id'],s['cat'],len(s['man']),len(tr),round(f1(tr,s['man'],0.075),3)))
print(pd.DataFrame(rows,columns=['id','cat','n_manual','n_det','F1']).to_string(index=False))
json.dump(best,open('neotex_qrs_params.json','w'))
