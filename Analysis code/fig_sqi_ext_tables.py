"""Descriptive SQIs (kSQI, pSQI, basSQI) per window -> work/windows_sqi_ext.csv (needed by fig4 only if the SQI panels are used)."""
from pstyle import *
from scipy import stats, signal
d=cache(); q=pd.read_csv(SC+'quadrants.csv')
rows=[]
for _,r in q.iterrows():
    D=d[r.Trial]; a,b=r.Start_s,r.Start_s+10; out=dict(Trial=r.Trial,Start_s=a)
    for k,p in [('belt','B'),('bio','O')]:
        dev=D[k]; t0,fs=dev['t0'],dev['fs']; i0,i1=int((a-t0)*fs),int((b-t0)*fs); x=dev['sig'][max(i0,0):i1].astype(float)
        if len(x)<fs*5: continue
        out[p+'_kSQI']=stats.kurtosis(x,fisher=False); out[p+'_sSQI']=stats.skew(x)
        f,P=signal.welch(x,fs,nperseg=int(fs*4)); band=lambda lo,hi: P[(f>=lo)&(f<=hi)].sum()
        out[p+'_pSQI']=band(5,15)/band(5,40)
        out[p+'_basSQI']=1-band(0,1)/band(0,40)
    rows.append(out)
X=q.merge(pd.DataFrame(rows),on=['Trial','Start_s']); X.to_csv('windows_sqi_ext.csv',index=False)
for m in ['Still','Light','Active']:
    g=X[(X.Motion==m)&(X.In_comparison)]; print(m,g[['B_kSQI','O_kSQI','B_pSQI','O_pSQI','B_sSQI','O_sSQI','Belt_tSQI','Bio_tSQI']].median().round(2).to_dict())
