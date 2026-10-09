"""PPG display filtering within valid segments only."""
import numpy as np
from scipy import signal,ndimage
LOWCUT=.5;HIGHCUT=10.;ORDER=4;DESPIKE=9
DESCRIPTION='Valid segments only; 9-sample median, detrend, 0.5-10 Hz Butterworth; gaps and 2 s filter edges excluded; flat windows not amplified'

def process_ppg(time,values):
    t=np.asarray(time,float);x=np.asarray(values,float);out=np.full(len(x),np.nan)
    if len(t)<2 or len(t)!=len(x) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):return out
    dt=np.median(np.diff(t));fs=1/dt;high=min(HIGHCUT,.45*fs)
    if high<=LOWCUT:return out
    valid=np.isfinite(x)
    edges=np.flatnonzero(np.r_[True,(np.diff(t)>3*dt)|~valid[:-1]|~valid[1:],True])
    sos=signal.butter(ORDER,[LOWCUT,high],btype='bandpass',fs=fs,output='sos')
    for i,j in zip(edges[:-1],edges[1:]):
        if not valid[i:j].all() or j-i<=max(27,int(5*fs)):continue
        if np.ptp(x[i:j])==0:out[i:j]=0;continue
        y=ndimage.median_filter(x[i:j],size=DESPIKE)
        y=signal.sosfiltfilt(sos,signal.detrend(y))
        keep=(t[i:j]>=t[i]+2)&(t[i:j]<=t[j-1]-2)
        out[i:j]=np.where(keep,y,np.nan)
    return out

def display_values(time,values,raw,lo,hi):
    t=np.asarray(time);v=np.asarray(values);raw=np.asarray(raw)
    m=(t>=lo)&(t<=hi)&np.isfinite(v)&np.isfinite(raw)
    if not m.any():return np.full(len(v),np.nan)
    sd=np.std(v[m]);floor=max(1e-12,.01*np.nanstd(v),1e-9*np.max(np.abs(raw[m])))
    if np.ptp(raw[m])==0 or sd<=floor:return np.where(np.isfinite(v),0.,np.nan)
    return (v-np.mean(v[m]))/sd
