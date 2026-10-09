"""Label-free exploratory movement detection; not InfantMotion2Vec."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from data_validity import mask_imu


def process_imu(time, acc, gyro):
    t=np.asarray(time,dtype=float);a=np.asarray(acc,dtype=float);g=np.asarray(gyro,dtype=float)
    if len(t)<2 or a.shape!=(len(t),3) or g.shape!=a.shape:
        raise ValueError('Six IMU channels and timestamps are required.')
    if not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError('IMU timestamps must be finite and strictly increasing.')
    dt=float(np.median(np.diff(t)));fs=1/dt
    if fs<10:raise ValueError('IMU sampling below 10 Hz is unsupported.')
    a,g=mask_imu(t,a,g)
    valid=np.all(np.isfinite(a),axis=1)&np.all(np.isfinite(g),axis=1)
    # Do not interpolate across missing rows or gaps greater than three intervals.
    breaks=np.r_[True,(np.diff(t)>3*dt)|(~valid[:-1])|(~valid[1:]),True]
    edges=np.flatnonzero(breaks)
    filtered=[]
    hp=butter(2,.3,btype='highpass',fs=fs,output='sos')
    lp=butter(4,min(20.,.4*fs),btype='lowpass',fs=fs,output='sos')
    for i,j in zip(edges[:-1],edges[1:]):
        if not np.all(valid[i:j]) or j-i<max(30,int(10*fs)):continue
        grid=np.arange(t[i],t[j-1]+dt*.1,dt)
        av=np.column_stack([np.interp(grid,t[i:j],a[i:j,k]) for k in range(3)])
        gv=np.column_stack([np.interp(grid,t[i:j],g[i:j,k]) for k in range(3)])
        av=sosfiltfilt(lp,sosfiltfilt(hp,av,axis=0),axis=0)
        gv=sosfiltfilt(lp,gv,axis=0)
        # Exclude segment ends, where offline filter transients are most likely.
        keep=(grid>=grid[0]+1)&(grid<=grid[-1]-1)
        filtered.append((grid[keep],av[keep],gv[keep]))
    if not filtered:raise ValueError('No valid IMU segment of at least 10 seconds.')
    tt=np.concatenate([x[0] for x in filtered]);aa=np.concatenate([x[1] for x in filtered]);gg=np.concatenate([x[2] for x in filtered])
    starts=np.arange(np.floor(t[0]/.5)*.5,t[-1],.5)
    ar=np.full(len(starts),np.nan);gr=ar.copy();bias0=np.median(gg,axis=0)
    masks=[]
    for n,s in enumerate(starts):
        m=(tt>=s)&(tt<s+.5);masks.append(m)
        if m.sum()<.9*.5*fs:continue
        ar[n]=np.sqrt(np.mean(np.sum(aa[m]**2,axis=1)))
        gr[n]=np.sqrt(np.mean(np.sum((gg[m]-bias0)**2,axis=1)))
    good=np.flatnonzero(np.isfinite(ar)&np.isfinite(gr))
    if len(good)<20:raise ValueError('Fewer than 10 seconds of valid RMS intervals.')
    ranks=sum(np.searchsorted(np.sort(v[good]),v[good],side='left') for v in (ar,gr))
    chosen=good[np.argsort(ranks,kind='stable')[:max(10,int(np.ceil(.2*len(good))))]]
    selected=np.any(np.stack([masks[i] for i in chosen]),axis=0)
    bias=np.median(gg[selected],axis=0)
    for n in good:gr[n]=np.sqrt(np.mean(np.sum((gg[masks[n]]-bias)**2,axis=1)))
    def threshold(v):
        base=v[chosen];med=np.median(base)
        allv=v[good];center=np.median(allv);spread=1.4826*np.median(np.abs(allv-center))
        return float(max(40.96 if v is ar else 20.,med+6*1.4826*np.median(np.abs(base-med)),center+6*spread))
    return dict(time=starts+.25,acc=ar,gyro=gr,acc_threshold=threshold(ar),gyro_threshold=threshold(gr),
                reference_starts=starts[chosen],sampling_hz=fs,gyro_bias=bias)


def summary(result,start):
    t=result['time'];m=(t>=start)&(t<start+5)
    a=result['acc'][m];g=result['gyro'][m];valid=np.isfinite(a)&np.isfinite(g)
    coverage=valid.sum()/10
    if not valid.any():return 'IMU unavailable'
    am=a[valid]>result['acc_threshold'];gm=g[valid]>result['gyro_threshold'];moving=am|gm
    flags=np.zeros(len(a),bool);flags[valid]=moving
    label='Movement detected' if np.any(flags[1:]&flags[:-1]) else 'No sustained movement detected'
    if coverage<.8:label='Insufficient IMU coverage'
    return (f'{label} · movement {moving.mean()*100:.0f}% of valid time '
            f'(acc {am.mean()*100:.0f}%, gyro {gm.mean()*100:.0f}%) · '
            f'coverage {coverage*100:.0f}% · RMS: acc {np.sqrt(np.mean(a[valid]**2)):.1f}, '
            f'gyro {np.sqrt(np.mean(g[valid]**2)):.1f} raw units')
