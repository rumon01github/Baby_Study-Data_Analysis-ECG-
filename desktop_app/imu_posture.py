"""Belt tilt and activity from the accelerometer, adapted from Chung et al., Nat Med 2020 (doi:10.1038/s41591-020-0792-9)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from data_validity import mask_imu

COUNTS_PER_G=4096.  # ±8 g range: resting |acc| in the study data is ~4,100 counts
STALE_S=.5  # the accelerometer updates ~24 times/s; identical XYZ held longer than this is a dropout, not stillness


def _segments(t,a,min_s):
    """Finite runs without time gaps over three intervals, at least min_s long, resampled to the median interval."""
    t=np.asarray(t,dtype=float);a=np.asarray(a,dtype=float)
    if len(t)<2 or a.shape!=(len(t),3):raise ValueError('Three accelerometer channels and timestamps are required.')
    if not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):raise ValueError('IMU timestamps must be finite and strictly increasing.')
    dt=float(np.median(np.diff(t)))
    if 1/dt<10:raise ValueError('IMU sampling below 10 Hz is unsupported.')
    a=mask_imu(t,a)
    valid=np.all(np.isfinite(a),axis=1)
    edges=np.flatnonzero(np.r_[True,(np.diff(t)>3*dt)|(~valid[:-1])|(~valid[1:]),True])
    for i,j in zip(edges[:-1],edges[1:]):
        if not valid[i:j].all() or t[j-1]-t[i]<min_s:continue
        grid=np.arange(t[i],t[j-1]+dt*.1,dt)
        yield i,j,grid,np.column_stack([np.interp(grid,t[i:j],a[i:j,k]) for k in range(3)])


def tilt(time,acc):
    """Belt orientation in degrees from the gravity direction (third-order Butterworth low-pass at 0.1 Hz).

    from_flat: angle between belt Z and gravity, 0 when the belt lies flat facing up; sign-free, defined in every posture.
    roll: rotation about X, atan2(y, z), unwrapped within each segment; undefined when X points along gravity.
    pitch: rotation about Y, atan2(-x, sqrt(y^2 + z^2)), between -90 and 90.
    NaN outside continuous segments of at least 30 s. Angles are relative to the belt, not the infant's body."""
    t=np.asarray(time,dtype=float);out={k:np.full(len(t),np.nan) for k in ('from_flat','roll','pitch')}
    for i,j,grid,a in _segments(t,acc,30.):
        x,y,z=sosfiltfilt(butter(3,.1,btype='lowpass',fs=1/(grid[1]-grid[0]),output='sos'),a,axis=0).T
        for key,angle in [('from_flat',np.arctan2(np.hypot(x,y),z)),('roll',np.unwrap(np.arctan2(y,z))),('pitch',np.arctan2(-x,np.hypot(y,z)))]:
            out[key][i:j]=np.interp(t[i:j],grid,np.degrees(angle))
    return out


def activity(time,acc,band=(1.,8.)):
    """Activity in g each second: RMS of the three-axis acceleration after a third-order Butterworth band-pass.

    Chung et al. used 1-10 Hz on a 100 Hz accelerometer. The belt accelerometer updates ~24 times/s (saved on a 100 Hz
    grid by repeating values), so the default upper edge is 8 Hz. Samples inside held runs over STALE_S, and one second at
    each segment end, are excluded; a second needs 90% of its expected samples or it is NaN."""
    t=np.asarray(time,dtype=float);a=np.asarray(acc,dtype=float)
    same=np.r_[False,np.all(a[1:]==a[:-1],axis=1)];run=np.cumsum(~same)
    dt=float(np.median(np.diff(t))) if len(t)>1 else np.nan
    stale=np.bincount(run)[run]*dt>STALE_S
    energy=np.full(len(t),np.nan)
    for i,j,grid,seg in _segments(t,a,10.):
        sos=butter(3,[band[0],min(band[1],.4/dt)],btype='bandpass',fs=1/dt,output='sos')
        f=sosfiltfilt(sos,seg,axis=0)
        e=np.interp(t[i:j],grid,np.sum(f**2,axis=1))
        e[(t[i:j]<t[i]+1)|(t[i:j]>t[j-1]-1)]=np.nan
        energy[i:j]=e
    energy[stale]=np.nan
    start=np.floor(t[0]);n=int(np.ceil(t[-1]-start)) if len(t) else 0
    sec=np.clip(((t-start)//1).astype(int),0,max(n-1,0));ok=np.isfinite(energy)
    count=np.bincount(sec[ok],minlength=n);total=np.bincount(sec[ok],weights=energy[ok],minlength=n)
    with np.errstate(invalid='ignore',divide='ignore'):
        g=np.where(count>=.9/dt,np.sqrt(total/count),np.nan)/COUNTS_PER_G
    return dict(time=start+np.arange(n)+.5,g=g)


def window_activity(result,start,end):
    """Root-mean-square of the valid per-second values whose centres fall in [start, end), or NaN."""
    m=(result['time']>=start)&(result['time']<end);v=result['g'][m];v=v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v**2))) if len(v) else np.nan
