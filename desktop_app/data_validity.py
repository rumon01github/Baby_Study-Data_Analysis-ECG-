"""Conservative missing-data masks; source files remain unchanged."""
import numpy as np

def imu_valid(time, acc, gyro=None):
    t=np.asarray(time,float);a=np.asarray(acc,float)
    valid=np.isfinite(a).all(axis=1)&~np.all(a==0,axis=1)
    arrays=[a]
    if gyro is not None:
        g=np.asarray(gyro,float);valid &= np.isfinite(g).all(axis=1);arrays.append(g)
    if len(t)>1:
        dt=np.median(np.diff(t))
        for x in arrays:
            edges=np.r_[0,np.flatnonzero(np.any(x[1:]!=x[:-1],axis=1))+1,len(t)]
            for i,j in zip(edges[:-1],edges[1:]):
                if t[j-1]-t[i]+dt>.5:valid[i:j]=False
    return valid

def mask_imu(time,acc,gyro=None):
    valid=imu_valid(time,acc,gyro)
    a=np.asarray(acc,float).copy();a[~valid]=np.nan
    if gyro is None:return a
    g=np.asarray(gyro,float).copy();g[~valid]=np.nan
    return a,g
