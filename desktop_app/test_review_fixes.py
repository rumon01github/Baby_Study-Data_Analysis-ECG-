"""Regression tests executed on Unity; no patient data required."""
import numpy as np
from data_validity import mask_imu
from ppg_processing import process_ppg,display_values
from imu_processing import process_imu,summary
from breathing import estimate_imu,estimate,clean_capacitive

def test_ppg_gaps_flat_and_short():
    t=np.arange(0,40,.01);x=np.sin(2*np.pi*2*t)
    x[(t>=18)&(t<22)]=np.nan
    y=process_ppg(t,x)
    assert np.isnan(y[(t>=16)&(t<24)]).all()
    assert np.isfinite(y[(t>3)&(t<15)]).all()
    assert np.isnan(process_ppg(np.arange(27)/100,np.ones(27))).all()
    raw=np.ones(len(t))*1000
    displayed=display_values(t,np.sin(t)*1e-9,raw,5,10)
    assert np.nanmax(np.abs(displayed))==0

def test_noise_and_dropout():
    t=np.arange(0,65,.01);rng=np.random.default_rng(73)
    a=rng.normal(0,2,(len(t),3));a[:,2]+=4096
    g=rng.normal(0,2,(len(t),3))
    r=process_imu(t,a,g)
    assert not ((r['acc']>r['acc_threshold'])|(r['gyro']>r['gyro_threshold'])).any()
    assert np.isnan(estimate_imu(t,a)[1]).all()
    a[(t>=25)&(t<28)]=0;g[(t>=25)&(t<28)]=0
    aa,gg=mask_imu(t,a,g)
    assert np.isnan(aa[(t>=25)&(t<28)]).all()
    r=process_imu(t,a,g)
    assert 'Insufficient IMU coverage' in summary(r,25)

def test_fast_ecg_modulation_not_halved():
    t=np.arange(0,65,.01);x=np.zeros(len(t))
    for bt in np.arange(.5,64.5,.4):
        x+=(1+.4*np.sin(2*np.pi*bt))*np.exp(-((t-bt)/.015)**2)
    _,r=estimate(t,x,'ECG')
    assert np.isnan(r).all(),r

def test_sustained_capacitive_dropout():
    t=np.arange(0,65,.01);x=22.8+.03*np.sin(2*np.pi*.55*t)
    x[(t>=25)&(t<28)]=16.43
    y,mask=clean_capacitive(t,x)
    assert mask[(t>=25)&(t<28)].all()
    assert np.isnan(y[(t>=25)&(t<28)]).all()

if __name__=='__main__':
    for name,f in list(globals().items()):
        if name.startswith('test_'):f();print('PASS',name,flush=True)

