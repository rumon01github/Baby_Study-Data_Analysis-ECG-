"""Run with the app's Python dependencies; synthetic algorithm checks."""
import numpy as np
from breathing import estimate, estimate_imu, estimate_capacitive

def test_rates_and_missing_data():
    t = np.arange(0,65,.01)
    x = np.sin(2*np.pi*.6*t)
    times, rates = estimate(t,x,'Resp0')
    assert times[0] == 5  # first 5 s window has an estimate (span 0-30 s)
    assert np.allclose(rates,36,atol=1)
    assert np.isnan(estimate(t,np.zeros_like(t),'Resp0')[1]).all()
    x[(t>20)&(t<21)] = np.nan
    # Spans containing the gap are rejected; windows whose span avoids it (from 40-45 s on) still estimate.
    r = estimate(t,x,'Resp0')[1]
    assert np.isnan(r[:7]).all() and np.isfinite(r[8:]).all()
    pulses = (1+.35*np.sin(2*np.pi*.5*t))*np.sin(2*np.pi*2.5*t)
    assert np.allclose(estimate(t,pulses,'IR')[1],30,atol=2)
    assert len(estimate(t[:500],pulses[:500],'IR')[0]) == 0
    assert len(estimate(t[::-1],pulses,'IR')[0]) == 0

def test_imu_tilt():
    t = np.arange(0,65,.01)
    rng = np.random.default_rng(0)
    tilt = .002*np.sin(2*np.pi*.7*t)  # small breathing tilt (rad) on a lying posture
    acc = 4096*np.column_stack([np.sin(.3+tilt), np.zeros_like(t), np.cos(.3+tilt)]) + rng.normal(0,1,(len(t),3))
    assert np.allclose(estimate_imu(t,acc)[1],42,atol=2)
    assert np.isnan(estimate_imu(t,acc,moving=lambda lo,hi:1.0)[1]).all()

def test_imu_fused_with_jolts():
    # Chest tilt of 0.15 deg at 36 breaths/min seen by a gyro (16.4 counts per deg/s) and the accelerometer
    # (4096 counts per g), plus short linear-acceleration jolts that only the accelerometer feels.
    t = np.arange(0,90,.01)
    rng = np.random.default_rng(2)
    theta = np.radians(.15)*np.sin(2*np.pi*.6*t)
    gyro = np.column_stack([np.degrees(np.gradient(theta,t))*16.4, np.zeros_like(t), np.zeros_like(t)]) + rng.normal(0,.5,(len(t),3))
    acc = 4096*np.column_stack([np.zeros_like(t), np.sin(.3+theta), np.cos(.3+theta)]) + rng.normal(0,2,(len(t),3))
    for s in rng.uniform(1,89,30):
        acc[(t>=s)&(t<s+.3)] += rng.normal(0,150,3)
    from breathing import gyro_scale
    assert abs(gyro_scale(t,acc,gyro)/np.radians(1/16.4) - 1) < .3  # recovers the gyro scale
    rates = estimate_imu(t,acc,gyro=gyro)[1]
    assert np.isfinite(rates).mean() > .8 and np.allclose(np.nanmedian(rates),36,atol=2)

def test_stationary_line_detected_and_removed():
    # A gyro line fixed at 0.64 Hz for the whole recording is interference; breathing that drifts is not.
    from breathing import stationary_line, remove_line
    t = np.arange(0,300,.01)
    rng = np.random.default_rng(3)
    drifting = np.sin(2*np.pi*np.cumsum(.5+.1*np.sin(2*np.pi*t/60))*.01)
    gyro = np.column_stack([40*np.sin(2*np.pi*.64*t), 3*drifting, np.zeros_like(t)]) + rng.normal(0,1,(len(t),3))
    f0 = stationary_line(t,gyro)
    assert f0 is not None and abs(f0-.64) <= .02
    cleaned = remove_line(t,gyro,f0)
    assert cleaned[:,0].std() < .2*gyro[:,0].std()
    assert stationary_line(t,np.column_stack([3*drifting,rng.normal(0,1,len(t)),rng.normal(0,1,len(t))])) is None

def test_breath_by_breath():
    # 30 breaths/min for 60 s, then 45 breaths/min: each breath's own rate should follow the change.
    t = np.arange(0,120,.01)
    phase = 2*np.pi*np.cumsum(np.where(t<60,.5,.75))*.01
    times, rates, bt, br = estimate(t,np.sin(phase),'Resp0',with_breaths=True)
    assert len(bt) and np.all(np.diff(bt) > 0)  # each breath reported once, in order
    assert np.allclose(br[bt<55],30,atol=2) and np.allclose(br[bt>65],45,atol=3)

def test_capacitive_dips_with_spikes():
    # Breathing of 0.06 raw units at 33 breaths/min on a 22.8 baseline, quantised to 0.01 like the belt, plus
    # sharp dropouts to the ~16.4 rail every few seconds: the dropouts must not be counted as breaths.
    t = np.arange(0,90,.01)
    rng = np.random.default_rng(1)
    x = np.round(22.8 + .03*np.cos(2*np.pi*.55*t) + rng.normal(0,.005,len(t)), 2)
    for s in rng.uniform(1,89,25):
        x[(t>=s)&(t<s+.08)] = 16.43
    times, rates, bt, br = estimate_capacitive(t,x,with_breaths=True)
    assert np.isfinite(rates).mean() > .9 and np.allclose(rates[np.isfinite(rates)],33,atol=2)
    assert np.allclose(np.median(br),33,atol=2)
    assert np.isnan(estimate_capacitive(t,np.full_like(t,22.8))[1]).all()  # flat: no breaths

if __name__ == '__main__':
    test_rates_and_missing_data()
    test_imu_tilt()
    test_imu_fused_with_jolts()
    test_stationary_line_detected_and_removed()
    test_breath_by_breath()
    test_capacitive_dips_with_spikes()
    print('Breathing checks passed')
