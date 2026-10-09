"""Run with the app's Python dependencies; synthetic algorithm checks."""
import numpy as np
from imu_posture import tilt, activity, window_activity, COUNTS_PER_G

def test_tilt_angles():
    t = np.arange(0,120,.01)
    rng = np.random.default_rng(0)
    # Flat for 60 s, then tipped 60 deg about Y (gravity moves from +Z towards -X); jolts must not move the angle.
    a = np.radians(np.where(t<60,0,60))
    acc = COUNTS_PER_G*np.column_stack([-np.sin(a), np.zeros_like(t), np.cos(a)]) + rng.normal(0,20,(len(t),3))
    acc[(t>20)&(t<20.3)] += 800
    r = tilt(t,acc)
    flat, tipped = (t>10)&(t<50), (t>70)&(t<110)
    assert np.allclose(r['from_flat'][flat],0,atol=3) and np.allclose(r['from_flat'][tipped],60,atol=1)
    assert np.allclose(r['pitch'][tipped],60,atol=1) and np.allclose(r['roll'],0,atol=3)
    # X along gravity: roll is undefined but from_flat is still exactly 90.
    up = COUNTS_PER_G*np.column_stack([np.ones_like(t), np.zeros_like(t), np.zeros_like(t)]) + rng.normal(0,20,(len(t),3))
    assert np.allclose(tilt(t,up)['from_flat'],90,atol=2)
    # Short segments (under 30 s) give no angle.
    assert np.isnan(tilt(t[:2000],acc[:2000])['from_flat']).all()

def test_activity_in_g():
    t = np.arange(0,60,.01)
    rng = np.random.default_rng(1)
    acc = COUNTS_PER_G*np.column_stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]) + rng.normal(0,2,(len(t),3))
    acc[:,0] += .1*COUNTS_PER_G*np.sin(2*np.pi*3*t)*((t>=20)&(t<40))  # 0.1 g at 3 Hz, inside the band
    acc[:,1] += .1*COUNTS_PER_G*np.sin(2*np.pi*.2*t)  # slow sway, below the band
    r = activity(t,acc)
    assert len(r['time']) == 60 and r['time'][0] == .5
    assert abs(window_activity(r,25,35)-.1/np.sqrt(2)) < .005
    assert window_activity(r,5,15) < .005
    assert np.isnan(r['g'][0]) and np.isnan(r['g'][-1])  # segment ends excluded
    # A held value (dropout) is not reported as stillness.
    held = acc.copy(); held[(t>=45)&(t<48)] = held[np.searchsorted(t,45)]
    g = activity(t,held)['g']
    assert np.isnan(g[45:48]).all() and np.isfinite(g[49:58]).all()

def test_rejects_bad_input():
    t = np.arange(0,60,.01); acc = np.ones((len(t),3))
    for bad_t in (t[::-1], np.arange(0,60,.2)[:len(t)]):
        try:tilt(bad_t,acc[:len(bad_t)])
        except ValueError:pass
        else:raise AssertionError('expected ValueError')


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):test()
    print("IMU posture checks passed")
