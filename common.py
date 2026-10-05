"""Shared helpers: beat matching, reliability gate, belt CSV loading, lag alignment, motion level."""
import glob, pickle, numpy as np, pandas as pd
from config import *

def f1(a, b, tol=TOL):
    """F1 between two sorted beat-time arrays at +-tol seconds."""
    if len(a) == 0 or len(b) == 0: return 0.0
    j = np.searchsorted(b, a); tp = 0
    for i, x in enumerate(a):
        c = [k for k in (j[i] - 1, j[i]) if 0 <= k < len(b)]
        if c and min(abs(b[k] - x) for k in c) <= tol: tp += 1
    return 2 * tp / (len(a) + len(b))

def sqi(dev, a, b):
    """Reliability gate for one device in [a, b): bSQI (NeuroKit vs Hamilton), HR, R-R CV, tSQI (descriptive)."""
    fs = dev['fs']; t0 = dev['t0']
    nk = dev['nk'] / fs + t0; hm = dev['ham'] / fs + t0
    A = nk[(nk >= a) & (nk < b)]; B = hm[(hm >= a) & (hm < b)]
    r = {'n': len(A)}
    if len(A) < 4:
        r.update(bsqi=0, tsqi=0, hr=np.nan, rrcv=np.nan, good=False); return r
    rr = np.diff(A); hr = 60 / np.median(rr)
    r['bsqi'] = f1(A, B); r['hr'] = hr; r['rrcv'] = np.std(rr) / np.mean(rr)
    w = int(0.12 * fs); sig = dev['sig']; idx = (dev['nk'][(nk >= a) & (nk < b)]).astype(int)
    segs = np.array([sig[i - w:i + w] for i in idx if i - w >= 0 and i + w < len(sig)])
    if len(segs) >= 3:
        tmpl = np.median(segs, axis=0); r['tsqi'] = float(np.nanmean([np.corrcoef(s, tmpl)[0, 1] for s in segs]))
    else: r['tsqi'] = 0
    r['good'] = bool(r['bsqi'] >= BSQI_MIN and HR_MIN <= hr <= HR_MAX and r['rrcv'] < RRCV_MAX)
    return r

def cache(): return pickle.load(open(WORK + 'beats_cache.pkl', 'rb'))
def belt_path(trial):
    """Belt CSV for 'P03 T01'. Trial folders may carry a suffix, e.g. 'T02 (Excluded)'."""
    p, t = trial.split(maxsplit=1)
    hits = glob.glob(BELT + f'{p}/{t}/*BELT*.csv') or glob.glob(BELT + f'{p}/{t} */*BELT*.csv') or glob.glob(BELT + f'{p}/{t}*/*BELT*.csv')
    if not hits: raise FileNotFoundError(f'no *BELT*.csv for {trial} under {BELT}{p}/{t}*/')
    return hits[0]
def belt_csv(trial):
    d = pd.read_csv(belt_path(trial), usecols=['time_s', 'ECG', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'])
    d['acc_g'] = np.sqrt(d.AccX**2 + d.AccY**2 + d.AccZ**2) / 4096
    d['gyro'] = np.sqrt(d.GyroX**2 + d.GyroY**2 + d.GyroZ**2)
    return d
def beats(dev, which='nk'): return dev[which] / dev['fs'] + dev['t0']
def tvec(dev, shift=0): return np.arange(len(dev['sig'])) / dev['fs'] + dev['t0'] - shift

def local_lag(bt, ct, a, b, search=1.0):
    """Belt clock drifts: find the lag (s) that best aligns belt beats bt to reference beats ct inside [a, b)."""
    B = bt[(bt >= a - search) & (bt < b + search)]; R = ct[(ct >= a) & (ct < b)]
    if len(B) < 3 or len(R) < 3: return 0.0
    best = (-1, 0)
    for L in np.arange(-search, search, 0.004):
        d = np.abs((B - L)[:, None] - R[None, :]).min(1); s = (d < 0.03).sum()
        if s > best[0]: best = (s, L)
    L = best[1]; d = (B - L)[:, None] - R[None, :]; i = np.abs(d).argmin(1); dd = d[np.arange(len(B)), i]
    m = np.abs(dd) < 0.05
    return L + np.median(dd[m]) if m.any() else L

def moving_flag(b):
    """Per-sample moving flag from gyro magnitude (trial 20th percentile + MOVE_OFFSET)."""
    gy = b.gyro.values; return gy > np.percentile(gy, 20) + MOVE_OFFSET
def motion_class(share):
    return 'Still' if share < STILL_MAX else ('Light' if share < LIGHT_MAX else 'Active')
