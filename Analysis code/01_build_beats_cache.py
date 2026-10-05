"""Step 1. Detect beats once per paired trial with two generic detectors (NeuroKit2, Hamilton-Tompkins) on belt and BIOPAC.
Belt is resampled to 250 Hz, BIOPAC to 500 Hz; the BIOPAC ECG column is chosen by beat plausibility (export labels are unreliable).
Output: work/beats_cache.pkl  {trial: {'belt': {...}, 'bio': {...}, 'T': duration}}"""
import os, numpy as np, pandas as pd, glob, pickle, warnings, neurokit2 as nk; warnings.filterwarnings('ignore')
from config import *
def bpath(bf, p, t):
    folder = os.path.dirname(bf)   # the BIOPAC file normally sits next to the belt file
    c = glob.glob(os.path.join(folder, '*BIOPAC*.csv')) or glob.glob(DATA + f'Organized Study Data/{p}/{t}*/*BIOPAC*.csv') or glob.glob(VIDEO + f'{p}/*{t}*/*BIOPAC*.csv')
    return c[0] if c else None
def det(x, fs, m):
    try:
        cl = nk.ecg_clean(x, sampling_rate=fs, method='neurokit') if m == 'neurokit' else x
        _, r = nk.ecg_peaks(cl, sampling_rate=fs, method=m); return np.asarray(r['ECG_R_Peaks'])
    except Exception: return np.array([], int)
out = {}
for bf in sorted(glob.glob(BELT + 'P*/T*/*BELT*.csv')):
    p, tdir = bf.replace('\\', '/').split('/')[-3:-1]; t = tdir.split()[0]      # 'T02 (Excluded)' -> 'T02'
    pf = bpath(bf, p, t)
    if pf is None: continue
    b = pd.read_csv(bf, usecols=['time_s', 'ECG']); FS = 250
    tg = np.arange(b.time_s.iloc[0], b.time_s.iloc[-1], 1 / FS); x = np.interp(tg, b.time_s.values, b.ECG.values)
    xc = nk.ecg_clean(x, sampling_rate=FS, method='neurokit')
    belt = {'t0': tg[0], 'fs': FS, 'sig': xc.astype(np.float32), 'nk': det(x, FS, 'neurokit'), 'ham': det(x, FS, 'hamilton2002')}
    bp = pd.read_csv(pf).dropna(subset=['time_s']); best = None
    for col in [c for c in bp.columns if 'ECG MODULE' in c or 'AHA' in c]:
        y = bp[col].values[::2].astype(float); k = det(y, 500, 'neurokit'); tt = k / 500
        hr = 60 / np.diff(tt) if len(tt) > 2 else np.array([0]); sc = np.mean((hr > 80) & (hr < 220)) * len(tt)
        if best is None or sc > best[0]: best = (sc, col, y, k)
    _, col, y, k = best
    yc = nk.ecg_clean(y, sampling_rate=500, method='neurokit')
    bio = {'t0': bp.time_s.iloc[0], 'fs': 500, 'sig': yc.astype(np.float32), 'nk': k, 'ham': det(y, 500, 'hamilton2002'), 'col': col}
    out[f'{p} {t}'] = {'belt': belt, 'bio': bio, 'T': min(tg[-1], bp.time_s.iloc[-1])}
    print(p, t, len(belt['nk']), len(bio['nk']), flush=True)
pickle.dump(out, open(WORK + 'beats_cache.pkl', 'wb'))
