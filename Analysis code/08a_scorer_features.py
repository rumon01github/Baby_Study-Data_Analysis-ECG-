"""S3 step 1-3: candidate generation + features + labels from the manual handoff."""
import numpy as np, pandas as pd, json, sys
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from common import *
import neotex_qrs_v2 as v2

P = json.load(open('neotex_qrs_params_v2.json'))
mb = pd.read_csv(WORK + 'manual_beats_clean.csv')
ss = pd.read_csv(WORK + 'strips_gate_vs_human.csv')[['Trial', 'start_s', 'quality', 'confidence', 'n_human', 'moving']]
FS = 100; TOL = 0.075
rows = []
for tr, g in ss.groupby('Trial'):
    b = belt_csv(tr); tt = b.time_s.values; x = b.ECG.values.astype(float)
    r, xb, m = v2.detect(x, FS, return_feat=True, **P)
    det_t = tt[r]
    # all local maxima of the integrator above a tiny floor = candidate pool (includes rejected ones)
    cand, _ = find_peaks(m, distance=int(0.2 * FS), height=np.percentile(m, 20))
    # refine like the detector: argmax |xb| in [-120, +20] ms
    ref = np.array([max(0, c - 12) + int(np.argmax(np.abs(xb[max(0, c - 12):c + 2]))) for c in cand])
    ref = np.unique(ref); ct = tt[ref]
    is_det = np.isin(ref, r)
    # gyro motion (sample-level moving flag as in paper)
    gy = b.gyro.values; mv = (gy > np.percentile(gy, 20) + 200).astype(float)
    # running template from detector beats (median of +-120 ms) per trial, built on first 60 s of detections
    w = 12; tpl_src = [xb[i - w:i + w] for i in r if i - w >= 0 and i + w < len(xb)][:150]
    tpl = np.median(tpl_src, 0) if len(tpl_src) >= 5 else None
    hm = mb[mb.Trial == tr]
    m_thr = np.r_[m[0], m]
    for _, s in g.iterrows():
        a, e = s.start_s, s.start_s + 8
        h = np.sort(hm.loc[hm.strip_start_s == a, 'peak_time_s'].values)
        sel = np.where((ct >= a) & (ct < e))[0]
        if len(sel) == 0: continue
        dts = det_t[(det_t >= a - 4) & (det_t < e)]
        for k in sel:
            i = ref[k]; t = ct[k]
            # label
            lab = int(len(h) > 0 and np.min(np.abs(h - t)) <= TOL)
            # features
            amp = abs(xb[i]); mh = m[min(len(m) - 1, i + 6)]
            seg = xb[max(0, i - 50):i + 50]; ks = kurtosis(seg) if len(seg) > 10 else np.nan
            loc_amp = np.max(np.abs(xb[max(0, i - 100):i + 100])); rel_amp = amp / loc_amp if loc_amp > 0 else 0
            prev = dts[dts < t - 0.05]; rr_prev = (t - prev[-1]) if len(prev) else np.nan
            rr_med = np.median(np.diff(prev[-8:])) if len(prev) >= 3 else np.nan
            rr_ratio = rr_prev / rr_med if rr_med and rr_med > 0 else np.nan
            if tpl is not None and i - w >= 0 and i + w < len(xb):
                seg2 = xb[i - w:i + w]; tc = np.corrcoef(seg2, tpl)[0, 1] if seg2.std() > 0 else 0
            else: tc = np.nan
            mv1 = mv[max(0, i - 50):i + 50].mean()
            # integrator peak relative to its 2-s neighbourhood
            nb = m[max(0, i - 200):i + 200]; m_rel = mh / nb.max() if nb.max() > 0 else 0
            rows.append(dict(Trial=tr, P=tr[:3], start_s=a, quality=s.quality, t=t, label=lab, is_det=int(is_det[k]),
                             amp=amp, rel_amp=rel_amp, m_rel=m_rel, kurt=ks, rr_prev=rr_prev, rr_ratio=rr_ratio, tcorr=tc, moving=mv1, strip_moving=s.moving))
F = pd.DataFrame(rows); F.to_csv(WORK + 's3_candidates.csv', index=False)
print(len(F), F.label.mean(), F.groupby('quality').agg(n=('label', 'size'), pos=('label', 'mean'), det=('is_det', 'mean')).round(3))
# how many human beats have a candidate within tol (recall ceiling of the candidate pool)
tot = hit = 0
for tr, g in ss.groupby('Trial'):
    c = np.sort(F.loc[F.Trial == tr, 't'].values); hm = mb[mb.Trial == tr]
    for _, s in g.iterrows():
        if s.quality == 'unreadable': continue
        h = hm.loc[hm.strip_start_s == s.start_s, 'peak_time_s'].values
        for v in h:
            tot += 1; j = np.searchsorted(c, v)
            if any(abs(c[k] - v) <= TOL for k in (j - 1, j) if 0 <= k < len(c)): hit += 1
print('candidate-pool recall of human beats (non-unreadable):', hit, tot, round(hit / tot, 4))
