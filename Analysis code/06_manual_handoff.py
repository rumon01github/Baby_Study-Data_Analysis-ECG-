"""Step 6. Prepare the expert beat labels (manual review handoff) for validation.
- merges markers duplicated at strip boundaries (same beat labelled in two adjacent 8-s strips, within TOL)
- per-strip human HR / R-R CV
- per-strip run-time detector output (v2), two-threshold agreement, RR gate, motion share, F1 vs expert
Outputs: work/manual_beats_clean.csv, work/strip_summary_hr.csv, work/strips_gate_vs_human.csv"""
import numpy as np, pandas as pd, json
from common import *
import neotex_qrs_v2 as v2
P = json.load(open('neotex_qrs_params_v2.json'))
mb = pd.read_csv(HANDOFF + 'Manual labels/ALL_MANUAL_BEATS.csv'); mb['Trial'] = mb.trial.str.replace(' (Excluded)', '', regex=False).str.replace('_', ' ')   # handoff folder names carry an '(Excluded)' tag for 2 trials
ss = pd.read_csv(HANDOFF + 'Review report/strip_summary.csv'); ss['Trial'] = ss.trial.str.replace(' (Excluded)', '', regex=False).str.replace('_', ' ')
# --- boundary duplicates ------------------------------------------------------
keep = []
for tr, g in mb.sort_values(['Trial', 'peak_time_s']).groupby('Trial'):
    t = g.peak_time_s.values; s = g.strip_start_s.values; ok = np.ones(len(g), bool)
    for i in range(1, len(g)):
        if t[i] - t[i - 1] <= TOL and s[i] != s[i - 1]: ok[i] = False
    keep.append(g[ok])
mb = pd.concat(keep); mb.to_csv(WORK + 'manual_beats_clean.csv', index=False); print('markers after boundary de-duplication:', len(mb))
# --- human HR per strip ------------------------------------------------------
hr = []
for (tr, a), g in mb.groupby(['Trial', 'strip_start_s']):
    rr = np.diff(np.sort(g.peak_time_s.values)); hr.append(dict(Trial=tr, start_s=a, human_peaks=len(g), human_hr=60 / np.median(rr) if len(rr) >= 3 else np.nan, human_cv=rr.std() / rr.mean() if len(rr) >= 3 else np.nan))
H = ss.rename(columns={'start_s': 'start_s'}).merge(pd.DataFrame(hr), on=['Trial', 'start_s'], how='left'); H.to_csv(WORK + 'strip_summary_hr.csv', index=False)
# --- detector vs expert per strip ---------------------------------------------
rows = []
for tr, g in ss.groupby('Trial'):
    b = belt_csv(tr); tt = b.time_s.values; x = b.ECG.values.astype(float); mv = moving_flag(b)
    r1 = tt[v2.detect(x, 100, **P)]; r2 = tt[v2.detect(x, 100, lo=P['lo'], hi=P['hi'], W_ms=P['W_ms'], beta=0.5, searchback=False, clip=P['clip'], tau_s=P['tau_s'], dead_s=P['dead_s'])]
    hm = mb[mb.Trial == tr]
    for s in g.itertuples():
        a, e = s.start_s, s.start_s + 8; h = np.sort(hm.loc[hm.strip_start_s == a, 'peak_time_s'].values)
        d1 = r1[(r1 >= a) & (r1 < e)]; d2 = r2[(r2 >= a) & (r2 < e)]
        q = v2.rr_quality((d1 * 100).astype(int), 100) if len(d1) >= 4 else dict(hr=np.nan, rrcv=np.nan, ok=False)
        bs = f1(d1, d2); m = (tt >= a) & (tt < e)
        rows.append(dict(Trial=tr, start_s=a, quality=s.quality, confidence=s.confidence, n_human=len(h), n_det=len(d1), det_hr=q['hr'], det_cv=q['rrcv'], gate_rr=q['ok'],
                         bsqi=bs, gate_full=bool(q['ok'] and bs >= BSQI_MIN), moving=mv[m].mean(), F1=f1(d1, h) if len(h) else np.nan))
S = pd.DataFrame(rows); S.to_csv(WORK + 'strips_gate_vs_human.csv', index=False)
rd = S[S.quality == 'readable']; un = S[S.quality == 'unreadable']
print('readable F1 mean %.4f; RR-gate sens %.3f spec %.3f; full-gate sens %.3f spec %.3f' % (rd.F1.mean(), rd.gate_rr.mean(), 1 - un.gate_rr.mean(), rd.gate_full.mean(), 1 - un.gate_full.mean()))
