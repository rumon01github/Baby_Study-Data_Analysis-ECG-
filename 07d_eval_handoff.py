"""Evaluate detector variants against the full manual-review handoff (1,819 strips, 21,941 beats)."""
import numpy as np, pandas as pd, json, sys
from common import *
from neotex_qrs import detect, rr_quality
import neotex_qrs_v2 as v2

P = json.load(open('neotex_qrs_params.json'))
mb = pd.read_csv(WORK + 'manual_beats_clean.csv')
ss = pd.read_csv(WORK + 'strips_gate_vs_human.csv')[['Trial', 'start_s', 'quality', 'confidence', 'n_human', 'moving']]
VAR = {'v1': lambda x: detect(x, 100, **P), 'v2': lambda x: v2.detect(x, 100, **P)}
rows = []
for tr, g in ss.groupby('Trial'):
    b = belt_csv(tr); tt = b.time_s.values; x = b.ECG.values
    det = {k: tt[f(x)] for k, f in VAR.items()}
    hm = mb[mb.Trial == tr]
    for _, s in g.iterrows():
        a, e = s.start_s, s.start_s + 8
        h = np.sort(hm.loc[(hm.strip_start_s == a), 'peak_time_s'].values)
        r = dict(Trial=tr, start_s=a, quality=s.quality, confidence=s.confidence, n_human=len(h), moving=s.moving)
        for k in VAR:
            d = det[k][(det[k] >= a) & (det[k] < e)]
            q = rr_quality((d * 100).astype(int), 100) if len(d) >= 4 else dict(hr=np.nan, rrcv=np.nan, ok=False)
            r[f'n_{k}'] = len(d); r[f'F1_{k}'] = f1(d, h, 0.075) if len(h) else np.nan; r[f'hr_{k}'] = q['hr']; r[f'ok_{k}'] = q['ok']
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(WORK + 'handoff_eval.csv', index=False)
R['P'] = R.Trial.str[:3]; R['motion'] = pd.cut(R.moving, [-0.01, 0.10, 0.50, 1.01], labels=['Still', 'Light', 'Active'])
rd = R[R.quality == 'readable']
for k in VAR:
    print(f'\n== {k} ==  readable strips n={len(rd)}: F1 mean {rd[f"F1_{k}"].mean():.4f} median {rd[f"F1_{k}"].median():.3f} | F1<0.9: {(rd[f"F1_{k}"]<0.9).sum()} | n_det==0: {(rd[f"n_{k}"]==0).sum()} | under(<0.9 n_h): {(rd[f"n_{k}"]<0.9*rd.n_human).sum()} over(>1.1): {(rd[f"n_{k}"]>1.1*rd.n_human).sum()}')
    print('  by motion:', rd.groupby('motion', observed=True)[f'F1_{k}'].mean().round(4).to_dict())
    print('  by infant:', rd.groupby('P')[f'F1_{k}'].mean().round(3).to_dict())
    pr = R[R.quality == 'partly_readable']; un = R[R.quality == 'unreadable']
    print(f'  partly_readable F1 mean {pr[f"F1_{k}"].mean():.3f} | unreadable strips passing RR gate: {un[f"ok_{k}"].sum()}/{len(un)} | readable passing RR gate: {rd[f"ok_{k}"].sum()}/{len(rd)}')
