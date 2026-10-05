"""Step 4. Video-annotated events in data time, gated per device with the same 10-s-style gate applied to the event interval.
Uses ecg_start_s/ecg_end_s written into the annotation files (video time + offset from step 3). Events shorter than MIN_EVENT s are
widened symmetrically. Outputs: work/events12.csv (all resolved trials), work/events.csv (trials with validated alignment)."""
import os, numpy as np, pandas as pd, glob
from common import *
MIN_EVENT = 5.0
VALID = ['P01 T01', 'P03 T01', 'P03 T02', 'P04 T02', 'P04 T03', 'P06 T04', 'P09 T03']   # alignment significant / landmark-checked; P01 T01 flagged (+-3 s)
OFF = pd.read_csv('video_offsets_final.csv'); OFF = OFF[OFF.lag_s.notna()].set_index('Trial').lag_s   # video time + lag_s = data time
d = cache(); q = pd.read_csv(WORK + 'quadrants.csv'); cls = q.groupby('Trial')[['In_comparison', 'Belt_contact_class']].first()
rows = []
files = sorted(os.path.join(r, f) for r, _, fs in os.walk(VIDEO) for f in fs if f.endswith('_annotations.csv'))   # any depth, any separator
print(f'{len(files)} annotation files under {VIDEO}')
for f in files:
    a = pd.read_csv(f); tr = f'{a.participant.iloc[0]} {a.trial.iloc[0]}'
    if tr not in d: print('  skip', tr, '(not in beats cache)'); continue
    if a.ecg_start_s.isna().all():                       # annotation file without data-time columns: apply the offset table
        if OFF is None or tr not in OFF.index: continue
        a['ecg_start_s'] = a.video_start_s + OFF[tr]; a['ecg_end_s'] = a.video_end_s + OFF[tr]
    D = d[tr]; T = D['T']
    for r in a.dropna(subset=['ecg_start_s']).itertuples():
        s, e = r.ecg_start_s, r.ecg_end_s
        if e - s < MIN_EVENT: c = (s + e) / 2; s, e = c - MIN_EVENT / 2, c + MIN_EVENT / 2
        s, e = max(0, s), min(T, e)
        if e - s < 3: continue
        L = local_lag(beats(D['belt']), beats(D['bio']), s - 5, e + 5)
        sb = sqi(D['bio'], s, e); st = sqi(D['belt'], s + L, e + L)
        rows.append(dict(Trial=tr, Label=r.label, Start=round(s, 2), End=round(e, 2), Dur=round(e - s, 2), Belt_good=st['good'], BIOPAC_good=sb['good'],
                         Belt_HR=st['hr'], BIOPAC_HR=sb['hr'], Belt_bSQI=st['bsqi'], Bio_bSQI=sb['bsqi'], Belt_tSQI=st['tsqi'], Bio_tSQI=sb['tsqi'], Belt_RRcv=st['rrcv'], Bio_RRcv=sb['rrcv']))
if not rows: raise SystemExit('No events built: check that the annotation files have ecg_start_s/ecg_end_s or that video_offsets_final.csv lists their trials.')
E = pd.DataFrame(rows).merge(cls, left_on='Trial', right_index=True, how='left')
E.to_csv(WORK + 'events12.csv', index=False); E[E.Trial.isin(VALID)].to_csv(WORK + 'events.csv', index=False)
print(len(E), 'events in', E.Trial.nunique(), 'trials;', int(E.Trial.isin(VALID).sum()), 'in validated trials')
