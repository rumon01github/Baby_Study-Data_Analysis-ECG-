"""Step 2. Per 10-s window: reliability gate for both devices (belt aligned to BIOPAC by a per-window lag), HR, motion level;
trial inclusion rule; belt contact class; non-circular 60-s contact check.
Outputs: work/quadrants.csv, work/trial_table_v2.csv, work/contact_first60.csv"""
import os, numpy as np, pandas as pd
from common import *
d = cache(); meta = pd.read_csv('trial_metadata.csv').set_index('Trial')   # Trial, Participant, Condition
def condition(tr):
    if tr in meta.index: return meta.Condition[tr]
    name = os.path.basename(belt_path(tr)); return 'ParentHeld' if 'ParentHeld' in name else ('Independent' if 'Independent' in name else '')
rows = []
for tr, D in d.items():
    b = belt_csv(tr); mv = moving_flag(b); T = D['T']
    for a in np.arange(0, T, WIN):          # last window may be shorter than WIN (as in the original analysis)
        L = local_lag(beats(D['belt']), beats(D['bio']), a, a + WIN)
        sb = sqi(D['bio'], a, a + WIN); st = sqi(D['belt'], a + L, a + WIN + L)
        m = (b.time_s >= a) & (b.time_s < a + WIN); share = mv[m].mean() if m.any() else np.nan
        rows.append(dict(Trial=tr, Participant=tr[:3], Condition=condition(tr), Start_s=a, Motion=motion_class(share), **{'Moving_%': 100 * share},
                         Belt_good=st['good'], BIOPAC_good=sb['good'], Belt_HR=st['hr'], BIOPAC_HR=sb['hr'],
                         Belt_bSQI=st['bsqi'], Belt_tSQI=st['tsqi'], Belt_RRcv=st['rrcv'], Bio_bSQI=sb['bsqi'], Bio_tSQI=sb['tsqi'], Bio_RRcv=sb['rrcv'], Lag_s=L))
q = pd.DataFrame(rows)
q['Quadrant'] = np.select([q.Belt_good & q.BIOPAC_good, q.Belt_good, q.BIOPAC_good], ['Both good', 'Belt only', 'BIOPAC only'], 'Neither')
# trial inclusion + contact class (whole trial) ---------------------------------
g = q.groupby('Trial').agg(Windows=('Start_s', 'size'), BIOPAC_reliable=('BIOPAC_good', 'sum'), Belt_reliable=('Belt_good', 'sum'), Active_pct=('Motion', lambda s: 100 * (s == 'Active').mean()))
g['BIOPAC_reliable_pct'] = 100 * g.BIOPAC_reliable / g.Windows; g['Belt_contact_pct'] = 100 * g.Belt_reliable / g.Windows
g['In_comparison'] = g.BIOPAC_reliable >= MIN_REF_WINDOWS
g['Belt_contact_class'] = np.select([g.Belt_contact_pct >= 100 * CONTACT_GOOD, g.Belt_contact_pct <= 100 * CONTACT_POOR], ['good', 'poor'], 'partial')
q = q.merge(g[['In_comparison', 'Belt_contact_class', 'Belt_contact_pct']], left_on='Trial', right_index=True)
bg = q[q.Belt_good & q.BIOPAC_good]; g['Both_reliable_n'] = bg.groupby('Trial').size(); g['HR_MAE_both'] = (bg.Belt_HR - bg.BIOPAC_HR).abs().groupby(bg.Trial).mean().round(2)
g['Participant'] = [t[:3] for t in g.index]; g['Condition'] = [condition(t) for t in g.index]; g = g.reset_index()
q.to_csv(WORK + 'quadrants.csv', index=False); g.to_csv(WORK + 'trial_table_v2.csv', index=False)
# non-circular: class from the first 60 s only ---------------------------------
c60 = q[q.Start_s < 60].groupby('Trial').Belt_good.mean()
pd.DataFrame({'Trial': c60.index, 'contact_first60': np.where(c60 >= 0.5, 'good', 'poor')}).to_csv(WORK + 'contact_first60.csv', index=False)   # good if >= 3 of the first 6 windows reliable
print(len(q), 'windows;', g.In_comparison.sum(), 'trials in comparison;', g.Belt_contact_class.value_counts().to_dict())
