"""Step 5. Event-level and window-level statistics quoted in the paper: McNemar on discordant events, trial-cluster bootstrap,
per-infant Wilcoxon, quiet->moving drop per trial, contact-class splits, both-reliable HR agreement."""
import pandas as pd, numpy as np
from scipy import stats
from config import *
q = pd.read_csv(WORK + 'quadrants.csv'); e = pd.read_csv(WORK + 'events.csv'); c = q[q.In_comparison]
def mcnemar(g):
    b = int((g.Belt_good & ~g.BIOPAC_good).sum()); cc = int((~g.Belt_good & g.BIOPAC_good).sum())
    return b, cc, (stats.binomtest(b, b + cc).pvalue if b + cc else np.nan)
print('== events (validated trials):', len(e), e.Trial.nunique(), 'belt %.1f%% BIOPAC %.1f%%' % (100 * e.Belt_good.mean(), 100 * e.BIOPAC_good.mean()), mcnemar(e))
for lab, g in e.groupby('Label'): print('  ', lab, len(g), round(100 * g.Belt_good.mean(), 1), round(100 * g.BIOPAC_good.mean(), 1), mcnemar(g))
for cls, g in e.groupby('Belt_contact_class'): print('  contact', cls, g.Trial.nunique(), len(g), round(100 * g.Belt_good.mean(), 1), round(100 * g.BIOPAC_good.mean(), 1), mcnemar(g))
rng = np.random.default_rng(0); trials = e.Trial.unique(); ds = []
for _ in range(5000):
    s = pd.concat([e[e.Trial == t] for t in rng.choice(trials, len(trials))]); ds.append(s.Belt_good.mean() - s.BIOPAC_good.mean())
print('  trial-bootstrap diff', round(100 * (e.Belt_good.mean() - e.BIOPAC_good.mean()), 1), np.round(100 * np.percentile(ds, [2.5, 97.5]), 1))
# quiet -> moving drop per trial
mov = ['limb_movement', 'head_movement', 'caregiver_handling', 'wire_contact', 'crying', 'trunk_repositioning']
E12 = pd.read_csv(WORK + 'events12.csv'); E12['cls'] = np.where(E12.Label.isin(mov), 'm', np.where(E12.Label == 'minimal_visible_movement', 'q', 'o'))
drops = []
for tr, g in E12.groupby('Trial'):
    qq, mm = g[g.cls == 'q'], g[g.cls == 'm']
    if len(qq) >= 5 and len(mm) >= 10: drops.append((tr, 100 * (qq.Belt_good.mean() - mm.Belt_good.mean()), 100 * (qq.BIOPAC_good.mean() - mm.BIOPAC_good.mean())))
D = pd.DataFrame(drops, columns=['Trial', 'belt_drop', 'bio_drop']); print('== quiet->moving drop (points), median belt %.1f BIOPAC %.1f; drops>5: belt %d/%d BIOPAC %d/%d' % (D.belt_drop.median(), D.bio_drop.median(), (D.belt_drop > 5).sum(), len(D), (D.bio_drop > 5).sum(), len(D)))
bg = e[e.Belt_good & e.BIOPAC_good]; dd = bg.Belt_HR - bg.BIOPAC_HR
print('== both-reliable events', len(bg), 'bias %.2f LoA %.1f..%.1f within5 %.1f%%' % (dd.mean(), dd.mean() - 1.96 * dd.std(), dd.mean() + 1.96 * dd.std(), 100 * (dd.abs() <= 5).mean()))
# windows
print('== windows in comparison', len(c), c.Trial.nunique(), 'belt %.1f%% BIOPAC %.1f%%' % (100 * c.Belt_good.mean(), 100 * c.BIOPAC_good.mean()))
for cls, g in c.groupby('Belt_contact_class'):
    print('  contact', cls, g.Trial.nunique(), len(g), g.groupby('Motion')[['Belt_good', 'BIOPAC_good']].mean().round(3).to_dict())
g = c[c.Belt_contact_class == 'good'].groupby('Trial')[['Belt_good', 'BIOPAC_good']].mean()
print('  contact-good per-trial Wilcoxon p', stats.wilcoxon(g.Belt_good, g.BIOPAC_good).pvalue, 'belt ahead', int((g.Belt_good > g.BIOPAC_good).sum()), '/', len(g))
c60 = pd.read_csv(WORK + 'contact_first60.csv').set_index('Trial').contact_first60; nc = c[(c.Start_s >= 60)].copy(); nc['c60'] = nc.Trial.map(c60)
for cls, g in nc.groupby('c60'): print('  60-s check', cls, g.Trial.nunique(), 'belt %.1f%% BIOPAC %.1f%%' % (100 * g.Belt_good.mean(), 100 * g.BIOPAC_good.mean()))
gg = nc[nc.c60 == 'good'].groupby('Trial')[['Belt_good', 'BIOPAC_good']].mean(); print('  60-s good per-trial Wilcoxon p', stats.wilcoxon(gg.Belt_good, gg.BIOPAC_good).pvalue, int((gg.Belt_good > gg.BIOPAC_good).sum()), '/', len(gg))
bgw = c[c.Belt_good & c.BIOPAC_good]; dw = bgw.Belt_HR - bgw.BIOPAC_HR
print('== both-reliable windows', len(bgw), 'r %.3f MAE %.2f bias %.2f LoA %.1f..%.1f within5 %.1f%%' % (np.corrcoef(bgw.Belt_HR, bgw.BIOPAC_HR)[0, 1], dw.abs().mean(), dw.mean(), dw.mean() - 1.96 * dw.std(ddof=1), dw.mean() + 1.96 * dw.std(ddof=1), 100 * (dw.abs() <= 5).mean()))
# failure attribution (non-circular)
nc['ctx'] = np.where(nc.c60 != 'good', 'belt contact poor (setup check)', 'contact good · ' + nc.Motion.map({'Still': 'still', 'Light': 'light motion', 'Active': 'active motion'}))
rows = []
for nm, col in [('Belt', 'Belt_good'), ('BIOPAC', 'BIOPAC_good')]:
    f = nc[~nc[col]]; share = 100 * f.ctx.value_counts(normalize=True); rate = 100 * (1 - nc.groupby('ctx')[col].mean())
    for o in rate.index: rows.append(dict(device=nm, context=o, n_windows=int((nc.ctx == o).sum()), fail_rate_pct=round(rate[o], 1), share_of_failures_pct=round(share.get(o, 0), 1)))
FA = pd.DataFrame(rows); FA.to_csv(WORK + 's5_failure_attribution.csv', index=False); print(FA.to_string())
