"""S3 step 4-5: leave-one-infant-out beat scorer + learning curve."""
import numpy as np, pandas as pd, sys
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, '.'); from common import *
F = pd.read_csv(WORK + 's3_candidates.csv'); mb = pd.read_csv(WORK + 'manual_beats_clean.csv')
FEAT = ['amp', 'rel_amp', 'm_rel', 'kurt', 'rr_prev', 'rr_ratio', 'tcorr', 'moving', 'is_det']
TR = F[F.quality != 'unreadable']               # training mask: unreadable strips excluded
inf = sorted(F.P.unique()); rng = np.random.default_rng(0)
def mk(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
def strip_f1(sub, col, thr=0.5):
    """per-strip F1 of accepted candidates (p>=thr with 200-ms non-max suppression) vs human markers"""
    out = []
    for (tr, a, q), g in sub.groupby(['Trial', 'start_s', 'quality']):
        acc = g[g[col] >= thr].sort_values('t'); keep = []; last = -1
        for _, r in acc.iterrows():
            if r.t - last > 0.2: keep.append(r.t); last = r.t
        h = np.sort(mb.loc[(mb.Trial == tr) & (mb.strip_start_s == a), 'peak_time_s'].values)
        out.append(dict(Trial=tr, P=tr[:3], start_s=a, quality=q, n_h=len(h), n_acc=len(keep), F1=f1(np.array(keep), h, 0.075) if len(h) else np.nan))
    return pd.DataFrame(out)
# ---- LOIO
F['p'] = np.nan
for p in inf:
    tr = TR[TR.P != p]; te = F.P == p
    clf = mk().fit(tr[FEAT], tr.label); F.loc[te, 'p'] = clf.predict_proba(F.loc[te, FEAT])[:, 1]
F['p_det'] = F.is_det.astype(float)
S_sc = strip_f1(F, 'p'); S_v2 = strip_f1(F, 'p_det')
M = S_v2.merge(S_sc, on=['Trial', 'P', 'start_s', 'quality', 'n_h'], suffixes=('_v2', '_sc'))
M.to_csv(WORK + 's3_strip_results.csv', index=False)
print('LOIO strip F1 by quality:'); print(M.groupby('quality')[['F1_v2', 'F1_sc']].mean().round(4))
print('readable by infant:'); print(M[M.quality == 'readable'].groupby('P')[['F1_v2', 'F1_sc']].mean().round(3).T)
rd = M[M.quality == 'readable']; print('readable F1<0.9: v2', (rd.F1_v2 < 0.9).sum(), 'scorer', (rd.F1_sc < 0.9).sum(), ' zero-beat strips: v2', (rd.n_acc_v2 == 0).sum(), 'scorer', (rd.n_acc_sc == 0).sum())
# beat-level Se/PPV on readable+partly strips (LOIO)
for q in ['readable', 'partly_readable']:
    g = F[F.quality == q]; y = g.label
    for col, nm in [('p_det', 'v2'), ('p', 'scorer')]:
        yp = g[col] >= 0.5; tp = (yp & (y == 1)).sum(); print(f'{q:16s} {nm:7s} Se {tp / (y == 1).sum():.4f} PPV {tp / max(1, yp.sum()):.4f}')
# false positives on unreadable strips (any accepted candidate is wrong)
u = F[F.quality == 'unreadable']; print('unreadable: accepted per strip v2 %.2f scorer %.2f' % ((u.p_det >= 0.5).groupby([u.Trial, u.start_s]).sum().mean(), (u.p >= 0.5).groupby([u.Trial, u.start_s]).sum().mean()))
# ---- learning curve: train on fraction of strips (readable+partly+uncertain), LOIO, test readable+partly strips
strips = TR[['Trial', 'start_s']].drop_duplicates().reset_index(drop=True)
LC = []
for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
    for rep in range(3 if frac < 1 else 1):
        pick = strips.sample(frac=frac, random_state=rep * 10 + int(frac * 100)); key = set(map(tuple, pick.values))
        Fp = F.copy(); Fp['p'] = np.nan
        for p in inf:
            tr = TR[(TR.P != p) & [ (a, b) in key for a, b in zip(TR.Trial, TR.start_s)]]
            if tr.label.nunique() < 2: continue
            te = Fp.P == p; clf = mk().fit(tr[FEAT], tr.label); Fp.loc[te, 'p'] = clf.predict_proba(Fp.loc[te, FEAT])[:, 1]
        S = strip_f1(Fp[Fp.quality.isin(['readable', 'partly_readable'])], 'p')
        LC.append(dict(frac=frac, rep=rep, n_strips=len(pick), n_beats=int(tr.label.sum() * len(inf) / (len(inf) - 1)), F1_readable=S[S.quality == 'readable'].F1.mean(), F1_partly=S[S.quality == 'partly_readable'].F1.mean()))
        print(LC[-1])
pd.DataFrame(LC).to_csv(WORK + 's3_learning_curve.csv', index=False)
# feature importance (permutation, on full model, last fold as proxy)
from sklearn.inspection import permutation_importance
clf = mk().fit(TR[FEAT], TR.label); pi = permutation_importance(clf, TR[FEAT].sample(20000, random_state=0), TR.label.sample(20000, random_state=0), n_repeats=5, random_state=0)
print(pd.Series(pi.importances_mean, FEAT).sort_values(ascending=False).round(4))
