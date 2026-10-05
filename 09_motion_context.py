"""S5: motion-context classifier. Window (10 s) features from belt IMU + belt ECG SQIs; labels from video annotations."""
import numpy as np, pandas as pd, sys, json
from scipy.signal import welch
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv(WORK + 'windows_sqi_ext.csv'); E = pd.read_csv(WORK + 'events12.csv')
# windows_sqi_ext.csv already carries every quadrants.csv column (In_comparison, Belt_contact_class, ...)
TRIALS_OK = ['P02 T01', 'P03 T01', 'P03 T02', 'P04 T01', 'P04 T02', 'P04 T03', 'P09 T03']   # video offset significant and not flagged
E = E[E.Trial.isin(TRIALS_OK)]
CLASS = {'minimal_visible_movement': 'quiet', 'limb_movement': 'body', 'head_movement': 'body', 'trunk_repositioning': 'body',
         'wire_contact': 'wire', 'caregiver_handling': 'handling', 'caregiver_contact_candidate': 'handling', 'crying': 'crying'}
ORDER = ['quiet', 'body', 'wire', 'handling', 'crying']
# ---- IMU features per window
rows = []
for tr, g in X.groupby('Trial'):
    b = belt_csv(tr); t = b.time_s.values; gy = b.gyro.values; base = np.percentile(gy, 20)
    acc = b[['AccX', 'AccY', 'AccZ']].values / 4096.0; gyr = b[['GyroX', 'GyroY', 'GyroZ']].values
    for _, w in g.iterrows():
        m = (t >= w.Start_s) & (t < w.Start_s + 10)
        if m.sum() < 500: continue
        a = acc[m]; gg = gyr[m]; am = np.linalg.norm(a, axis=1); gm = gy[m]
        ad = a - a.mean(0); en = (ad ** 2).sum(0); axr = en.max() / en.sum() if en.sum() > 0 else 1 / 3
        f, P = welch(am - am.mean(), fs=100, nperseg=256)
        tot = P[(f >= 0.3) & (f <= 12)].sum() + 1e-12
        fr15 = P[(f >= 1) & (f <= 5)].sum() / tot; fr03 = P[(f >= 0.3) & (f < 1)].sum() / tot; fdom = f[(f >= 0.3)][np.argmax(P[f >= 0.3])]
        rows.append(dict(Trial=tr, Start_s=w.Start_s, acc_std=am.std(), acc_p95=np.percentile(np.abs(am - np.median(am)), 95), gyro_mean=gm.mean() - base,
                         gyro_std=gm.std(), gyro_p95=np.percentile(gm, 95) - base, moving=(gm > base + 200).mean(), axis_ratio=axr, f_1_5=fr15, f_03_1=fr03, f_dom=fdom,
                         grav_change=np.linalg.norm(a[-100:].mean(0) - a[:100].mean(0))))
I = pd.DataFrame(rows)
D = X.merge(I, on=['Trial', 'Start_s'])
# belt amplitude CV from cache beats
import pickle; d = cache()
def amp_cv(tr, a):
    dev = d[tr]['belt']; fs = dev['fs']; t0 = dev['t0']; r = dev['nk']; tt = r / fs + t0; sel = r[(tt >= a) & (tt < a + 10)]
    if len(sel) < 4: return np.nan
    v = np.abs(dev['sig'][sel]); return v.std() / v.mean() if v.mean() > 0 else np.nan
D['amp_cv'] = [amp_cv(tr, a) for tr, a in zip(D.Trial, D.Start_s)]
FE_IMU = ['acc_std', 'acc_p95', 'gyro_mean', 'gyro_std', 'gyro_p95', 'moving', 'axis_ratio', 'f_1_5', 'f_03_1', 'f_dom', 'grav_change']
FE_ECG = ['B_kSQI', 'B_pSQI', 'B_basSQI', 'Belt_tSQI', 'Belt_bSQI', 'Belt_RRcv', 'amp_cv']
FE_REF = ['Bio_bSQI', 'Bio_RRcv', 'O_kSQI', 'O_basSQI']
# ---- labels: dominant annotated class covering >= 50 % of the window
E['cls'] = E.Label.map(CLASS); E = E.dropna(subset=['cls'])
lab = []
for _, w in D.iterrows():
    ev = E[(E.Trial == w.Trial) & (E.End > w.Start_s) & (E.Start < w.Start_s + 10)]
    if len(ev) == 0: lab.append(None); continue
    cov = {}
    for _, e in ev.iterrows():
        cov[e.cls] = cov.get(e.cls, 0) + (min(e.End, w.Start_s + 10) - max(e.Start, w.Start_s))
    k = max(cov, key=cov.get); lab.append(k if cov[k] >= 5 else None)
D['cls'] = lab; D['P'] = D.Trial.str[:3]
D.to_csv(WORK + 's5_windows.csv', index=False)
L = D.dropna(subset=['cls'] + FE_IMU + FE_ECG).copy()
print('labelled windows', len(L), L.cls.value_counts().to_dict(), 'trials', L.Trial.nunique(), 'infants', L.P.nunique())
# ---- (a) unsupervised clustering on belt-only features
Z = StandardScaler().fit_transform(D.dropna(subset=FE_IMU + FE_ECG)[FE_IMU + FE_ECG])
for k in range(2, 7):
    km = KMeans(k, n_init=10, random_state=0).fit(Z); print(f'k={k} silhouette {silhouette_score(Z, km.labels_):.3f}')
km = KMeans(3, n_init=10, random_state=0).fit(Z); Dc = D.dropna(subset=FE_IMU + FE_ECG).copy(); Dc['cluster'] = km.labels_
Lc = Dc.dropna(subset=['cls']); print(pd.crosstab(Lc.cls, Lc.cluster))
print(Dc.groupby('cluster')[['moving', 'Belt_bSQI', 'B_kSQI', 'Belt_good', 'BIOPAC_good']].mean().round(3))
# ---- (b) supervised, leave-one-infant-out
def loio(feats, name):
    yp = pd.Series(index=L.index, dtype=object)
    for p in L.P.unique():
        tr, te = L[L.P != p], L[L.P == p]
        rf = RandomForestClassifier(400, min_samples_leaf=3, class_weight='balanced', random_state=0).fit(tr[feats], tr.cls)
        yp[te.index] = rf.predict(te[feats])
    cm = confusion_matrix(L.cls, yp, labels=ORDER)
    print(f'\n{name}: macro-F1 {f1_score(L.cls, yp, average="macro"):.3f}  balanced acc {balanced_accuracy_score(L.cls, yp):.3f}')
    print(pd.DataFrame(cm, index=['true ' + o for o in ORDER], columns=ORDER))
    rec = {o: cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan for i, o in enumerate(ORDER)}; print('recall', {k: round(v, 2) for k, v in rec.items()})
    return yp, cm
yp_imu, _ = loio(FE_IMU, 'IMU only')
yp_belt, cm_belt = loio(FE_IMU + FE_ECG, 'Belt only (IMU + ECG SQIs)')
yp_all, cm_all = loio(FE_IMU + FE_ECG + FE_REF, 'Belt + BIOPAC discordance (analysis only)')
L['pred_belt'] = yp_belt; L['pred_all'] = yp_all; L.to_csv(WORK + 's5_labelled.csv', index=False)
# 3-class version (quiet / body / handling) which belt-only can be expected to separate
L3 = L[L.cls.isin(['quiet', 'body', 'handling'])]
yp3 = pd.Series(index=L3.index, dtype=object)
for p in L3.P.unique():
    tr, te = L3[L3.P != p], L3[L3.P == p]
    rf = RandomForestClassifier(400, min_samples_leaf=3, class_weight='balanced', random_state=0).fit(tr[FE_IMU + FE_ECG], tr.cls); yp3[te.index] = rf.predict(te[FE_IMU + FE_ECG])
print('\n3-class belt-only: macro-F1 %.3f' % f1_score(L3.cls, yp3, average='macro')); print(pd.DataFrame(confusion_matrix(L3.cls, yp3, labels=['quiet', 'body', 'handling']), index=['quiet', 'body', 'handling'], columns=['quiet', 'body', 'handling']))
# ---- failure attribution on all windows: train belt-only on all labelled windows, predict all 1392
rf = RandomForestClassifier(400, min_samples_leaf=3, class_weight='balanced', random_state=0).fit(L[FE_IMU + FE_ECG], L.cls)
A = D.dropna(subset=FE_IMU + FE_ECG).copy(); A['pred'] = rf.predict(A[FE_IMU + FE_ECG])
A['belt_fail'] = ~A.Belt_good; A['bio_fail'] = ~A.BIOPAC_good
print('\nFailure attribution (predicted context of gate-fail windows, %):')
for dev in ['belt_fail', 'bio_fail']:
    for cc in ['good', 'poor']:
        s = A[(A[dev]) & (A.Belt_contact_class == cc)]
        print(f'  {dev:9s} contact-{cc:4s} n={len(s):4d}', (100 * s.pred.value_counts(normalize=True)).round(1).reindex(ORDER).fillna(0).to_dict())
imp = pd.Series(rf.feature_importances_, FE_IMU + FE_ECG).sort_values(ascending=False); print(imp.round(3).head(8))
A.to_csv(WORK + 's5_all_pred.csv', index=False)
