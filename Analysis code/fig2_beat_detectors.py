"""Figure 2: Fig2_Beat_detectors. Run after steps 1-10; writes PDF/PNG to work/figures/."""
"""Fig 2 (new): which beat detector works on the belt — expert markers vs generic detectors vs belt-specific detector vs learned scorer."""
from pstyle import *
import json, sys
from neotex_qrs import detect as det_v1; import neotex_qrs_v2 as v2
from sklearn.ensemble import HistGradientBoostingClassifier
P1 = json.load(open('neotex_qrs_params.json')); P2 = json.load(open('neotex_qrs_params_v2.json'))
d = cache(); mb = pd.read_csv(SC + 'manual_beats_clean.csv'); R = pd.read_csv(SC + 'detector_table_strips.csv')
F = pd.read_csv(SC + 's3_candidates.csv'); FEAT = ['amp', 'rel_amp', 'm_rel', 'kurt', 'rr_prev', 'rr_ratio', 'tcorr', 'moving', 'is_det']
TOL = 0.075
EX = [('P06 T01', 152.0, 'Vigorous limb movement (65% of time moving)'), ('P04 T04', 56.0, 'After a large artifact (detector lock-up)')]
METH = [('Expert', INK), ('NeuroKit2', '#8C8C8C'), ('Hamilton', '#B5B5B5'), ('Belt v1', C['belt_light']), ('Belt v2', C['belt']), ('Scorer', C['bio'])]

def scorer_beats(tr):
    p = tr[:3]; TR = F[(F.P != p) & (F.quality != 'unreadable')]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(TR[FEAT], TR.label)
    g = F[F.Trial == tr].copy(); g['p'] = clf.predict_proba(g[FEAT])[:, 1]; acc = g[g.p >= 0.5].sort_values('t')
    keep = []; last = -1
    for t in acc.t.values:
        if t - last > 0.2: keep.append(t); last = t
    return np.array(keep)

def all_markers(tr):
    b = belt_csv(tr); tt = b.time_s.values; x = b.ECG.values.astype(float)
    dev = d[tr]['belt']; fs = dev['fs']; t0 = dev['t0']
    return b, {'Expert': np.sort(mb.loc[mb.Trial == tr, 'peak_time_s'].values), 'NeuroKit2': dev['nk'] / fs + t0, 'Hamilton': dev['ham'] / fs + t0,
               'Belt v1': tt[det_v1(x, 100, **P1)], 'Belt v2': tt[v2.detect(x, 100, **P2)], 'Scorer': scorer_beats(tr)}

W, H = COL2, 345
fig = go.Figure(layout=base_layout(W, H))
XS = [(0.075, 0.49), (0.585, 0.985)]; YT = [0.56, 0.93]
ai = 1
for k, (tr, a, title) in enumerate(EX):
    b, M = all_markers(tr); e = a + 8; m = (b.time_s >= a) & (b.time_s < e); t = b.time_s[m].values - a; x = b.ECG[m].values
    xa, ya = ('x', 'y') if ai == 1 else (f'x{ai}', f'y{ai}'); yb = f'y{ai + 1}'
    x = x - np.median(x); amp = float(np.percentile(np.abs(x), 99.8)) * 1.15; x = np.clip(x, -amp, amp)
    fig.add_trace(line(t, x, C['belt'], 0.7, xaxis=xa, yaxis=ya))
    # marker rows
    exp = M['Expert']; exp = exp[(exp >= a) & (exp < e)]
    for j, (nm, col) in enumerate(METH):
        mt = M[nm]; mt = mt[(mt >= a) & (mt < e)]; yv = len(METH) - 1 - j
        fig.add_trace(go.Scatter(x=mt - a, y=np.full(len(mt), yv), mode='markers', marker=dict(symbol='line-ns', size=7, line=dict(width=1.1, color=col)), xaxis=xa, yaxis=yb, showlegend=False, hoverinfo='skip'))
        # missed expert beats (no marker within tol) -> hollow red circle
        if nm != 'Expert' and len(exp):
            miss = [v for v in exp if len(mt) == 0 or np.min(np.abs(mt - v)) > TOL]
            fig.add_trace(go.Scatter(x=np.array(miss) - a, y=np.full(len(miss), yv), mode='markers', marker=dict(symbol='circle-open', size=6, color=C['fail'], line=dict(width=0.9, color=C['fail'])), xaxis=xa, yaxis=yb, showlegend=False, hoverinfo='skip'))
            f1v = R[(R.Trial == tr) & (R.start_s == a)][{'NeuroKit2': 'F1_nk', 'Hamilton': 'F1_ham', 'Belt v1': 'F1_v1', 'Belt v2': 'F1_v2', 'Scorer': 'F1_sc'}[nm]].iloc[0]
            fig.add_annotation(x=0.995, y=yv, xref=f'{xa} domain', yref=yb, text=f'F1 {f1v:.2f}', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='right', yanchor='middle', bgcolor='rgba(255,255,255,0.9)', borderpad=1)
    fig.update_layout({f'xaxis{ai}' if ai > 1 else 'xaxis': ax(domain=list(XS[k]), range=[0, 8], dtick=2, title_text='Time (s)', anchor=yb),
                       f'yaxis{ai}' if ai > 1 else 'yaxis': ax(domain=[YT[0] + 0.19, YT[1]], range=[-amp * 0.75, amp * 1.05], tickvals=[0, 0.5] if amp > 0.6 else [0, 0.2], title_text='Belt ECG (mV)', anchor=xa),
                       f'yaxis{ai + 1}': ax(domain=[YT[0], YT[0] + 0.175], range=[-0.7, len(METH) - 0.3], tickvals=list(range(len(METH))), ticktext=[m[0] for m in METH][::-1], ticks='', showline=False, anchor=xa, tickfont=dict(size=SZ['annot']))})
    ptitle(fig, XS[k][0], YT[1] + 0.008, f'<b>{title}</b>'); fig.add_annotation(x=0.99, y=0.98, xref=f'{xa} domain', yref=f'{ya} domain', text=f'{tr} · {a:.0f}–{a + 8:.0f} s', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='right', yanchor='top')
    letter(fig, XS[k][0] - 0.055, YT[1] + 0.004, 'ab'[k])
    ai += 2
# c: F1 by strip quality, 5 methods (42 paired trials)
r = R[R.F1_nk.notna()]; qual = ['readable', 'partly_readable', 'uncertain']; qn = ['Read-<br>able', 'Partly<br>readable', 'Uncer-<br>tain']
cols = {'NeuroKit2': 'F1_nk', 'Hamilton': 'F1_ham', 'Belt v1': 'F1_v1', 'Belt v2': 'F1_v2', 'Scorer': 'F1_sc'}
YB = [0.125, 0.42]; XB = [(0.075, 0.49), (0.585, 0.985)]
for j, (nm, col) in enumerate(METH[1:]):
    vals = [r[r.quality == q][cols[nm]].mean() for q in qual]
    fig.add_trace(go.Bar(x=np.arange(3) + (j - 2) * 0.17, y=vals, width=0.155, marker=dict(color=col, line=dict(width=0)), xaxis='x5', yaxis='y5', showlegend=False, hoverinfo='skip'))
    for i, v in enumerate(vals):
        fig.add_annotation(x=i + (j - 2) * 0.17, y=v + 0.01, xref='x5', yref='y5', text=f'{v:.2f}', showarrow=False, font=dict(size=SZ['annot'] - 0.8, color=INK), xanchor='center', yanchor='bottom', textangle=-90)
nq = [int((r.quality == q).sum()) for q in qual]
fig.update_layout(xaxis5=ax(domain=list(XB[0]), range=[-0.55, 2.55], tickvals=[0, 1, 2], ticktext=[f'{a}<br>({n})' for a, n in zip(qn, nq)], tickangle=0, anchor='y5'),
                  yaxis5=ax(domain=YB, range=[0.5, 1.1], tickvals=[0.5, 0.75, 1.0], title_text='Beat F1 vs. expert (±75 ms)', anchor='x5'))
ptitle(fig, XB[0][0], YB[1] + 0.008, f'<b>Agreement with expert by strip quality</b> ({r.Trial.nunique()} trials)'); letter(fig, XB[0][0] - 0.055, YB[1] + 0.004, 'c')
# d: per infant, readable strips: NeuroKit2 vs Belt v2 vs Scorer
rd = r[r.quality == 'readable']; pi = rd.groupby(rd.Trial.str[:3])[['F1_nk', 'F1_v2', 'F1_sc']].mean(); inf = list(pi.index)
for i, p in enumerate(inf):
    fig.add_shape(type='line', x0=i, x1=i, y0=pi.loc[p, 'F1_nk'], y1=pi.loc[p, 'F1_v2'], xref='x6', yref='y6', line=dict(color='#C8C8C8', width=1.0))
for colname, col, nm in [('F1_nk', '#8C8C8C', 'NeuroKit2'), ('F1_v2', C['belt'], 'Belt v2'), ('F1_sc', C['bio'], 'Scorer')]:
    fig.add_trace(go.Scatter(x=list(range(len(inf))), y=pi[colname].values, mode='markers', marker=dict(size=5.5 if colname != 'F1_sc' else 4, color=col, symbol='circle' if colname != 'F1_sc' else 'diamond', line=dict(color='white', width=0.6)), xaxis='x6', yaxis='y6', showlegend=False, hoverinfo='skip'))
fig.update_layout(xaxis6=ax(domain=list(XB[1]), range=[-0.6, len(inf) - 0.4], tickvals=list(range(len(inf))), ticktext=inf, title_text='Infant', anchor='y6'),
                  yaxis6=ax(domain=YB, range=[0.82, 1.005], tickvals=[0.85, 0.9, 0.95, 1.0], title_text='Beat F1, readable strips', anchor='x6'))
ptitle(fig, XB[1][0], YB[1] + 0.008, '<b>Per infant</b> (readable strips)'); letter(fig, XB[1][0] - 0.055, YB[1] + 0.004, 'd')
# legend
xpos = 0.0; yl = 0.985
for nm, col in METH[1:]:
    fig.add_shape(type='rect', x0=xpos, x1=xpos + 0.018, y0=yl - 0.016, y1=yl + 0.016, xref='paper', yref='paper', fillcolor=col, line=dict(width=0))
    fig.add_annotation(x=xpos + 0.022, y=yl, xref='paper', yref='paper', text=nm, showarrow=False, font=dict(size=SZ['legend'], color=INK), xanchor='left', yanchor='middle')
    xpos += 0.022 + 0.0074 * len(nm) + 0.02
fig.add_shape(type='circle', x0=xpos, x1=xpos + 0.012, y0=yl - 0.012 * W / H, y1=yl + 0.012 * W / H, xref='paper', yref='paper', line=dict(color=C['fail'], width=0.9))
fig.add_annotation(x=xpos + 0.016, y=yl, xref='paper', yref='paper', text='Expert beat missed by that method', showarrow=False, font=dict(size=SZ['legend'], color=INK), xanchor='left', yanchor='middle')
save(fig, 'Fig2_Beat_detectors')
