"""Figure 4: Fig4_Reliability_vs_motion_and_events. Run after steps 1-10; writes PDF/PNG to work/figures/."""
"""Fig 4 (new): reliability vs motion, split by belt contact; and by video event type in contact-good trials."""
from pstyle import *
from scipy import stats
Q = pd.read_csv(SC + 'quadrants.csv'); X = Q[Q.In_comparison].copy()
bins = np.array([0, 5, 10, 20, 30, 40, 50, 60, 75, 100]); X['bin'] = pd.cut(X['Moving_%'], bins, include_lowest=True)
rng = np.random.default_rng(1)
W, H = COL2, 240
fig = go.Figure(layout=base_layout(W, H))
XA = [(0.06, 0.29), (0.335, 0.565), (0.79, 0.93)]; YD = [0.17, 0.78]
def band(xa, ya, mids, lo, hi, col):
    fig.add_trace(go.Scatter(x=list(mids) + list(mids)[::-1], y=list(hi) + list(lo)[::-1], fill='toself', mode='lines', fillcolor=col, opacity=0.15, line=dict(width=0), xaxis=xa, yaxis=ya, showlegend=False, hoverinfo='skip'))
def passrate(sub, xa, ya):
    ends = {}
    for col, c, lab in [('BIOPAC_good', C['bio'], 'BIOPAC'), ('Belt_good', C['belt'], 'Belt')]:
        mids, mu, lo, hi = [], [], [], []
        for iv, g in sub.groupby('bin', observed=True):
            if len(g) < 12: continue
            tr = g.Trial.unique(); bs = []
            for _ in range(400):
                s_ = rng.choice(tr, len(tr)); bs.append(pd.concat([g[g.Trial == t] for t in s_])[col].mean())
            mids.append(iv.mid); mu.append(g[col].mean() * 100); lo.append(np.percentile(bs, 2.5) * 100); hi.append(np.percentile(bs, 97.5) * 100)
        band(xa, ya, mids, lo, hi, c)
        fig.add_trace(go.Scatter(x=mids, y=mu, mode='lines+markers', line=dict(color=c, width=1.2), marker=dict(size=4.5, color=c, line=dict(color='white', width=0.6)), xaxis=xa, yaxis=ya, showlegend=False, hoverinfo='skip'))
        ends[lab] = (mids[-1], mu[-1])
    up = 'Belt' if ends['Belt'][1] >= ends['BIOPAC'][1] else 'BIOPAC'
    for lab, c in [('BIOPAC', C['bio']), ('Belt', C['belt'])]:
        x_, y_ = ends[lab]; above = (lab == up) or (y_ < 10)
        fig.add_annotation(x=x_ - 3, y=min(97, y_ + 5) if above else max(3, y_ - 5), xref=xa, yref=ya, text=lab, showarrow=False, font=dict(size=SZ['annot'], color=c), xanchor='right', yanchor='bottom' if above else 'top')
CG = X[X.Belt_contact_class == 'good']; CP = X[X.Belt_contact_class != 'good']
passrate(CG, 'x', 'y'); passrate(CP, 'x2', 'y2')
fig.update_layout(xaxis=ax(domain=list(XA[0]), range=[0, 100], tickvals=[0, 25, 50, 75, 100], title_text='Time infant moving (% of window)', anchor='y'),
                  yaxis=ax(domain=YD, range=[0, 100], tickvals=[0, 25, 50, 75, 100], title_text='Windows with reliable HR (%)', anchor='x'),
                  xaxis2=ax(domain=list(XA[1]), range=[0, 100], tickvals=[0, 25, 50, 75, 100], title_text='Time infant moving (% of window)', anchor='y2'),
                  yaxis2=ax(domain=YD, range=[0, 100], tickvals=[0, 25, 50, 75, 100], showticklabels=False, anchor='x2'))
ptitle(fig, XA[0][0], YD[1] + 0.012, '<b>Belt fits well</b>'); letter(fig, 0.0, YD[1] + 0.008, 'a')
fig.add_annotation(x=0.98, y=0.04, xref='x domain', yref='y domain', text=f'{CG.Trial.nunique()} trials<br>{len(CG)} windows', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='right', yanchor='bottom', align='right')
ptitle(fig, XA[1][0], YD[1] + 0.012, '<b>Belt fits poorly</b>'); letter(fig, XA[1][0] - 0.03, YD[1] + 0.008, 'b')
fig.add_annotation(x=0.98, y=0.96, xref='x2 domain', yref='y2 domain', text=f'{CP.Trial.nunique()} trials<br>{len(CP)} windows', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='right', yanchor='top', align='right')
# c: by event type, contact-good trials with validated video
e = pd.read_csv(SC + 'events.csv'); cls = Q.groupby('Trial').Belt_contact_class.first(); e = e[e.Trial.map(cls) == 'good']
labs = [('minimal_visible_movement', 'Quiet'), ('head_movement', 'Head movement'), ('limb_movement', 'Limb movement'), ('wire_contact', 'Wire contact'), ('caregiver_handling', 'Caregiver handling'), ('crying', 'Crying'), ('ALL', 'All events')]
y = np.arange(len(labs))[::-1]; XC = [0.735, 0.895]
for yi, (l, name) in zip(y, labs):
    g = e if l == 'ALL' else e[e.Label == l]; pb, po = g.Belt_good.mean() * 100, g.BIOPAC_good.mean() * 100
    b = int((g.Belt_good & ~g.BIOPAC_good).sum()); c = int((~g.Belt_good & g.BIOPAC_good).sum()); p = stats.binomtest(b, b + c).pvalue if b + c else np.nan
    fig.add_shape(type='line', x0=po, x1=pb, y0=yi, y1=yi, xref='x3', yref='y3', line=dict(color='#BDBDBD', width=1.6))
    tie = abs(po - pb) < 1.0
    for v, col, sz in [(po, C['bio'], 9 if tie else 6), (pb, C['belt'], 6)]:
        fig.add_trace(go.Scatter(x=[v], y=[yi], mode='markers', marker=dict(size=sz, color=col, line=dict(color='white', width=0.6)), xaxis='x3', yaxis='y3', showlegend=False, hoverinfo='skip'))
    ps = 'p < 0.001' if p < 0.001 else f'p = {p:.2f}'
    fig.add_annotation(x=XC[1] + 0.005, y=yi, xref='paper', yref='y3', text=f'<b>{ps}</b>' if p < 0.05 else ps, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='middle')
    if l == 'ALL': fig.add_shape(type='line', x0=40, x1=100, y0=yi + 0.5, y1=yi + 0.5, xref='x3', yref='y3', line=dict(color='#DDDDDD', width=0.6))
fig.update_layout(xaxis3=ax(domain=XC, range=[40, 100], tickvals=[40, 60, 80, 100], title_text='Events with reliable HR (%)', anchor='y3'),
                  yaxis3=ax(domain=YD, range=[-0.6, 6.6], tickvals=list(y), ticktext=[f'{n} ({len(e if l == "ALL" else e[e.Label == l])})' for l, n in labs], ticks='', showline=False, anchor='x3'))
ptitle(fig, XC[0] - 0.19, YD[1] + 0.012, f'<b>By video event, belt fits well</b><br><span style="font-size:6.8px;color:{INK2}">{e.Trial.nunique()} trials with aligned video, {len(e)} events</span>'); letter(fig, XC[0] - 0.215, YD[1] + 0.008, 'c')
fig.add_annotation(x=XC[1] + 0.005, y=y[0] + 0.5, xref='paper', yref='y3', text='McNemar', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='left', yanchor='bottom')
# legend
for i, (col, lab) in enumerate([(C['bio'], 'BIOPAC (wired reference)'), (C['belt'], 'NeoTex belt')]):
    x0 = 0.0 + i * 0.24
    fig.add_shape(type='line', x0=x0, x1=x0 + 0.025, y0=0.975, y1=0.975, xref='paper', yref='paper', line=dict(color=col, width=1.4))
    fig.add_annotation(x=x0 + 0.03, y=0.975, xref='paper', yref='paper', text=lab, showarrow=False, font=dict(size=SZ['legend'], color=INK), xanchor='left', yanchor='middle')
fig.add_annotation(x=0.0, y=0.925, xref='paper', yref='paper', text='Bands: 95% CI, bootstrap over trials. Fit class from the belt\'s own reliable-window share: good ≥ 60%, poor ≤ 20% (partial in between).', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='left', yanchor='middle')
print('events contact-good', len(e), e.Trial.unique(), e.Belt_good.mean(), e.BIOPAC_good.mean())
save(fig, 'Fig4_Reliability_vs_motion_and_events')
