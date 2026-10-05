"""Figure 5: Fig5_Event_and_window_quality. Run after steps 1-10; writes PDF/PNG to work/figures/."""
from pstyle import *
from scipy import stats
e = pd.read_csv(SC + 'events.csv'); E12 = pd.read_csv(SC + 'events12.csv'); q = pd.read_csv(SC + 'quadrants.csv'); q = q[q.In_comparison]
labs = [('minimal_visible_movement', 'Minimal movement'), ('head_movement', 'Head movement'), ('limb_movement', 'Limb movement'),
        ('wire_contact', 'Wire contact'), ('caregiver_handling', 'Caregiver handling'), ('crying', 'Crying'), ('ALL', 'All events')]
W, H = COL2, 235
fig = go.Figure(layout=base_layout(W, H))
XA = [0.165, 0.345]; XB = [0.525, 0.70]; XC = [0.815, 0.995]; YD = [0.14, 0.84]
ntr, ninf = e.Trial.nunique(), e.Trial.str[:3].nunique()
y = np.arange(len(labs))[::-1]
for yi, (l, name) in zip(y, labs):
    g = e if l == 'ALL' else e[e.Label == l]; pb, po = g.Belt_good.mean() * 100, g.BIOPAC_good.mean() * 100
    b = int((g.Belt_good & ~g.BIOPAC_good).sum()); c = int((~g.Belt_good & g.BIOPAC_good).sum()); p = stats.binomtest(b, b + c).pvalue if b + c else np.nan
    fig.add_shape(type='line', x0=po, x1=pb, y0=yi, y1=yi, xref='x', yref='y', line=dict(color='#BDBDBD', width=1.6))
    tie = abs(po - pb) < 1.0
    for v, col, sz in [(po, C['bio'], 9.5 if tie else 6), (pb, C['belt'], 6)]:
        fig.add_trace(go.Scatter(x=[v], y=[yi], mode='markers', marker=dict(size=sz, color=col, line=dict(color='white', width=0.6)), xaxis='x', yaxis='y', showlegend=False, hoverinfo='skip'))
    ps = 'p < 0.001' if p < 0.001 else f'p = {p:.2f}'
    fig.add_annotation(x=XA[1] + 0.028, y=yi, xref='paper', yref='y', text=f'{b} : {c}', showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='center', yanchor='middle')
    fig.add_annotation(x=XA[1] + 0.066, y=yi, xref='paper', yref='y', text=f'<b>{ps}</b>' if p < 0.05 else ps, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='middle')
    if l == 'ALL': fig.add_shape(type='line', x0=40, x1=100, y0=yi + 0.5, y1=yi + 0.5, xref='x', yref='y', line=dict(color='#DDDDDD', width=0.6))
fig.add_annotation(x=XA[1] + 0.028, y=y[0] + 0.6, xref='paper', yref='y', text='belt-only :<br>BIOPAC-only', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='center', yanchor='bottom')
fig.add_annotation(x=XA[1] + 0.066, y=y[0] + 0.6, xref='paper', yref='y', text='McNemar', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='left', yanchor='bottom')
fig.update_layout(xaxis=ax(domain=XA, range=[40, 100], tickvals=[40, 60, 80, 100], title_text='Events with reliable HR (%)', anchor='y'),
                  yaxis=ax(domain=YD, range=[-0.6, 7.6], tickvals=list(y), ticktext=[f'{n} ({len(e if l == "ALL" else e[e.Label == l])})' for l, n in labs], ticks='', showline=False, anchor='x'))
letter(fig, 0.0, 0.93, 'a'); ptitle(fig, 0.025, 0.93, f'<b>By event type</b> ({ntr} trials, {ninf} infants, n = {len(e)})', size=SZ['annot'])
for i, (col, lab) in enumerate([(C['bio'], 'BIOPAC'), (C['belt'], 'Belt')]):
    x0 = 0.30 + i * 0.075
    fig.add_shape(type='circle', x0=x0 - 0.006, x1=x0 + 0.006, y0=0.985 - 0.006 * W / H, y1=0.985 + 0.006 * W / H, xref='paper', yref='paper', fillcolor=col, line=dict(width=0))
    fig.add_annotation(x=x0 + 0.012, y=0.985, xref='paper', yref='paper', text=lab, showarrow=False, font=dict(size=SZ['legend'], color=INK), xanchor='left', yanchor='middle')
# b: per-trial quiet vs movement (slope chart), all resolved videos incl. Poor
mov = ['limb_movement', 'head_movement', 'caregiver_handling', 'wire_contact', 'crying', 'trunk_repositioning']
E12['cls'] = np.where(E12.Label.isin(mov), 'm', np.where(E12.Label == 'minimal_visible_movement', 'q', 'o'))
rows = []
for tr, g in E12[E12.In_comparison].groupby('Trial'):
    qq, mm = g[g.cls == 'q'], g[g.cls == 'm']
    if len(qq) < 5 or len(mm) < 10: continue
    rows.append((tr, g.Belt_contact_class.iloc[0], 100 * qq.Belt_good.mean(), 100 * mm.Belt_good.mean(), 100 * qq.BIOPAC_good.mean(), 100 * mm.BIOPAC_good.mean()))
R = pd.DataFrame(rows, columns=['Trial', 'Contact', 'bq', 'bm', 'oq', 'om'])
for _, r in R.iterrows():
    fig.add_trace(go.Scatter(x=[0, 1.2], y=[r.oq, r.om], mode='lines+markers', line=dict(color=C['bio'], width=0.9), marker=dict(size=4, color=C['bio']), opacity=0.8, xaxis='x2', yaxis='y2', showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[2.6, 3.8], y=[r.bq, r.bm], mode='lines+markers', line=dict(color=C['belt'], width=0.9), marker=dict(size=4, color=C['belt']), opacity=0.8, xaxis='x2', yaxis='y2', showlegend=False, hoverinfo='skip'))
fig.update_layout(xaxis2=ax(domain=XB, range=[-0.5, 4.3], tickvals=[0, 1.2, 2.6, 3.8], ticktext=['quiet', 'moving', 'quiet', 'moving'], tickangle=0, anchor='y2'),
                  yaxis2=ax(domain=YD, range=[-3, 112], tickvals=[0, 50, 100], title_text='Reliable HR (%)', anchor='x2'))
fig.add_annotation(x=0.6, y=1.0, xref='x2', yref='y2 domain', text='<b>BIOPAC</b>', showarrow=False, font=dict(size=SZ['annot'], color=C['bio']), xanchor='center', yanchor='bottom')
fig.add_annotation(x=3.2, y=1.0, xref='x2', yref='y2 domain', text='<b>Belt</b>', showarrow=False, font=dict(size=SZ['annot'], color=C['belt']), xanchor='center', yanchor='bottom')
letter(fig, XB[0] - 0.075, 0.93, 'b'); ptitle(fig, XB[0] - 0.05, 0.93, f'<b>Per trial</b> (n = {len(R)})', size=SZ['annot'])
# c: quadrants
order = ['Both good', 'Belt only', 'BIOPAC only', 'Neither']; cols = [C['both'], C['belt'], C['bio'], C['neutral']]
rows = []; labels = []
for sub, sn in [(q, 'All'), (q[q.Belt_contact_class == 'good'], 'CG')]:
    for m in ['Still', 'Light', 'Active']:
        g = sub[sub.Motion == m]; rows.append([(g.Quadrant == o).mean() * 100 for o in order]); labels.append(f'{m} ({len(g)})')
rows = np.array(rows); yy = np.array([6.6, 5.6, 4.6, 2.4, 1.4, 0.4]); left = np.zeros(len(rows))
for k, (o, col) in enumerate(zip(order, cols)):
    fig.add_trace(go.Bar(x=rows[:, k], y=yy, base=left, orientation='h', marker=dict(color=col, line=dict(color='white', width=1.0)), width=0.8, xaxis='x3', yaxis='y3', showlegend=False, hoverinfo='skip'))
    for yi, l_, v in zip(yy, left, rows[:, k]):
        if v >= 9: fig.add_annotation(x=l_ + v / 2, y=yi, xref='x3', yref='y3', text=f'{v:.0f}', showarrow=False, font=dict(size=SZ['annot'], color='white' if o != 'Neither' else INK), xanchor='center', yanchor='middle')
    left += rows[:, k]
fig.update_layout(xaxis3=ax(domain=XC, range=[0, 100], tickvals=[0, 50, 100], title_text='10-s windows (%)', anchor='y3'),
                  yaxis3=ax(domain=YD, range=[-0.2, 7.9], tickvals=list(yy), ticktext=labels, ticks='', showline=False, anchor='x3'), barmode='overlay')
fig.add_annotation(x=0, y=7.2, xref='x3', yref='y3', text=f'<b>All compared</b> ({q.Trial.nunique()})', showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='bottom')
fig.add_annotation(x=0, y=3.0, xref='x3', yref='y3', text=f'<b>Belt contact good</b> ({q[q.Belt_contact_class == "good"].Trial.nunique()})', showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='bottom')
letter(fig, XC[0] - 0.09, 0.95, 'c')
for k, (o, col) in enumerate(zip(order, cols)):
    xpos = XC[0] - 0.045 + (k % 2) * 0.095; yl = 0.99 - (k // 2) * 0.052
    fig.add_shape(type='rect', x0=xpos, x1=xpos + 0.018, y0=yl - 0.013, y1=yl + 0.013, xref='paper', yref='paper', fillcolor=col, line=dict(width=0))
    fig.add_annotation(x=xpos + 0.022, y=yl, xref='paper', yref='paper', text=o, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='middle')
save(fig, 'Fig5_Event_and_window_quality')
