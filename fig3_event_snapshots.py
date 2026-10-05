"""Figure 3: Fig3_Event_snapshots. Run after steps 1-10; writes PDF/PNG to work/figures/."""
from pstyle import *
EV = '#E6EAEF'   # neutral event shading
from scipy.ndimage import uniform_filter1d
from common import sqi
d = cache(); e = pd.read_csv(SC + 'events12.csv'); e['c'] = e.Start + e.Dur / 2
SEL = [('P03 T02', 215.8, 'minimal_visible_movement', 'Quiet (infant still)'), ('P09 T03', 250.0, 'limb_movement', 'Limb movement, both reliable'),
       ('P09 T03', 230.5, 'wire_contact', 'Touches wire'), ('P09 T03', 280.6, 'caregiver_handling', 'Caregiver handling'),
       ('P09 T03', 191.0, 'crying', 'Crying'), ('P04 T01', 152.9, 'limb_movement', 'Limb movement, BIOPAC fails')]
WIN = 6.0
W, H = COL2, 385
fig = go.Figure(layout=base_layout(W, H))
B = {t: belt_csv(t) for t in set(s[0] for s in SEL)}
cols = [(0.03, 0.335), (0.365, 0.67), (0.70, 1.0)]
rows = [(0.52, 0.90), (0.035, 0.415)]      # (bottom, top) of each panel block
sub = [0.41, 0.41, 0.18]                     # BIOPAC, belt, acc fractions
ai = 1
for k, (tr, c, lab, title) in enumerate(SEL):
    xd = cols[k % 3]; yb, yt = rows[k // 3]; hgt = yt - yb; gap = 0.012
    ydom = {}; cur = yt
    for nm_, fr in zip(['bio', 'belt', 'acc'], sub):
        ydom[nm_] = [cur - hgt * fr + gap / 2, cur - gap / 2]; cur -= hgt * fr
    D = d[tr]; a, b = c - WIN / 2, c + WIN / 2; L = local_lag(beats(D['belt']), beats(D['bio']), a - 5, b + 5)
    ev = e[(e.Trial == tr) & (e.Label == lab) & (e.Start <= c) & (e.End >= c)].iloc[0]
    xa = 'x' if ai == 1 else f'x{ai}'
    for j, (key, dev, sh, col, name) in enumerate([('bio', D['bio'], 0, C['bio'], 'BIOPAC'), ('belt', D['belt'], L, C['belt'], 'Belt')]):
        ya = f'y{ai + j}' if ai + j > 1 else 'y'
        t = tvec(dev, sh); m = (t >= a) & (t < b)
        fig.add_shape(type='rect', x0=max(ev.Start, a) - a, x1=min(ev.End, b) - a, y0=-0.6, y1=1.0, xref=xa, yref=ya, fillcolor=EV, line=dict(width=0), layer='below')
        fig.add_trace(line(t[m] - a, np.clip(dev['sig'][m], -0.55, 0.72), col, 0.7, xaxis=xa, yaxis=ya))
        bt = beats(dev) - sh; bb = bt[(bt >= a) & (bt < b)]
        fig.add_trace(go.Scatter(x=bb - a, y=np.full(len(bb), 0.82), mode='markers', marker=dict(symbol='line-ns', size=4, line=dict(width=0.8, color=col)), xaxis=xa, yaxis=ya, showlegend=False, hoverinfo='skip'))
        s = sqi(dev, a + sh, b + sh); ok = s['good']
        fig.add_annotation(x=0, y=1.0, xref=f'{xa} domain', yref=f'{ya} domain', text=f'<b>{name}</b>', showarrow=False, font=dict(size=SZ['annot'], color=col), xanchor='left', yanchor='top')
        fig.add_annotation(x=1, y=1.0, xref=f'{xa} domain', yref=f'{ya} domain', text=f"bSQI {s['bsqi']:.2f} · HR {s['hr']:.0f} · <b>{'PASS' if ok else 'FAIL'}</b>", showarrow=False,
                           font=dict(size=SZ['annot'], color=C['pass_'] if ok else C['fail']), xanchor='right', yanchor='top')
        # amplitude scale bar 0.5 mV
        fig.add_shape(type='line', x0=-0.25, x1=-0.25, y0=-0.45, y1=0.05, xref=xa, yref=ya, line=dict(color=INK, width=1.2))
        fig.update_layout({f'yaxis{ai + j}' if ai + j > 1 else 'yaxis': dict(domain=ydom[key], range=[-0.62, 1.42], visible=False, anchor=xa, fixedrange=True)})
    # accelerometer
    ya = f'y{ai + 2}'; b_ = B[tr]; m = (b_.time_s >= a - 1) & (b_.time_s < b + 1); tt = b_.time_s[m].values - a
    fig.add_shape(type='rect', x0=max(ev.Start, a) - a, x1=min(ev.End, b) - a, y0=-0.35, y1=0.35, xref=xa, yref=ya, fillcolor=EV, line=dict(width=0), layer='below')
    for axn_, cc in zip(['AccX', 'AccY', 'AccZ'], C['acc']):
        x = b_[axn_][m].values / 4096; x = x - uniform_filter1d(x, 100); fig.add_trace(line(tt, np.clip(x, -0.34, 0.34), cc, 0.6, xaxis=xa, yaxis=ya))
    fig.add_shape(type='line', x0=-0.25, x1=-0.25, y0=-0.25, y1=-0.05, xref=xa, yref=ya, line=dict(color=INK, width=1.2))
    fig.add_shape(type='line', x0=WIN - 1.0, x1=WIN, y0=-0.3, y1=-0.3, xref=xa, yref=ya, line=dict(color=INK, width=1.2))
    fig.add_annotation(x=WIN - 0.5, y=-0.33, xref=xa, yref=ya, text='1 s', showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='center', yanchor='top')
    if k % 3 == 0:
        fig.add_annotation(x=-0.33, y=-0.2, xref=xa, yref='y' if ai == 1 else f'y{ai}', text='0.5 mV', textangle=-90, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='right', yanchor='middle')
        fig.add_annotation(x=-0.33, y=-0.2, xref=xa, yref=f'y{ai+1}', text='0.5 mV', textangle=-90, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='right', yanchor='middle')
        fig.add_annotation(x=-0.33, y=-0.15, xref=xa, yref=ya, text='0.2 g', textangle=-90, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='right', yanchor='middle')
    fig.update_layout({f'yaxis{ai + 2}': dict(domain=ydom['acc'], range=[-0.4, 0.4], visible=False, anchor=xa),
                       f'xaxis{ai}' if ai > 1 else 'xaxis': dict(domain=list(xd), range=[-0.45, WIN], visible=False, anchor='y' if ai == 1 else f'y{ai}')})
    fig.add_annotation(x=xd[0], y=yt + 0.006, xref='paper', yref='paper', text=f'<b>{title}</b>', showarrow=False, font=dict(size=SZ['title'], color=INK), xanchor='left', yanchor='bottom')
    fig.add_annotation(x=0.1, y=-0.33, xref=xa, yref=ya, text=f'{tr} · {c:.0f} s', showarrow=False, font=dict(size=SZ['annot'], color=INK2), xanchor='left', yanchor='top')
    letter(fig, xd[0] - 0.028, yt + 0.028, 'abcdef'[k])
    ai += 3
# legend
lg = [(C['bio'], 'BIOPAC ECG', 'line'), (C['belt'], 'Belt ECG', 'line'), (C['acc'][0], 'Acc X', 'line'), (C['acc'][1], 'Acc Y', 'line'), (C['acc'][2], 'Acc Z', 'line'), (EV, 'Video-annotated event', 'box')]
xpos = 0.16; yl = 0.983
for col, lab, kind in lg:
    if kind == 'line': fig.add_shape(type='line', x0=xpos, x1=xpos + 0.028, y0=yl, y1=yl, xref='paper', yref='paper', line=dict(color=col, width=1.2))
    else: fig.add_shape(type='rect', x0=xpos, x1=xpos + 0.028, y0=yl - 0.012, y1=yl + 0.012, xref='paper', yref='paper', fillcolor=col, line=dict(width=0))
    fig.add_annotation(x=xpos + 0.034, y=yl, xref='paper', yref='paper', text=lab, showarrow=False, font=dict(size=SZ['legend'], color=INK), xanchor='left', yanchor='middle')
    xpos += 0.034 + 0.0074 * len(lab) + 0.03
save(fig, 'Fig3_Event_snapshots')
