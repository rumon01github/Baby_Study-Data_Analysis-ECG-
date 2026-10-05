"""Figure 7: Fig7_Pipeline_validation. Run after steps 1-10; writes PDF/PNG to work/figures/."""
from pstyle import *
R = pd.read_csv(SC + 'handoff_eval.csv'); M = pd.read_csv(SC + 's3_strip_results.csv'); LC = pd.read_csv(SC + 's3_learning_curve.csv')
V = pd.read_csv(SC + 'val_qrs_windows_v2.csv'); FA = pd.read_csv(SC + 's5_failure_attribution.csv')
V = V[V.In_comparison & V.BIOPAC_good]
W, H = COL2, 200
fig = go.Figure(layout=base_layout(W, H))
xs = [None, (0.07, 0.36), (0.45, 0.70), (0.80, 0.975)]; YD = [0.215, 0.80]
# b: learning curve
g = LC.groupby('n_strips')
for col, c, lab in [('F1_readable', C['belt'], 'Readable strips'), ('F1_partly', C['bio'], 'Partly readable')]:
    mu = g[col].mean(); lo = g[col].min(); hi = g[col].max(); xv = mu.index.values
    fig.add_trace(go.Scatter(x=xv, y=mu.values, mode='lines+markers', line=dict(color=c, width=1.2), marker=dict(size=5, color=c, line=dict(color='white', width=0.6)),
                             error_y=dict(type='data', symmetric=False, array=(hi - mu).values, arrayminus=(mu - lo).values, color=c, thickness=0.8, width=2), xaxis='x2', yaxis='y2', showlegend=False, hoverinfo='skip'))
    fig.add_annotation(x=np.log10(xv[-1]), y=mu.values[-1] - 0.012, xref='x2', yref='y2', text=lab, showarrow=False, font=dict(size=SZ['annot'], color=c), xanchor='right', yanchor='top')
fig.update_layout(xaxis2=ax(domain=list(xs[1]), type='log', range=[np.log10(45), np.log10(1700)], tickvals=[62, 124, 311, 622, 1243], ticktext=['62', '124', '311', '622', '1243'], title_text='Labelled strips used for training', anchor='y2'),
                  yaxis2=ax(domain=YD, range=[0.86, 1.005], tickvals=[0.9, 0.95, 1.0], title_text='Beat F1, held-out infants', anchor='x2'))
ptitle(fig, xs[1][0], YD[1] + 0.01, '<b>Learning curve</b> (leave-one-infant-out)'); letter(fig, 0.0, YD[1] + 0.005, 'a')
# c: gate validation ECDF of |HR error| pass vs fail
V['err'] = (V.hr_new - V.BIOPAC_HR).abs(); gate = V.ok_rr & (V.agree2 >= 0.9)
for m, c, lab, dash in [(gate, C['pass_'], 'Gate pass', 'solid'), (~gate, C['fail'], 'Gate fail', 'solid')]:
    e = np.sort(V.loc[m, 'err'].dropna().values); e = np.clip(e, 0, 40); y = np.arange(1, len(e) + 1) / len(e)
    fig.add_trace(go.Scatter(x=np.r_[0, e], y=np.r_[0, y], mode='lines', line=dict(color=c, width=1.3, shape='hv'), xaxis='x3', yaxis='y3', showlegend=False, hoverinfo='skip'))
    fig.add_annotation(x=1.0, y=0.08 if lab == 'Gate pass' else 0.9, xref='x3 domain', yref='y3 domain', text=f'{lab} (n = {m.sum()})<br>MAE {V.loc[m, "err"].mean():.1f} bpm', showarrow=False, font=dict(size=SZ['annot'], color=c), xanchor='right', yanchor='bottom' if lab == 'Gate pass' else 'top', align='right')
fig.add_shape(type='line', x0=5, x1=5, y0=0, y1=1, xref='x3', yref='y3', line=dict(color=INK3, width=0.6, dash='dot'))
fig.update_layout(xaxis3=ax(domain=list(xs[2]), range=[0, 40], tickvals=[0, 5, 10, 20, 30, 40], title_text='|belt − BIOPAC| HR (bpm)', anchor='y3'),
                  yaxis3=ax(domain=YD, range=[0, 1.02], tickvals=[0, 0.5, 1], title_text='Cumulative share of windows', anchor='x3'))
ptitle(fig, xs[2][0], YD[1] + 0.01, '<b>Reliability gate</b>'); letter(fig, xs[2][0] - 0.06, YD[1] + 0.005, 'b')
# d: failure attribution stacked bars
ctx = ['belt contact poor (setup check)', 'contact good · still', 'contact good · light motion', 'contact good · active motion']
cn = ['Belt fit poor (fails 60-s contact check)', 'Fit good · infant still (<10% of window moving)', 'Fit good · light motion (10–50%)', 'Fit good · active motion (>50%)']
cc = ['#7F7F7F', '#CFE6F4', '#6BAED6', '#08519C']
yy = {'Belt': 1, 'BIOPAC': 0}
for dev, yv in yy.items():
    left = 0
    for o, col, nm in zip(ctx, cc, cn):
        v = float(FA[(FA.device == dev) & (FA.context == o)].share_of_failures_pct.iloc[0])
        fig.add_trace(go.Bar(x=[v], y=[yv], base=[left], orientation='h', width=0.62, marker=dict(color=col, line=dict(color='white', width=0.8)), xaxis='x4', yaxis='y4', showlegend=False, hoverinfo='skip'))
        if v >= 9: fig.add_annotation(x=left + v / 2, y=yv, xref='x4', yref='y4', text=f'{v:.0f}', showarrow=False, font=dict(size=SZ['annot'], color='white' if col in ('#7F7F7F', '#08519C', '#6BAED6') else INK), xanchor='center', yanchor='middle')
        left += v
fig.update_layout(xaxis4=ax(domain=list(xs[3]), range=[0, 100], tickvals=[0, 50, 100], title_text='Share of gate-fail<br>windows (%)', anchor='y4'),
                  yaxis4=ax(domain=YD, range=[-0.6, 1.6], tickvals=[1, 0], ticktext=['Belt', 'BIOPAC'], ticks='', showline=False, anchor='x4'), barmode='overlay')
ptitle(fig, xs[3][0] - 0.03, YD[1] + 0.01, '<b>Why windows fail</b>'); letter(fig, xs[3][0] - 0.06, YD[1] + 0.005, 'c')
# legends
yl = 0.985
for k, (col, nm) in enumerate(zip(cc, cn)):
    xpos = 0.46 + (k % 2) * 0.27; yl2 = 0.985 - (k // 2) * 0.062
    fig.add_shape(type='rect', x0=xpos, x1=xpos + 0.018, y0=yl2 - 0.016, y1=yl2 + 0.016, xref='paper', yref='paper', fillcolor=col, line=dict(width=0))
    fig.add_annotation(x=xpos + 0.022, y=yl2, xref='paper', yref='paper', text=nm, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='middle')
save(fig, 'Fig7_Pipeline_validation')
