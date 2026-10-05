"""Figure 1: Fig1_ECG_morphology. Run after steps 1-10; writes PDF/PNG to work/figures/."""
from pstyle import *
d = cache(); q = pd.read_csv(SC + 'quadrants.csv'); tier = q.groupby('Trial').Belt_contact_class.first()

def ensemble(dev, pre=0.15, post=0.25):
    fs = dev['fs']; idx = dev['nk'].astype(int); w0, w1 = int(pre * fs), int(post * fs)
    S = np.array([dev['sig'][i - w0:i + w1] for i in idx if i - w0 >= 0 and i + w1 < len(dev['sig'])])
    return np.arange(-w0, w1) / fs * 1000, S
def beatcorr(dev):
    _, S = ensemble(dev, 0.12, 0.12)
    if len(S) < 5: return np.array([])
    m = np.median(S, 0); return np.array([np.corrcoef(s, m)[0, 1] for s in S])

W, H = COL2, 310
fig = go.Figure(layout=base_layout(W, H))
XL, XR = 0.055, 0.995
# --- panel a: synchronous strip ---
TR, a0, dur = 'P03 T01', 40.2, 6.0
D = d[TR]; bio, bel = D['bio'], D['belt']; L = local_lag(beats(bel), beats(bio), a0, a0 + dur)
yd = {'bio': [0.785, 0.955], 'belt': [0.59, 0.76]}
br = beats(bio); br = br[(br >= a0) & (br < a0 + dur)]
for k, dev, sh, col, lab in [('bio', bio, 0, C['bio'], 'BIOPAC (wet Ag/AgCl electrodes, wired)'), ('belt', bel, L, C['belt'], 'NeoTex belt (dry textile electrodes, wireless)')]:
    t = tvec(dev, sh); m = (t >= a0) & (t < a0 + dur)
    axn = 'x' if k == 'bio' else 'x2'; ayn = 'y' if k == 'bio' else 'y2'
    fig.add_trace(line(t[m] - a0, dev['sig'][m], col, 0.8, xaxis=axn, yaxis=ayn))
    bt = beats(dev) - sh; bb = bt[(bt >= a0) & (bt < a0 + dur)]
    fig.add_trace(go.Scatter(x=bb - a0, y=np.interp(bb, t, dev['sig']) + 0.09, mode='markers', marker=dict(symbol='triangle-down', size=4, color=col),
                             xaxis=axn, yaxis=ayn, showlegend=False, hoverinfo='skip'))
    for x in br:   # BIOPAC R times as reference guides on both strips
        fig.add_shape(type='line', x0=x - a0, x1=x - a0, y0=-0.32, y1=0.75, xref=axn, yref=ayn, line=dict(color='#BDBDBD', width=0.5, dash='dot'), layer='below')
    fig.add_annotation(x=0.0, y=0.78, xref=axn, yref=ayn, text=f'<b>{lab}</b>', showarrow=False, font=dict(size=SZ['title'], color=col), xanchor='left', yanchor='top')
fig.update_layout(
    xaxis=ax(domain=[XL, XR], range=[0, dur], showticklabels=False, ticks='', showline=False, anchor='y'),
    yaxis=ax(domain=yd['bio'], range=[-0.32, 0.78], tickvals=[0, 0.4], title_text='mV', anchor='x'),
    xaxis2=ax(domain=[XL, XR], range=[0, dur], title_text='Time (s)', dtick=1, anchor='y2'),
    yaxis2=ax(domain=yd['belt'], range=[-0.32, 0.78], tickvals=[0, 0.4], title_text='mV', anchor='x2'))
letter(fig, 0.0, 0.952, 'a')

# --- bottom row ---
YB = [0.125, 0.445]; xs = [(0.055, 0.245), (0.31, 0.50), (0.565, 0.745), (0.825, 0.995)]
# b, c ensembles
for k, (dev, col, lab, axi) in enumerate([(bio, C['bio'], 'BIOPAC', 3), (bel, C['belt'], 'Belt', 4)]):
    tm, S = ensemble(dev); xs_, ys_ = [], []
    for s in S: xs_ += list(tm) + [None]; ys_ += list(s) + [None]
    axn, ayn = f'x{axi}', f'y{axi}'
    fig.add_trace(go.Scatter(x=xs_, y=ys_, mode='lines', line=dict(color=col, width=0.3), opacity=0.12, xaxis=axn, yaxis=ayn, showlegend=False, hoverinfo='skip'))
    q1, q3 = np.percentile(S, [25, 75], 0); med = np.median(S, 0)
    fig.add_trace(go.Scatter(x=np.r_[tm, tm[::-1]], y=np.r_[q3, q1[::-1]], fill='toself', fillcolor=col, opacity=0.35, line=dict(width=0), xaxis=axn, yaxis=ayn, showlegend=False, hoverinfo='skip'))
    fig.add_trace(line(tm, med, INK, 1.1, xaxis=axn, yaxis=ayn))
    fig.update_layout({f'xaxis{axi}': ax(domain=list(xs[k]), range=[-150, 250], tickvals=[-100, 0, 100, 200], title_text='Time from R peak (ms)', anchor=ayn),
                       f'yaxis{axi}': ax(domain=YB, range=[-0.35, 0.65], tickvals=[-0.2, 0, 0.2, 0.4, 0.6], title_text='Amplitude (mV)', anchor=axn)})
    ptitle(fig, xs[k][0], YB[1] + 0.01, f'<b>{lab}</b>, n = {len(S)} beats', color=col)
    letter(fig, xs[k][0] - 0.055, YB[1] + 0.005, 'bc'[k])
# d: belt vs reference HR where both reliable (36 compared trials); e: Bland-Altman
from scipy import stats
bg = q[(q.Quadrant == 'Both good') & q.In_comparison]; xh, yh = bg.BIOPAC_HR.values, bg.Belt_HR.values
r_ = stats.pearsonr(xh, yh)[0]; mae = np.mean(np.abs(yh - xh))
axn, ayn = 'x5', 'y5'
fig.add_trace(go.Scatter(x=xh, y=yh, mode='markers', marker=dict(size=3.2, color=C['belt'], opacity=0.45), xaxis=axn, yaxis=ayn, showlegend=False, hoverinfo='skip'))
fig.add_shape(type='line', x0=100, x1=190, y0=100, y1=190, xref=axn, yref=ayn, line=dict(color=INK, width=0.6, dash='dash'))
fig.add_annotation(x=0.04, y=0.97, xref='x5 domain', yref='y5 domain', text=f'r = {r_:.3f}<br>MAE = {mae:.2f} bpm<br>n = {len(xh)} windows', showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='top', align='left')
fig.update_layout(xaxis5=ax(domain=list(xs[2]), range=[100, 190], tickvals=[100, 130, 160, 190], title_text='BIOPAC HR (bpm)', anchor=ayn),
                  yaxis5=ax(domain=YB, range=[100, 190], tickvals=[100, 130, 160, 190], title_text='Belt HR (bpm)', anchor=axn))
ptitle(fig, xs[2][0], YB[1] + 0.01, '<b>Both reliable: HR</b>')
letter(fig, xs[2][0] - 0.055, YB[1] + 0.005, 'd')
axn, ayn = 'x6', 'y6'; mm = (xh + yh) / 2; dd = yh - xh; bias = dd.mean(); sd = dd.std(ddof=1); lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
fig.add_trace(go.Scatter(x=mm, y=dd, mode='markers', marker=dict(size=3.2, color=C['belt'], opacity=0.45), xaxis=axn, yaxis=ayn, showlegend=False, hoverinfo='skip'))
for v, dash in [(bias, 'solid'), (lo, 'dash'), (hi, 'dash')]:
    fig.add_shape(type='line', x0=100, x1=190, y0=v, y1=v, xref=axn, yref=ayn, line=dict(color=INK, width=0.6, dash=dash))
for v, s_ in [(hi, f'+1.96 SD  {hi:.1f}'), (bias, f'Bias  {bias:.2f}'), (lo, f'−1.96 SD  {lo:.1f}')]:
    fig.add_annotation(x=189, y=v, xref=axn, yref=ayn, text=s_, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='right', yanchor='bottom', bgcolor='rgba(255,255,255,0.85)', borderpad=1)
fig.update_layout(xaxis6=ax(domain=list(xs[3]), range=[100, 190], tickvals=[100, 130, 160, 190], title_text='Mean HR (bpm)', anchor=ayn),
                  yaxis6=ax(domain=YB, range=[-20, 15], tickvals=[-20, -10, 0, 10], title_text='Belt − BIOPAC (bpm)', anchor=axn))
ptitle(fig, xs[3][0], YB[1] + 0.01, '<b>Bland–Altman</b>')
letter(fig, xs[3][0] - 0.055, YB[1] + 0.005, 'e')
print('HR agreement', len(xh), r_, mae, bias, lo, hi)
save(fig, 'Fig1_ECG_morphology')
