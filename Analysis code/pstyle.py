"""Plotly style for ACM sigconf figures (SenSys). 1 px = 1 pt in PDF export."""
import os, glob, pickle, numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# kaleido needs a Chromium; set BROWSER_PATH if it is not found automatically
_c = glob.glob('/opt/pw-browsers/chromium*/chrome-linux/chrome')
if _c: os.environ.setdefault('BROWSER_PATH', _c[0])

from config import WORK, BELT, FIGS
SC = WORK; OUT = FIGS

# ACM sigconf geometry (acmart): \columnwidth = 240 pt (3.33 in), \textwidth = 504 pt (7.0 in)
COL1, COL2 = 240, 504
# Type sizes in pt at print size. acmart body 9 pt, caption 8 pt -> figure text 7-8 pt, never below 6.5
FONT = 'Liberation Sans'           # metric-compatible with Arial/Helvetica, embedded in the PDF
SZ = dict(letter=9, title=8, axis=7.5, tick=7, legend=7, annot=6.8)
INK = '#1A1A1A'; INK2 = '#4D4D4D'; INK3 = '#8A8A8A'; GRID = '#E4E4E4'
# Okabe-Ito, validated with the dataviz palette checker (all checks pass)
C = dict(belt='#0072B2', bio='#D55E00', both='#009E73', neutral='#C8C8C8',
         belt_light='#9CC7E3', bio_light='#F2B48A', motion='#5C5C5C', band='#FFF3CF',
         acc=['#CC79A7', '#009E73', '#E69F00'], fail='#B2182B', pass_='#1B7837')

def base_layout(width, height, legend=None):
    return dict(
        width=width, height=height, template='none',
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=FONT, size=SZ['tick'], color=INK),
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        showlegend=legend is not None,
        legend=legend or {},
    )

AX = dict(showline=True, linecolor=INK, linewidth=0.75, mirror=False, ticks='outside',
          ticklen=2.5, tickwidth=0.75, tickcolor=INK, showgrid=False, zeroline=False,
          tickfont=dict(size=SZ['tick'], color=INK), title_font=dict(size=SZ['axis'], color=INK),
          title_standoff=4, automargin=False)

def ax(**kw):
    d = dict(AX); d.update(kw); return d

def letter(fig, x, y, s):
    """Bold panel letter at paper coords (x,y) = top-left anchor."""
    fig.add_annotation(x=x, y=y, xref='paper', yref='paper', text=f'<b>{s}</b>', showarrow=False,
                       font=dict(size=SZ['letter'], color=INK), xanchor='left', yanchor='bottom', align='left')

def ptitle(fig, x, y, s, color=INK, size=None, anchor='left'):
    fig.add_annotation(x=x, y=y, xref='paper', yref='paper', text=s, showarrow=False,
                       font=dict(size=size or SZ['title'], color=color), xanchor=anchor, yanchor='bottom')

def save(fig, name):
    fig.write_image(OUT + f'{name}.pdf', scale=4/3)   # kaleido: 1 css px = 0.75 pt; 4/3 makes 1 px = 1 pt
    fig.write_image(OUT + f'{name}.png', scale=600 / 72)   # 600 dpi at print size
    fig.write_image(OUT + f'{name}_preview.png', scale=200 / 72)
    return qa(OUT + f'{name}.pdf')

def qa(pdf, min_pt=6.4):
    """Publication QA: min font size and overlapping text boxes, from the PDF itself."""
    import pdfplumber
    with pdfplumber.open(pdf) as p:
        pg = p.pages[0]; W, H = pg.width, pg.height
        words = [w for w in pg.extract_words(extra_attrs=['size', 'upright'], keep_blank_chars=False, use_text_flow=False) if w.get('upright', True)]
    sizes = sorted(set(round(w['size'], 1) for w in words))
    small = [(w['text'], round(w['size'], 1)) for w in words if w['size'] < min_pt]
    # collisions: boxes from different words that overlap by > 1 pt in both axes
    coll = []
    for i in range(len(words)):
        a = words[i]
        for j in range(i + 1, len(words)):
            b = words[j]
            ox = min(a['x1'], b['x1']) - max(a['x0'], b['x0']); oy = min(a['bottom'], b['bottom']) - max(a['top'], b['top'])
            if ox > 1.0 and oy > 1.0: coll.append((a['text'], b['text']))
    rep = dict(page_pt=(round(W, 1), round(H, 1)), inches=(round(W / 72, 2), round(H / 72, 2)), font_sizes_pt=sizes,
               n_words=len(words), too_small=small[:10], collisions=coll[:15])
    print('QA', os.path.basename(pdf), rep)
    return rep

# ---------- data helpers (same as before) ----------
def cache(): return pickle.load(open(SC + 'beats_cache.pkl', 'rb'))
def belt_csv(trial):
    from common import belt_path
    f = belt_path(trial)
    d = pd.read_csv(f, usecols=['time_s', 'ECG', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'])
    d['acc_g'] = np.sqrt(d.AccX**2 + d.AccY**2 + d.AccZ**2) / 4096
    d['gyro'] = np.sqrt(d.GyroX**2 + d.GyroY**2 + d.GyroZ**2)
    return d
def beats(dev, which='nk'): return dev[which] / dev['fs'] + dev['t0']
def tvec(dev, shift=0): return np.arange(len(dev['sig'])) / dev['fs'] + dev['t0'] - shift
def local_lag(bt, ct, a, b, search=1.0):
    B = bt[(bt >= a - search) & (bt < b + search)]; R = ct[(ct >= a) & (ct < b)]
    if len(B) < 3 or len(R) < 3: return 0.0
    best = (-1, 0)
    for L in np.arange(-search, search, 0.004):
        d = np.abs((B - L)[:, None] - R[None, :]).min(1); s = (d < 0.03).sum()
        if s > best[0]: best = (s, L)
    L = best[1]; d = (B - L)[:, None] - R[None, :]; i = np.abs(d).argmin(1); dd = d[np.arange(len(B)), i]
    m = np.abs(dd) < 0.05
    return L + np.median(dd[m]) if m.any() else L
def line(x, y, color, width=0.8, name=None, showlegend=False, **kw):
    return go.Scatter(x=x, y=y, mode='lines', line=dict(color=color, width=width), name=name, showlegend=showlegend,
                      hoverinfo='skip', **kw)
