"""Figure 6: Fig6_Pipeline. Run after steps 1-10; writes PDF/PNG to work/figures/."""
from pstyle import *
W, H = COL2, 160
fig = go.Figure(layout=base_layout(W, H))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='markers', marker=dict(opacity=0), showlegend=False, hoverinfo='skip'))
fig.update_layout(xaxis=dict(domain=[0, 1], range=[0, 1], visible=False, fixedrange=True), yaxis=dict(domain=[0, 1], range=[0, 1], visible=False, fixedrange=True))
# main chain (paper coords), top row
boxes = [('Belt ingest', 'ECG 100 Hz (mV)<br>IMU 24 Hz', C['belt']),
         ('Pre-emphasis', 'band-pass 5–25 Hz<br>derivative, square<br>50-ms integration', INK2),
         ('Adaptive threshold', 'THR = N + 0.25 (S − N)<br>refractory 200 ms<br>search-back<br>artifact guards', INK2),
         ('R refinement', 'argmax |y| in<br>[−120, +20] ms', INK2),
         ('Reliability gate', 'HR 80–220 bpm<br>R–R CV < 0.25<br>agreement ≥ 0.9', C['pass_']),
         ('Output', 'HR every 10 s<br>+ reliable / not', INK)]
n = len(boxes); gap = 0.022; bw = (1 - gap * (n - 1)) / n; y0, y1 = 0.56, 0.97
for i, (t, sub, col) in enumerate(boxes):
    x0 = i * (bw + gap); x1 = x0 + bw
    fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, xref='paper', yref='paper', line=dict(color=col, width=1.0), fillcolor='white')
    fig.add_shape(type='rect', x0=x0, x1=x1, y0=y1 - 0.13, y1=y1, xref='paper', yref='paper', line=dict(width=0), fillcolor=col)
    fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 0.065, xref='paper', yref='paper', text=f'<b>{t}</b>', showarrow=False, font=dict(size=SZ['title'], color='white'), xanchor='center', yanchor='middle')
    fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1 - 0.13) / 2, xref='paper', yref='paper', text=sub, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='center', yanchor='middle', align='center')
    if i < n - 1:
        fig.add_annotation(x=x1 + gap, y=(y0 + y1) / 2, ax=x1 + 0.002, ay=(y0 + y1) / 2, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0, arrowcolor=INK, text='')
# bottom row: offline / learning + context
lower = [(0.0, 0.47, 'Expert-in-the-loop labels', '1,819 strips × 8 s, 48 trials, 10 infants, 21.9 k beats;<br>proposals shown, reviewer adds / removes / moves, strip quality', C['bio']),
         (0.50, 0.73, 'Learned beat scorer', 'gradient-boosted trees on<br>amplitude, R–R, template, kurtosis', C['bio']),
         (0.76, 1.0, 'Context', 'gyro motion level (still / light / active)<br>60-s belt contact check', C['motion'])]
yb0, yb1 = 0.03, 0.38
for x0, x1, t, sub, col in lower:
    fig.add_shape(type='rect', x0=x0, x1=x1, y0=yb0, y1=yb1, xref='paper', yref='paper', line=dict(color=col, width=1.0, dash='dot' if col != C['motion'] else 'solid'), fillcolor='white')
    fig.add_annotation(x=x0 + 0.012, y=yb1 - 0.03, xref='paper', yref='paper', text=f'<b>{t}</b>', showarrow=False, font=dict(size=SZ['title'], color=col), xanchor='left', yanchor='top')
    fig.add_annotation(x=x0 + 0.012, y=yb0 + 0.03, xref='paper', yref='paper', text=sub, showarrow=False, font=dict(size=SZ['annot'], color=INK), xanchor='left', yanchor='bottom', align='left')
# arrows: labels -> scorer ; scorer -> threshold stage (validates / verifies) ; context -> gate (explains)
fig.add_annotation(x=0.50, y=(yb0 + yb1) / 2, ax=0.47, ay=(yb0 + yb1) / 2, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0, arrowcolor=C['bio'], text='')
xc = 2 * (bw + gap) + bw / 2
fig.add_annotation(x=xc, y=y0, ax=0.615, ay=yb1, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0, arrowcolor=C['bio'], text='')
fig.add_annotation(x=0.56, y=0.48, xref='paper', yref='paper', text='verifies candidates (offline)', showarrow=False, font=dict(size=SZ['annot'], color=C['bio']), xanchor='left', yanchor='middle')
xg = 4 * (bw + gap) + bw / 2
fig.add_annotation(x=xg, y=y0, ax=0.88, ay=yb1, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0, arrowcolor=C['motion'], text='')
fig.add_annotation(x=0.885, y=0.48, xref='paper', yref='paper', text='explains failures', showarrow=False, font=dict(size=SZ['annot'], color=C['motion']), xanchor='left', yanchor='middle')
# data-only arrow: belt ingest -> labels
fig.add_annotation(x=0.08, y=yb1, ax=0.08, ay=y0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=1.0, arrowcolor=C['bio'], text='')
save(fig, 'Fig6_Pipeline')
