"""Step 11. Excel workbook with all intermediate tables (methods, windows, quadrants, events, HR agreement, trial table)."""
import pandas as pd, numpy as np
from config import *
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
q=pd.read_csv(WORK + 'quadrants.csv'); e=pd.read_csv(WORK + 'events.csv'); le=pd.read_csv('label_eval.csv')
wb=Workbook(); F=Font(name='Arial',size=10); B=Font(name='Arial',size=10,bold=True); H=PatternFill('solid',fgColor='DDEBF7')
def sheet(name,rows,widths=None):
    ws=wb.create_sheet(name)
    for r in rows: ws.append(r)
    for row in ws.iter_rows():
        for c in row: c.font=F
    for c in ws[1]: c.font=B; c.fill=H
    for i in range(1,ws.max_column+1): ws.column_dimensions[get_column_letter(i)].width=(widths or {}).get(i,16)
    ws.freeze_panes='A2'; return ws
wb.remove(wb.active)
methods=[["Item","Description"],
["Windows","Non-overlapping 10-s windows over the BIOPAC-overlapping part of each trial (1,392 windows, 42 trials)."],
["Beat detection","NeuroKit2 'neurokit' detector on belt ECG (interpolated to 250 Hz) and BIOPAC ECG (500 Hz, best-plausibility channel). Validated against 445 manually labelled belt beats: Se 95.7%, PPV 98.8%, F1 97.3%."],
["bSQI","F1 agreement (±75 ms) between NeuroKit and Hamilton (2002) beats in the window. Pass ≥ 0.90."],
["tSQI","Mean correlation of each beat (±120 ms) with the window's median beat template. Reported as a descriptive morphology index only; NOT part of the pass criterion (HR, not waveform shape, is the monitoring target)."],
["Physiological checks","Median HR 80–220 bpm and R-R coefficient of variation < 0.25."],
["Good window (reliable HR)","bSQI ≥ 0.90 AND HR 80–220 AND R-R CV < 0.25, computed separately and independently for each device (no reference needed). Validated: belt windows passing have MAE 2.3 bpm vs clean BIOPAC (89% within 5 bpm); failing windows MAE 30 bpm (28% within 5 bpm). Relaxing bSQI to 0.7 adds mostly wrong-HR windows (within-5 of added windows < 30%)."],
["Quadrants","Both good / Belt only / BIOPAC only / Neither."],
["Motion level","Gyro magnitude > trial 20th percentile + 200 counts = moving sample. Still < 10%, Light 10–50%, Active > 50% of window."],
["Trial tier","NeuroKit belt-vs-BIOPAC beat F1 per trial: Good / Medium / Poor; Unrated = too few BIOPAC beats."],
["Event analysis","12 annotated videos; video-to-data offset from cross-correlation of annotated movement with the belt gyroscope (see Video offsets tab). Event comparison uses the 7 Good/Medium-tier trials with a resolved offset (5 infants, 577 events); each event widened to ≥ 5 s and HR reliability computed for both devices."],
["Statistics","Event level: exact McNemar on discordant events (Belt-only vs BIOPAC-only); trial-cluster bootstrap (5,000) for the difference. Window level: per-participant Wilcoxon signed-rank."],
["Caveat","Offsets for P01 T01 (timestamp only, ±3 s) and P03 T01 (annotation-based estimate disagrees with clock/pixel estimate) are flagged; P06 T02 and P08 T03 unresolved and excluded. Annotation timing ± ~1 s."]]
sheet("Methods",methods,{1:20,2:110})
flow=[["Step","Trials","Note"],["Belt trials recorded",48,"10 infants"],["No BIOPAC reference (P07)",6,"excluded from ECG comparison"],["Trials with belt + BIOPAC",42,""],["Unrated (too few BIOPAC beats)",2,"P02 T01, P10 T02"],["Rated trials",40,""],["Good tier",14,""],["Medium tier",9,""],["Poor tier",17,"kept for SQI analysis, excluded from Good+Medium subset"],["Good + Medium analysis set",23,""],["Video-annotated Good-tier trials",3,"P03 T01, P03 T02, P09 T03"]]
sheet("Trial flow",flow,{1:36,2:10,3:50})
# window data
cols=['Trial','Participant','Belt_contact_class','Motion','Moving_%','Quadrant','Belt_good','BIOPAC_good','Belt_HR','BIOPAC_HR','Belt_bSQI','Belt_tSQI','Belt_RRcv','Bio_bSQI','Bio_tSQI','Bio_RRcv']
wd=q[cols].copy(); wd['Belt_good']=wd.Belt_good.map({True:1,False:0}); wd['BIOPAC_good']=wd.BIOPAC_good.map({True:1,False:0})
sheet("Window data",[cols]+wd.round(3).values.tolist())
N=len(wd)+1
# quadrant summary with formulas
rows=[["Subset","Motion","Windows","Both good %","Belt only %","BIOPAC only %","Neither %","Belt good %","BIOPAC good %"]]
ws=sheet("Quadrants by motion",rows)
r=2
for sub in ["All rated","Good+Medium"]:
    for m in ["Still","Light","Active","All"]:
        ws.cell(r,1,sub); ws.cell(r,2,m)
        mc=f'"{m}"' if m!="All" else '"*"'
        if sub=="All rated": tf=f'\'Window data\'!$C$2:$C${N},"<>Unrated"'; tf2=None
        else: tf='\'Window data\'!$C$2:$C$%d,"Good"'%N; tf2='\'Window data\'!$C$2:$C$%d,"Medium"'%N
        def cnt(extra=""):
            a=f'COUNTIFS({tf},\'Window data\'!$D$2:$D${N},{mc}{extra})'
            if tf2: a+=f'+COUNTIFS({tf2},\'Window data\'!$D$2:$D${N},{mc}{extra})'
            return a
        ws.cell(r,3,f'={cnt()}')
        for j,qq in enumerate(["Both good","Belt only","BIOPAC only","Neither"]):
            ws.cell(r,4+j,f'=IFERROR(({cnt(f",\'Window data\'!$F$2:$F${N},\"{qq}\"")})/$C{r},0)')
        ws.cell(r,8,f'=D{r}+E{r}'); ws.cell(r,9,f'=D{r}+F{r}')
        for c in range(4,10): ws.cell(r,c).number_format='0.0%'
        for c in ws[r]: c.font=F
        r+=1
ws.cell(r+1,1,"Per-participant Wilcoxon (belt-good vs BIOPAC-good share, Active windows): all rated p=0.56 (9 participants), Good+Medium p=0.63 (7). No significant participant-level difference.").font=F
# per trial
tr=[["Trial","Contact","Windows","Both good","Belt only","BIOPAC only","Neither","Belt good %","BIOPAC good %"]]
ws=sheet("Per-trial quadrants",tr)
for i,(t,g) in enumerate(q.groupby('Trial'),start=2):
    ws.cell(i,1,t); ws.cell(i,2,g.Belt_contact_class.iloc[0]); ws.cell(i,3,f'=COUNTIF(\'Window data\'!$A$2:$A${N},A{i})')
    for j,qq in enumerate(["Both good","Belt only","BIOPAC only","Neither"]):
        ws.cell(i,4+j,f'=COUNTIFS(\'Window data\'!$A$2:$A${N},$A{i},\'Window data\'!$F$2:$F${N},"{qq}")')
    ws.cell(i,8,f'=IFERROR((D{i}+E{i})/C{i},0)'); ws.cell(i,9,f'=IFERROR((D{i}+F{i})/C{i},0)')
    ws.cell(i,8).number_format=ws.cell(i,9).number_format='0.0%'
    for c in ws[i]: c.font=F
# events
ev=e.copy(); ev['Belt_good']=ev.Belt_good.astype(int); ev['BIOPAC_good']=ev.BIOPAC_good.astype(int)
sheet("Event data",[list(ev.columns)]+ev.round(3).values.tolist())
M=len(ev)+1
ws=sheet("Events by type",[["Event type","Events","Belt good %","BIOPAC good %","Belt only (n)","BIOPAC only (n)","McNemar exact p"]],{1:28})
from scipy import stats
labs=['limb_movement','head_movement','minimal_visible_movement','wire_contact','caregiver_handling','crying','contact_target_uncertain','ALL']
for i,l in enumerate(labs,start=2):
    g=e if l=='ALL' else e[e.Label==l]; lc='"*"' if l=='ALL' else f'A{i}'
    ws.cell(i,1,l)
    ws.cell(i,2,f'=COUNTIF(\'Event data\'!$B$2:$B${M},{lc})')
    ws.cell(i,3,f'=IFERROR(SUMIF(\'Event data\'!$B$2:$B${M},{lc},\'Event data\'!$G$2:$G${M})/B{i},0)')
    ws.cell(i,4,f'=IFERROR(SUMIF(\'Event data\'!$B$2:$B${M},{lc},\'Event data\'!$H$2:$H${M})/B{i},0)')
    ws.cell(i,5,f'=COUNTIFS(\'Event data\'!$B$2:$B${M},{lc},\'Event data\'!$G$2:$G${M},1,\'Event data\'!$H$2:$H${M},0)')
    ws.cell(i,6,f'=COUNTIFS(\'Event data\'!$B$2:$B${M},{lc},\'Event data\'!$G$2:$G${M},0,\'Event data\'!$H$2:$H${M},1)')
    b=(g.Belt_good&~g.BIOPAC_good).sum(); c=(~g.Belt_good&g.BIOPAC_good).sum()
    ws.cell(i,7,float('%.3g'%stats.binomtest(b,b+c).pvalue) if b+c else None)
    ws.cell(i,3).number_format=ws.cell(i,4).number_format='0.0%'
    for c_ in ws[i]: c_.font=F
ws.cell(11,1,"p values from scipy binomtest on discordant counts. 7 trials / 5 infants / 577 events: overall belt 68.3% vs BIOPAC 73.7%, trial-cluster bootstrap difference −5.4 pts, 95% CI [−32.6, +16.0]; per-infant Wilcoxon p = 0.81. Belt reliability is trial-level (between-trial variance share 0.75), BIOPAC reliability is event-level (0.17).").font=F
ws=sheet("Events by trial",[["Trial","Events","Belt good %","BIOPAC good %"]])
for i,t in enumerate(sorted(e.Trial.unique()),start=2):
    ws.cell(i,1,t); ws.cell(i,2,f'=COUNTIF(\'Event data\'!$A$2:$A${M},A{i})')
    ws.cell(i,3,f'=SUMIF(\'Event data\'!$A$2:$A${M},A{i},\'Event data\'!$G$2:$G${M})/B{i}')
    ws.cell(i,4,f'=SUMIF(\'Event data\'!$A$2:$A${M},A{i},\'Event data\'!$H$2:$H${M})/B{i}')
    ws.cell(i,3).number_format=ws.cell(i,4).number_format='0.0%'
    for c_ in ws[i]: c_.font=F
# Agreement
bg=q[q.Quadrant=='Both good']; d=bg.Belt_HR-bg.BIOPAC_HR
eb=e[e.Belt_good&e.BIOPAC_good]; de=eb.Belt_HR-eb.BIOPAC_HR
sheet("HR agreement",[["Set","n","Bias (bpm)","SD","LoA low","LoA high","MAE","Within 5 bpm"],
 ["Both-good 10-s windows (40 trials)",len(bg),round(d.mean(),2),round(d.std(),2),round(d.mean()-1.96*d.std(),2),round(d.mean()+1.96*d.std(),2),round(d.abs().mean(),2),round((d.abs()<=5).mean(),3)],
 ["Both-good annotated events (3 trials)",len(eb),round(de.mean(),2),round(de.std(),2),round(de.mean()-1.96*de.std(),2),round(de.mean()+1.96*de.std(),2),round(de.abs().mean(),2),round((de.abs()<=5).mean(),3)]],{1:38})
print(le.columns.tolist()); 
sheet("SQI validation",[list(le.columns)]+le.round(3).values.tolist())
tt=pd.read_csv(WORK + 'trial_table_v2.csv'); c60=pd.read_csv(WORK + 'contact_first60.csv').set_index('Trial').contact_first60; tt['Contact_first60s']=tt.Trial.map(c60)
sheet("Trial table v2",[list(tt.columns)]+tt.fillna('').values.tolist())
ws=wb["Trial table v2"]; ws.cell(ws.max_row+2,1,"Inclusion rule: trial enters belt-vs-BIOPAC comparison if BIOPAC has >= 3 reliable 10-s windows (36 of 42). Belt_contact_pct = belt reliable-window share (reference-free). Contact_first60s = class from the first 60 s only (good if >= 3 of 6 windows reliable); agrees with whole-trial class in 92% of trials and predicts the rest of the session: contact-good (21 trials) belt 82.4% vs BIOPAC 73.2% at t >= 60 s (per-trial Wilcoxon p = 0.055, belt ahead in 15/21); contact-poor (15) belt 4.3% vs BIOPAC 57.2%.").font=F
vo=pd.read_csv('video_offsets_final.csv'); sheet("Video offsets",[list(vo.columns)]+vo.fillna('').values.tolist(),{6:90})
wb.save(WORK + 'NeoTex_Quality_Quadrants_Events.xlsx')
