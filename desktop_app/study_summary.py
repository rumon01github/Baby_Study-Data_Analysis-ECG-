"""Window-level study summaries with explicit participant/trial provenance."""
from PyQt6 import QtWidgets, QtCore
import numpy as np
import pandas as pd


def window_rows(data, reviewed=True, offset=None):
    ann=data.get('annotations',[])
    offsets=[float(a['ecg_start_s'])-float(a['video_start_s']) for a in ann if str(a.get('ecg_start_s','')).strip()]
    off=0 if data.get('annotations_synced') else (offset if offset is not None else (float(np.median(offsets)) if offsets else 0))
    participant,trial=data['trial'].split()
    rows=[]
    for w in data['windows']:
        a=float(w['Start_s']);moving=w.get('Moving_pct',w.get('Moving_%'))
        labels=sorted({r['label'].replace('_',' ') for r in ann if (not reviewed or r.get('review_status')=='human_reviewed') and float(r['video_end_s'])+off>a and float(r['video_start_s'])+off<a+5})
        rows.append(dict(Participant=participant,Trial=trial,Start_s=a,End_s=a+5,Moving_pct=moving,
            Movement_group='Unknown' if moving is None or not np.isfinite(moving) else ('Still' if moving<10 else 'Light' if moving<50 else 'Active'),
            Activities='; '.join(labels) or 'Unlabelled',Belt_bSQI=w.get('Belt_bSQI'),Belt_tSQI=w.get('Belt_tSQI'),
            Pass_gate=w.get('Belt_good'),Bio_bSQI=w.get('Bio_bSQI')))
    return rows


def grouped(rows,mode):
    df=pd.DataFrame(rows)
    if df.empty:return pd.DataFrame()
    if mode=='Activity':df=df.assign(Group=df.Activities.str.split('; ')).explode('Group')
    else:df['Group']=df.Movement_group
    results=[]
    for name,g in df.groupby('Group',sort=True):
        valid=g.Pass_gate.dropna()
        results.append({'Group':name,'Windows':len(g),'Participants':g.Participant.nunique(),'Trials':g[['Participant','Trial']].drop_duplicates().shape[0],
            'Median bSQI':g.Belt_bSQI.median(),'Median tSQI':g.Belt_tSQI.median(),
            'Gate assessed':len(valid),'Passing gate (%)':100*valid.astype(float).mean() if len(valid) else np.nan})
    return pd.DataFrame(results)

def grouped_by_trial(rows):
    if not rows:return pd.DataFrame()
    df=pd.DataFrame(rows);out=[]
    for (p,t),g in df.groupby(['Participant','Trial']):
        good=g.Pass_gate.dropna()
        out.append({'Participant':p,'Trial':t,'5-second windows':len(g),'Average time moving (%)':g.Moving_pct.mean(),
          'Typical ECG agreement (bSQI)':g.Belt_bSQI.median(),'Typical beat similarity (tSQI)':g.Belt_tSQI.median(),
          'Windows passing checks (%)':100*good.astype(float).mean() if len(good) else np.nan})
    return pd.DataFrame(out)


class StudySummary(QtWidgets.QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.rows=[];self.queue=[];self.failures=[];self.running=False
        layout=QtWidgets.QVBoxLayout(self)
        intro=QtWidgets.QLabel('What you get: one row per 5 seconds, then ECG quality grouped by movement or video activity.');intro.setWordWrap(True);layout.addWidget(intro)
        bar=QtWidgets.QHBoxLayout();layout.addLayout(bar)
        self.scope=QtWidgets.QComboBox();self.scope.addItems(['Current trial','All trials for current participant','Selected participants','Whole study'])
        bar.addWidget(QtWidgets.QLabel('Combine'));bar.addWidget(self.scope)
        self.group=QtWidgets.QComboBox();self.group.addItems(['Movement','Activity']);bar.addWidget(QtWidgets.QLabel('Group by'));bar.addWidget(self.group)
        self.build=QtWidgets.QPushButton('Build summary');self.build.clicked.connect(self.start);bar.addWidget(self.build)
        export=QtWidgets.QPushButton('Export tables');export.clicked.connect(self.export);bar.addWidget(export)
        self.people=QtWidgets.QListWidget();self.people.setMaximumHeight(95);layout.addWidget(self.people);self.people.hide()
        self.scope.currentIndexChanged.connect(lambda i:self.people.setVisible(i==2))
        self.scope.currentIndexChanged.connect(self.refresh)
        self.people.itemChanged.connect(self.refresh)
        self.group.currentIndexChanged.connect(self.refresh)
        self.status=QtWidgets.QLabel('');self.status.setWordWrap(True);layout.addWidget(self.status)
        note=QtWidgets.QLabel('Still: <10% moving; Light: 10–<50%; Active: ≥50%. Gate uses bSQI, heart rate and R–R regularity. Activity groups can overlap, so their counts may not add up. Combined windows are descriptive, not independent study samples.');note.setWordWrap(True);layout.addWidget(note)
        self.explanation=QtWidgets.QLabel();self.explanation.setWordWrap(True);layout.addWidget(self.explanation)
        views=QtWidgets.QTabWidget();layout.addWidget(views,1)
        self.summary=QtWidgets.QTableWidget();views.addTab(self.summary,'Compare movement / activities')
        self.trials=QtWidgets.QTableWidget();views.addTab(self.trials,'Compare recordings')
        self.windows=QtWidgets.QTableWidget();self.windows.cellDoubleClicked.connect(self.inspect);views.addTab(self.windows,'Every 5-second window — double-click to inspect')
    def set_participants(self,people):
        self.people.clear()
        for p in people:
            item=QtWidgets.QListWidgetItem(p);item.setFlags(item.flags()|QtCore.Qt.ItemFlag.ItemIsUserCheckable);item.setCheckState(QtCore.Qt.CheckState.Checked);self.people.addItem(item)
    def keys(self):
        o=self.owner
        if self.scope.currentIndex()==0:return [tuple(o.data['trial'].split())] if o.data else []
        if self.scope.currentIndex()==3:return sorted(k for k,v in o.paths.items() if '(excluded)' not in str(v).lower())
        people=[o.participant.currentText()] if self.scope.currentIndex()==1 else [self.people.item(i).text() for i in range(self.people.count()) if self.people.item(i).checkState()==QtCore.Qt.CheckState.Checked]
        return sorted(k for k in o.paths if k[0] in people and '(excluded)' not in str(o.paths[k]).lower())
    def start(self):
        o=self.owner
        if o.worker and o.worker.isRunning():return
        self.requested=self.keys();self.queue=[k for k in self.requested if ' '.join(k) not in o.datasets];self.failures=[];self.running=True;self.build.setEnabled(False);o.load.setEnabled(False);self.next_trial()
    def next_trial(self):
        o=self.owner
        if not self.queue:
            self.running=False;self.build.setEnabled(True);o.load.setEnabled(True);self.refresh();return
        key=self.queue.pop(0);path=o.paths.get(key)
        if path is None:
            self.failures.append(' '.join(key));self.next_trial();return
        # Sequential calculations keep memory bounded and retain the currently viewed trial.
        import sys
        module=sys.modules[o.__class__.__module__];Worker=module.Worker
        o.worker=Worker(path,' '.join(key),module.STUDY_ROOT);self.status.setText('Calculating '+ ' '.join(key)+f' — {len(self.queue)} trials remaining…')
        o.worker.ready.connect(o.cache_trial)
        o.worker.failed.connect(lambda error:self.failures.append(' '.join(key)+': '+error))
        o.worker.finished.connect(self.next_trial);o.worker.start()
    def refresh(self,*_):
        if self.running:return
        o=self.owner;keys=self.keys();self.rows=[]
        for key in keys:
            d=o.datasets.get(' '.join(key))
            if d:self.rows+=window_rows(d,o.reviewed.isChecked(),o.offset.value() if d is o.data else None)
        self.summary_df=grouped(self.rows,self.group.currentText())
        display=self.summary_df.rename(columns={'Group':'Movement or activity','Windows':'5-second windows','Median bSQI':'Typical ECG agreement (bSQI)','Median tSQI':'Typical beat similarity (tSQI)','Gate assessed':'Windows with quality checks','Passing gate (%)':'Windows passing checks (%)'})
        self.fill(self.summary,display);self.fill(self.windows,pd.DataFrame(self.rows))
        self.explanation.setText('Read across each row: how much data is in this group, its typical ECG quality, and how often it passes the quality checks. ECG agreement and beat similarity run from 0 to 1; higher is better. Passing checks is a percentage. A dash means unavailable, not zero. Compare recordings shows each participant/trial separately.')
        self.trial_df=grouped_by_trial(self.rows)
        self.fill(self.trials,self.trial_df)
        missing=sum(' '.join(k) not in o.datasets for k in keys)
        self.status.setText(f'{len(self.rows)} windows from {len(set((r["Participant"],r["Trial"]) for r in self.rows))} trials. '+(f'{missing} trials need calculation: click Build summary. ' if missing else '')+('Failed: '+'; '.join(self.failures) if self.failures else ''))
    @staticmethod
    def fill(table,df):
        table.setSortingEnabled(False);table.setRowCount(len(df));table.setColumnCount(len(df.columns));table.setHorizontalHeaderLabels([c.replace('_',' ') for c in df.columns])
        for i,row in enumerate(df.itertuples(index=False,name=None)):
            for j,v in enumerate(row):
                text='—' if pd.isna(v) else (f'{v:.1f}%' if '%' in df.columns[j] else f'{v:.3f}') if isinstance(v,(float,np.floating)) else str(v)
                table.setItem(i,j,QtWidgets.QTableWidgetItem(text))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows);table.resizeColumnsToContents()
    def inspect(self,row,col):
        if self.running:return
        r=self.rows[row];o=self.owner;data=o.datasets[r['Participant']+' '+r['Trial']]
        o.participant.setCurrentText(r['Participant']);o.trial.setCurrentText(r['Trial']);o.loaded(data)
        o.auto_timer.stop()
        o.signals.span.setCurrentText('5');o.signals.start.setValue(r['Start_s']);o.signals.range()
        o.select(next(w for w in data['windows'] if w['Start_s']==r['Start_s']),True);o.tabs.setCurrentWidget(o.signals)
    def export(self):
        if not self.rows:return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save combined windows','','CSV (*.csv)')
        if path:
            from pathlib import Path
            p=Path(path);pd.DataFrame(self.rows).to_csv(p,index=False);self.summary_df.to_csv(p.with_name(p.stem+'_grouped.csv'),index=False)
            self.trial_df.to_csv(p.with_name(p.stem+'_recordings.csv'),index=False)
            self.status.setText('Saved window table and grouped summary beside it.')
