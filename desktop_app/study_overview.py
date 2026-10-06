"""Read-only inventory of a complete local study tree."""
from pathlib import Path
import re
import pandas as pd
from PyQt6 import QtWidgets


def inventory(folder):
    records={}
    for path in Path(folder).rglob('*'):
        if not path.is_file():continue
        p=re.search(r'(?<![A-Za-z0-9])(P\d+)(?!\d)',str(path),re.I)
        t=re.search(r'(?<![A-Za-z0-9])(T\d+)(?!\d)',str(path),re.I)
        if not p:continue
        key=(p.group(1).upper(),t.group(1).upper() if t else 'Participant files')
        records.setdefault(key,[]).append(path)
    rows=[]
    for (p,t),files in sorted(records.items()):
        names=[f.name.lower() for f in files]
        belts=[f for f in files if 'belt' in f.name.lower() and f.suffix.lower()=='.csv']
        channels=set()
        for f in belts:
            try:channels.update(pd.read_csv(f,nrows=0).columns)
            except Exception:pass
        rows.append({'Participant':p,'Trial':t,'Files':len(files),'Belt ECG':'Yes' if 'ECG' in channels else 'No',
          'Movement':'Yes' if {'GyroX','GyroY','GyroZ'}.issubset(channels) else 'No',
          'BIOPAC':'Yes' if any('biopac' in n for n in names) else 'No',
          'Video':'Yes' if any(f.suffix.lower() in {'.mp4','.mov','.avi'} for f in files) else 'No',
          'Annotations':'Yes' if any('annotation' in n and n.endswith('.csv') for n in names) else 'No',
          'PPG files':'Yes' if any('ppg' in n for n in names) else 'Not identified',
          'SKT files':'Yes' if any('skt' in n or 'temperature' in n for n in names) else 'Not identified',
          'Quality/results files':sum(any(word in n for word in ['sqi','quality','result','window']) for n in names),
          '_files':files})
    return rows


class Overview(QtWidgets.QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.records=[]
        layout=QtWidgets.QVBoxLayout(self)
        title=QtWidgets.QLabel('Your study in one place');title.setStyleSheet('font-size:22px;font-weight:600');layout.addWidget(title)
        self.location=QtWidgets.QLabel('Choose your full Organized Study Data folder to see every available participant.');self.location.setWordWrap(True);layout.addWidget(self.location)
        bar=QtWidgets.QHBoxLayout();layout.addLayout(bar)
        choose=QtWidgets.QPushButton('Choose full study folder');choose.clicked.connect(owner.browse);bar.addWidget(choose)
        analyze=QtWidgets.QPushButton('Summarise movement & ECG across the study');analyze.clicked.connect(self.analyze);bar.addWidget(analyze)
        self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.table=QtWidgets.QTableWidget();self.table.cellClicked.connect(self.files);self.table.cellDoubleClicked.connect(self.open_trial);layout.addWidget(self.table,3)
        layout.addWidget(QtWidgets.QLabel('Select a recording to see its files below. Double-click to load its ECG and movement.'))
        self.filelist=QtWidgets.QListWidget();layout.addWidget(self.filelist,1)
        note=QtWidgets.QLabel('Availability is an inventory, not a quality judgement. Existing results are listed as files; the summary computes ECG quality using the repository method. PPG/SKT quality is not calculated. Duplicate copies count as files but each participant/trial is analysed once.');note.setWordWrap(True);layout.addWidget(note)
    def scan(self,folder):
        from study_summary import StudySummary
        self.records=inventory(folder);self.location.setText(str(Path(folder).resolve()))
        StudySummary.fill(self.table,pd.DataFrame([{k:v for k,v in r.items() if k!='_files'} for r in self.records]))
        participants=len({r['Participant'] for r in self.records});trials=sum(r['Trial']!='Participant files' for r in self.records)
        self.status.setText(f'{participants} participants • {trials} trial folders • {sum(r["Files"] for r in self.records)} files. Select a row, or summarise all analysable recordings.')
    def files(self,row,col):
        self.filelist.clear();self.filelist.addItems([str(f) for f in self.records[row]['_files']])
    def open_trial(self,row,col):
        r=self.records[row];o=self.owner
        if (r['Participant'],r['Trial']) not in o.paths:
            self.status.setText('This recording has no supported BELT CSV to analyse. Its available files are listed below.');return
        o.participant.setCurrentText(r['Participant']);o.trial.setCurrentText(r['Trial']);o.load_trial();o.tabs.setCurrentWidget(o.study)
    def analyze(self):
        o=self.owner;o.study.scope.setCurrentIndex(3);o.tabs.setCurrentWidget(o.study);o.study.start()
