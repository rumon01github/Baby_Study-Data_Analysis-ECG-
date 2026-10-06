"""Local PyQt movement / ECG-quality explorer. No upload or cloud writes."""
import sys, os, json, pathlib, re, tempfile, importlib
ROOT=pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode=True
LOCAL_RUNTIME=json.loads((ROOT/'local-runtime.json').read_text(encoding='utf-8')) if (ROOT/'local-runtime.json').exists() else {}
DEMO_ROOT=pathlib.Path(LOCAL_RUNTIME.get('demo_folder',str(ROOT)))
sys.path.insert(0,LOCAL_RUNTIME.get('packages',str(ROOT/'.packages')))
os.environ.setdefault('PYQTGRAPH_QT_LIB','PyQt6')
import numpy as np, pandas as pd
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import pyqtgraph as pg
from scipy.stats import spearmanr
REPO=ROOT.parent
BLUE='#257d98'; ORANGE='#d88832'
STUDY_ROOT=None
def bundled():
    data=json.loads((DEMO_ROOT/'p03-data.js').read_text(encoding='utf-8').split('=',1)[1].strip().rstrip(';'))
    files=discover(DEMO_ROOT/'data'/'Organized Study Data')
    worker=Worker(files[('P03','T01')],'P03 T01');results=[];errors=[]
    worker.ready.connect(results.append);worker.failed.connect(errors.append);worker.run()
    if errors:raise RuntimeError(errors[0])
    results[0]['annotations']=data['annotations']
    return results[0]
def discover(folder):
    found={}
    for f in pathlib.Path(folder).rglob('*BELT*.csv'):
        match=re.search(r'(P\d+).*?(T\d+)',f.name)
        if match:
            key=match.groups()
            # Prefer the main Organized Study Data layout over curated duplicate folders.
            priority=('Good ' in str(f),len(f.parts))
            if key not in found or priority<found[key][0]: found[key]=(priority,f)
    return {k:v[1] for k,v in found.items()}
class Worker(QtCore.QThread):
    ready=QtCore.pyqtSignal(object); failed=QtCore.pyqtSignal(str)
    def __init__(self,belt,trial): super().__init__();self.belt=belt;self.trial=trial
    def run(self):
        try:
            import neurokit2 as nk
            from scipy import signal,stats
            os.environ['NEOTEX_WORK']=str(ROOT/'work'/'desktop')
            sys.path.insert(0,str(REPO))
            from common import sqi,local_lag,beats
            b=pd.read_csv(self.belt)
            required={'time_s','ECG','AccX','AccY','AccZ','GyroX','GyroY','GyroZ'}
            if not required.issubset(b.columns): raise ValueError('BELT file is missing time, ECG or movement columns.')
            if b.time_s.isna().any() or not np.all(np.diff(b.time_s)>0): raise ValueError('Recording times must increase without missing values.')
            def det(x,fs,method):
                clean=nk.ecg_clean(x,sampling_rate=fs,method='neurokit') if method=='neurokit' else x
                _,r=nk.ecg_peaks(clean,sampling_rate=fs,method=method)
                return np.asarray(r['ECG_R_Peaks'])
            tg=np.arange(b.time_s.iloc[0],b.time_s.iloc[-1],1/250)
            x=np.interp(tg,b.time_s,b.ECG)
            belt={'t0':tg[0],'fs':250,'sig':nk.ecg_clean(x,sampling_rate=250,method='neurokit'),
                  'nk':det(x,250,'neurokit'),'ham':det(x,250,'hamilton2002')}
            bio=None
            pairs=list(self.belt.parent.glob('*BIOPAC*.csv'))
            if pairs:
                bp=pd.read_csv(pairs[0]).dropna(subset=['time_s']); best=None
                for col in [c for c in bp if 'ECG MODULE' in c or 'AHA' in c]:
                    y=bp[col].to_numpy(dtype=float)[::2];k=det(y,500,'neurokit')
                    hr=60/np.diff(k/500) if len(k)>2 else np.array([0])
                    score=np.mean((hr>80)&(hr<220))*len(k)
                    if best is None or score>best[0]:best=(score,col,y,k)
                if best:
                    _,col,y,k=best
                    bio={'t0':bp.time_s.iloc[0],'fs':500,'sig':nk.ecg_clean(y,sampling_rate=500,method='neurokit'),
                         'nk':k,'ham':det(y,500,'hamilton2002')}
                    dt=float(np.median(np.diff(bp.time_s)))
                    if abs(dt-.001)>.0001:raise ValueError('BIOPAC is not on the expected 1000 Hz grid. Check sampling before using the repository method.')
            b['gyro']=np.linalg.norm(b[['GyroX','GyroY','GyroZ']],axis=1)
            b['acc_g']=np.linalg.norm(b[['AccX','AccY','AccZ']],axis=1)/4096
            base=np.percentile(b.gyro,20); end=min(b.time_s.iloc[-1],bp.time_s.iloc[-1]) if bio else b.time_s.iloc[-1]
            windows=[]
            for a in np.arange(0,end-5+1e-6,5):
                lag=local_lag(beats(belt),beats(bio),a,a+5) if bio else 0
                st=sqi(belt,a+lag,a+5+lag);sb=sqi(bio,a,a+5) if bio else {}
                sub=b[(b.time_s>=a)&(b.time_s<a+5)]
                if len(sub)<250:continue
                z=belt['sig'][max(0,int((a-belt['t0'])*250)):int((a+5-belt['t0'])*250)]
                f,p=signal.welch(z,250,nperseg=1000)
                band=lambda lo,hi:p[(f>=lo)&(f<=hi)].sum()
                share=float((sub.gyro>base+200).mean()*100)
                windows.append(dict(Start_s=float(a),Moving_pct=share,gyro_mean=float(sub.gyro.mean()),gyro_std=float(sub.gyro.std()),
                  acc_std=float(sub.acc_g.std()),Belt_bSQI=st['bsqi'],Belt_tSQI=st['tsqi'],Belt_good=st['good'],
                  Bio_bSQI=sb.get('bsqi'),kSQI=float(stats.kurtosis(z,fisher=False)),
                  pSQI=float(band(5,15)/band(5,40)) if band(5,40)>0 else None,
                  basSQI=float(1-band(0,1)/band(0,40)) if band(0,40)>0 else None))
            annotations=[]
            candidates=pathlib.Path(STUDY_ROOT).rglob('*annotations*.csv') if STUDY_ROOT else self.belt.parent.glob('*annotations*.csv')
            for af in candidates:
                saved=pd.read_csv(af).fillna('')
                if {'participant','trial','video_start_s','video_end_s','label'}.issubset(saved.columns):
                    saved=saved[(saved.participant.astype(str)+' '+saved.trial.astype(str))==self.trial]
                    annotations+=saved.to_dict('records')
            annotations=list({json.dumps(r,sort_keys=True,default=str):r for r in annotations}.values())
            self.ready.emit(dict(trial=self.trial,windows=windows,trace=b.iloc[::5][['time_s','ECG','gyro','acc_g']].to_dict('records'),annotations=annotations,source_files=(str(self.belt),str(pairs[0]) if pairs else None)))
        except Exception as e:self.failed.emit(str(e))
class Explorer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('ECG-PPG_Neotex');self.resize(1320,900)
        self.data=None;self.worker=None;self.paths={};self.current=None;self.video_name='';self.datasets={}
        pg.setConfigOptions(background='w',foreground='#243746',antialias=True)
        root=QtWidgets.QWidget();self.setCentralWidget(root);layout=QtWidgets.QVBoxLayout(root)
        heading=QtWidgets.QLabel('ECG-PPG_Neotex');heading.setStyleSheet('font-size:26px;font-weight:600;');layout.addWidget(heading)
        layout.addWidget(QtWidgets.QLabel('Select a participant and trial. Drag the plots to move through time; scroll over a plot to zoom.'))
        connection=QtWidgets.QHBoxLayout();layout.addLayout(connection)
        self.folder_status=QtWidgets.QLabel('Demo only: P03 T01. Connect your full study folder to select other recordings.');self.folder_status.setWordWrap(True);connection.addWidget(self.folder_status,1)
        connect_folder=QtWidgets.QPushButton('Change study folder…');connect_folder.clicked.connect(self.browse);connection.addWidget(connect_folder)
        bar=QtWidgets.QHBoxLayout();layout.addLayout(bar)
        browse=QtWidgets.QPushButton('Choose study folder…');browse.clicked.connect(self.browse);bar.addWidget(browse)
        self.participant=QtWidgets.QComboBox();self.trial=QtWidgets.QComboBox()
        for title,widget in [('Participant',self.participant),('Trial',self.trial)]:bar.addWidget(QtWidgets.QLabel(title));bar.addWidget(widget)
        self.load=QtWidgets.QPushButton('Load trial');self.load.clicked.connect(self.load_trial);bar.addWidget(self.load)
        self.video=QtWidgets.QPushButton('Choose matching video…');self.video.clicked.connect(self.choose_video);bar.addWidget(self.video);bar.addStretch()
        annotations=QtWidgets.QPushButton('Load annotations…');annotations.clicked.connect(self.choose_annotations);bar.insertWidget(bar.count()-1,annotations)
        self.video.hide()
        self.participant.currentTextChanged.connect(self.populate_trials)
        self.message=QtWidgets.QLabel('');self.message.setWordWrap(True);layout.addWidget(self.message)
        controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        self.metric=QtWidgets.QComboBox()
        for name,key in [('Time moving (%)','Moving_pct'),('Rotation intensity','gyro_mean'),('Rotation variability','gyro_std'),('Acceleration variability','acc_std')]:self.metric.addItem(name,key)
        self.quality=QtWidgets.QComboBox()
        for name,key in [('Beat detector agreement (bSQI)','Belt_bSQI'),('Beat shape similarity (tSQI)','Belt_tSQI'),('QRS power ratio','pSQI'),('Baseline quality','basSQI'),('Kurtosis (descriptive)','kSQI')]:self.quality.addItem(name,key)
        self.activity=QtWidgets.QComboBox();self.activity.addItem('All activity', '')
        for title,widget in [('Movement',self.metric),('ECG quality',self.quality),('Activity',self.activity)]:controls.addWidget(QtWidgets.QLabel(title));controls.addWidget(widget)
        self.offset=QtWidgets.QDoubleSpinBox();self.offset.setRange(-10000,10000);self.offset.setDecimals(2);self.offset.setValue(32.6)
        filters=QtWidgets.QHBoxLayout();layout.addLayout(filters)
        filters.addWidget(QtWidgets.QLabel('Video offset (s)'));filters.addWidget(self.offset)
        self.reviewed=QtWidgets.QCheckBox('Reviewed labels only');self.reviewed.setChecked(True);filters.addWidget(self.reviewed)
        self.covered=QtWidgets.QCheckBox('Video-labelled windows only');filters.addWidget(self.covered);filters.addStretch()
        self.result=QtWidgets.QLabel();self.result.setStyleSheet('background:#eaf4f7;padding:12px;font-size:17px;');layout.addWidget(self.result)
        self.tabs=QtWidgets.QTabWidget();layout.addWidget(self.tabs,1)
        charts=QtWidgets.QWidget();grid=QtWidgets.QGridLayout(charts)
        self.motion=pg.PlotWidget(title='Movement through the trial');self.qplot=pg.PlotWidget(title='ECG quality through the trial')
        self.scatter=pg.PlotWidget(title='Movement versus ECG quality • click a point');self.ecg=pg.PlotWidget(title='ECG in the selected 5-second window')
        self.labels=pg.PlotWidget(title='What was the baby doing? Saved activity labels')
        grid.addWidget(self.motion,0,0,1,2);grid.addWidget(self.qplot,1,0,1,2);grid.addWidget(self.labels,2,0,1,2)
        grid.addWidget(self.scatter,3,0);grid.addWidget(self.ecg,3,1)
        for plot in [self.motion,self.qplot,self.labels,self.scatter,self.ecg]:plot.showGrid(x=True,y=True,alpha=.15)
        self.qplot.setXLink(self.motion);self.labels.setXLink(self.motion)
        self.motion.setTitle('1. Movement — percentage of each 5-second window')
        self.qplot.setTitle('2. ECG quality — compare changes at the same recording time')
        self.motion.setMinimumHeight(145);self.qplot.setMinimumHeight(145);self.labels.setMinimumHeight(125)
        self.tabs.addTab(charts,'Movement & quality')
        viewer=QtWidgets.QWidget();vl=QtWidgets.QVBoxLayout(viewer)
        self.vwidget=QVideoWidget();self.vwidget.setMinimumHeight(260);vl.addWidget(self.vwidget,1)
        self.player=QMediaPlayer();self.audio=QAudioOutput();self.audio.setVolume(.6);self.player.setAudioOutput(self.audio);self.player.setVideoOutput(self.vwidget)
        vb=QtWidgets.QHBoxLayout();vl.addLayout(vb);play=QtWidgets.QPushButton('Play / pause');play.clicked.connect(self.play);vb.addWidget(play)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);vb.addWidget(self.slider,1);self.clock=QtWidgets.QLabel();vb.addWidget(self.clock)
        self.slider.sliderMoved.connect(self.player.setPosition);self.player.durationChanged.connect(self.slider.setMaximum);self.player.positionChanged.connect(self.position)
        self.player.errorOccurred.connect(lambda *_:self.message.setText('Video playback error: '+self.player.errorString()))
        self.table=QtWidgets.QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['Video interval','Activity','Review status','Notes']);self.table.horizontalHeader().setSectionResizeMode(3,QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers);self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows);self.table.cellClicked.connect(self.annotation_click);vl.addWidget(self.table,1)
        self.vwidget.hide();play.hide();self.slider.hide();self.clock.hide()
        annotation_note=QtWidgets.QLabel('Saved activity labels and time intervals. No video required. Annotation time + offset = recording time.');annotation_note.setWordWrap(True);vl.insertWidget(0,annotation_note)
        self.table.cellClicked.disconnect(self.annotation_click)
        self.tabs.addTab(viewer,'Saved activity annotations')
        from study_summary import StudySummary
        self.study=StudySummary(self);self.tabs.addTab(self.study,'Windows & grouped quality')
        from study_overview import Overview
        self.overview=Overview(self);self.tabs.addTab(self.overview,'Whole study overview')
        from signal_stack import SignalStack
        self.signals=SignalStack(self);self.tabs.addTab(self.signals,'ECG & movement — shared time')
        footer=QtWidgets.QHBoxLayout();layout.addLayout(footer)
        helpbtn=QtWidgets.QPushButton('What does bSQI mean?');helpbtn.clicked.connect(self.help);footer.addWidget(helpbtn)
        export=QtWidgets.QPushButton('Export visible windows…');export.clicked.connect(self.export);footer.addWidget(export)
        self.selected=QtWidgets.QLabel();footer.addWidget(self.selected,1)
        self.simple_hidden=[browse,self.load,annotations,self.result,self.selected,helpbtn,export]
        for row in [controls,filters]:
            for i in range(row.count()):
                widget=row.itemAt(i).widget()
                if widget:self.simple_hidden.append(widget)
        for widget in self.simple_hidden:widget.hide()
        self.tabs.tabBar().hide()
        setup=self.menuBar().addMenu('Study setup')
        action=setup.addAction('Choose or change study folder…');action.triggered.connect(self.browse)
        action=setup.addAction('Load matching annotations…');action.triggered.connect(self.choose_annotations)
        advanced=self.menuBar().addAction('Show legacy analysis (different movement method)');advanced.setCheckable(True);advanced.toggled.connect(self.show_analysis)
        self.auto_timer=QtCore.QTimer(self);self.auto_timer.setSingleShot(True);self.auto_timer.setInterval(200);self.auto_timer.timeout.connect(self.auto_load)
        self.trial.currentTextChanged.connect(lambda *_:self.auto_timer.start())
        for w in [self.metric,self.quality,self.activity]:w.currentIndexChanged.connect(self.redraw)
        self.offset.valueChanged.connect(self.redraw);self.reviewed.toggled.connect(self.redraw);self.covered.toggled.connect(self.redraw)
        if (DEMO_ROOT/'p03-data.js').exists():
            self.populate({('P03','T01'):None});self.load_trial()
        else:
            self.populate({});self.message.setText('Use Study setup to choose your local study folder.')
        if (DEMO_ROOT/'data').is_dir():self.overview.scan(DEMO_ROOT/'data')
        self.tabs.setCurrentWidget(self.signals)
        connected=False
        try:
            settings=ROOT/'study-settings.json'
            if not settings.exists():settings=DEMO_ROOT/'study-settings.json'
            folder=json.loads(settings.read_text(encoding='utf-8-sig')).get('folder')
            if folder and pathlib.Path(folder).is_dir():connected=self.use_folder(folder)
            elif folder:self.folder_status.setText('Saved study folder is unavailable. Use Change study folder to reconnect it.')
        except (OSError,ValueError):pass
        if not connected:
            default=pathlib.Path(LOCAL_RUNTIME.get('study_folder',str(REPO/'data'/'Organized Study Data')))
            if default.is_dir():self.use_folder(default)
    def show_analysis(self,shown):
        for widget in self.simple_hidden:widget.setVisible(shown)
        self.tabs.tabBar().setVisible(shown)
        if not shown:self.tabs.setCurrentWidget(self.signals)
    def auto_load(self):
        if not self.trial.currentText():return
        if self.worker and self.worker.isRunning():
            self.auto_timer.start();return
        self.load_trial()
    def populate(self,paths):
        self.paths=paths;self.participant.blockSignals(True);self.trial.blockSignals(True)
        self.participant.clear();self.participant.addItems(sorted(set(p for p,t in paths)));self.populate_trials()
        self.participant.blockSignals(False);self.trial.blockSignals(False)
        self.study.set_participants(sorted(set(p for p,t in paths)))
    def populate_trials(self,*_):
        self.trial.clear();self.trial.addItems(sorted(t for p,t in self.paths if p==self.participant.currentText()))
    def browse(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Choose Organized Study Data or BABY DATA',str(STUDY_ROOT or REPO/'data'))
        if folder:self.use_folder(folder)
    def use_folder(self,folder):
        global STUDY_ROOT
        found=discover(folder)
        if not found:QtWidgets.QMessageBox.information(self,'No recordings','No participant/trial BELT CSV files were found.');return
        STUDY_ROOT=folder;self.datasets={};self.populate(found);self.overview.scan(folder);self.tabs.setCurrentWidget(self.signals)
        self.folder_status.setText(f'{len({p for p,t in found})} participants • {len(found)} trials • {folder}')
        if len(found)==1:self.folder_status.setText(f'Only {next(iter(found))[0]} {next(iter(found))[1]} is in this folder. Change study folder to the full dataset to access other recordings.')
        (ROOT/'study-settings.json').write_text(json.dumps({'folder':str(folder)}),encoding='utf-8')
        self.message.setText(f'{len(found)} recordings available. Loading selected trial…');self.auto_timer.start()
        return True
    def load_trial(self):
        if self.worker and self.worker.isRunning():return
        key=(self.participant.currentText(),self.trial.currentText());path=self.paths.get(key)
        if ' '.join(key) in self.datasets:
            self.loaded(self.datasets[' '.join(key)]);return
        if path is None and key==('P03','T01'):
            self.loaded(bundled());return
        if path is None:return
        self.signals.hide();self.data=None
        self.load.setEnabled(False);self.message.setText('Calculating ECG quality and movement. This may take a minute…')
        self.worker=Worker(path,' '.join(key));self.worker.ready.connect(self.accept_trial);self.worker.failed.connect(self.failed);self.worker.finished.connect(lambda:self.load.setEnabled(True));self.worker.start()
    def accept_trial(self,data):
        self.datasets[data['trial']]=data
        if data['trial']==' '.join((self.participant.currentText(),self.trial.currentText())):self.loaded(data)
    def failed(self,text):self.message.setText('Could not load this trial: '+text)
    def loaded(self,data):
        from annotation_loader import latest_annotations
        latest=latest_annotations(ROOT,data['trial'])
        if latest is not None:
            data=dict(data,annotations=latest,annotations_synced=True)
            self.reviewed.setChecked(False)
        self.signals.show()
        if 'signals' not in data and data.get('source_files'):
            from signal_stack import signal_payload
            data['signals']=signal_payload(*data['source_files'])
        self.player.stop();self.player.setSource(QtCore.QUrl());self.data=data
        self.datasets[data['trial']]=data
        for w in data['windows']:
            if 'Moving_pct' not in w:w['Moving_pct']=w.get('Moving_%')
        ann=data.get('annotations',[])
        offsets=[float(a['ecg_start_s'])-float(a['video_start_s']) for a in ann if str(a.get('ecg_start_s','')).strip()]
        self.offset.setValue(0 if data.get('annotations_synced') else (float(np.median(offsets)) if offsets else 0))
        self.offset.setEnabled(not data.get('annotations_synced',False))
        self.activity.clear();self.activity.addItem('All activity','')
        for label in sorted(set(a['label'] for a in ann)):self.activity.addItem(label.replace('_',' '),label)
        self.table.setRowCount(len(ann))
        for i,a in enumerate(ann):
            for j,value in enumerate([f"{a['video_start_s']}–{a['video_end_s']} s",a['label'].replace('_',' '),a.get('review_status','unknown'),a.get('notes','')]):self.table.setItem(i,j,QtWidgets.QTableWidgetItem(str(value)))
        self.message.setText('')
        if data['windows']:self.select(data['windows'][0],False)
        self.signals.load(data)
        self.redraw()
        key=tuple(data['trial'].split());path=self.paths.get(key)
        if path:
            pass
    def events(self,w):
        if not self.data:return []
        off=self.offset.value()
        return [a for a in self.data.get('annotations',[]) if (not self.reviewed.isChecked() or a.get('review_status')=='human_reviewed') and float(a['video_end_s'])+off>w['Start_s'] and float(a['video_start_s'])+off<w['Start_s']+5]
    def redraw(self,*_):
        if not self.data:return
        x=self.metric.currentData();y=self.quality.currentData()
        if not x or not y:return
        label=self.activity.currentData()
        self.visible=[w for w in self.data['windows'] if w.get(x) is not None and w.get(y) is not None and np.isfinite(w[x]) and np.isfinite(w[y]) and (not self.covered.isChecked() or self.events(w)) and (not label or any(a['label']==label for a in self.events(w)))]
        n=len(self.visible);xx=[w[x] for w in self.visible];yy=[w[y] for w in self.visible]
        rho=float(spearmanr(xx,yy).statistic) if n>=3 and len(set(xx))>1 and len(set(yy))>1 else None
        self.result.setText(f'{n} windows • Movement–quality association: '+(f'Spearman ρ = {rho:.2f}' if rho is not None else 'not enough variation to calculate')+' • One trial; exploratory')
        self.motion.clear();self.qplot.clear();self.scatter.clear()
        w=self.data['windows'];times=[a['Start_s']+2.5 for a in w]
        self.motion.plot(times,[a['Moving_pct'] for a in w],pen=BLUE,symbol='o',symbolSize=7);self.motion.setLabel('left','Time moving (%)');self.motion.setYRange(0,100);self.motion.setLabel('bottom','Recording time (s)')
        self.qplot.addLegend();self.qplot.plot(times,[a[y] if a.get(y) is not None else np.nan for a in w],pen=BLUE,name='Belt')
        if y=='Belt_bSQI':self.qplot.plot(times,[a.get('Bio_bSQI') if a.get('Bio_bSQI') is not None else np.nan for a in w],pen=ORANGE,name='BIOPAC reference')
        self.qplot.setLabel('left',self.quality.currentText());self.qplot.setLabel('bottom','Recording time (s)')
        self.labels.clear()
        annotations=[a for a in self.data.get('annotations',[]) if not self.reviewed.isChecked() or a.get('review_status')=='human_reviewed']
        names=sorted({a['label'] for a in annotations});self.labels.getAxis('left').setTicks([[(i,name.replace('_',' ')) for i,name in enumerate(names)]])
        for a in annotations:
            lane=names.index(a['label']);start=float(a['video_start_s'])+self.offset.value();end=float(a['video_end_s'])+self.offset.value()
            self.labels.plot([start,end],[lane,lane],pen=pg.mkPen(pg.intColor(lane,max(1,len(names))),width=9))
        self.labels.setYRange(-.6,max(.6,len(names)-.4));self.labels.setLabel('bottom','Recording time (s) — aligned with movement and ECG above')
        if not names:
            text=pg.TextItem('No saved activity annotations loaded',color=BLUE);self.labels.addItem(text)
        self.motion.setXRange(0,max(10,max(times)+5))
        for plot in [self.motion,self.qplot,self.labels]:plot.getAxis('left').setWidth(190)
        if self.current:self.highlight_window(self.current)
        spots=[dict(pos=(a[x],a[y]),data=a,brush=BLUE) for a in self.visible];item=pg.ScatterPlotItem(spots,size=10,pen=None);item.sigClicked.connect(self.point_clicked);self.scatter.addItem(item)
        self.scatter.setLabel('bottom',self.metric.currentText());self.scatter.setLabel('left',self.quality.currentText())
        self.study.refresh()
        self.signals.annotations()
    def point_clicked(self,item,points,*_):
        if points:self.select(points[0].data(),True)
    def select(self,w,seek=True):
        self.current=w;self.ecg.clear();trace=[r for r in self.data['trace'] if w['Start_s']<=r['time_s']<w['Start_s']+5]
        self.highlight_window(w)
        self.ecg.plot([r['time_s'] for r in trace],[r['ECG'] for r in trace],pen=BLUE);self.ecg.setLabel('left','Belt ECG (mV)');self.ecg.setLabel('bottom','Recording time (s)')
        self.selected.setText(f"{w['Start_s']:.0f}–{w['Start_s']+5:.0f} s • moving {w['Moving_pct']:.1f}% • bSQI {w['Belt_bSQI']:.3f}")
        if seek and not self.player.source().isEmpty():self.player.setPosition(max(0,int((w['Start_s']-self.offset.value())*1000)))
    def highlight_window(self,w):
        for plot in [self.motion,self.qplot,self.labels]:
            old=getattr(plot,'window_highlight',None)
            if old is not None:plot.removeItem(old)
            region=pg.LinearRegionItem([w['Start_s'],w['Start_s']+5],movable=False,brush=(37,125,152,25),pen=pg.mkPen(BLUE,width=1))
            region.setZValue(-10);plot.addItem(region);plot.window_highlight=region
    def choose_video(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Choose the video for the loaded participant and trial','','Videos (*.mp4 *.mov *.avi)')
        if path:
            key=self.data['trial'].split() if self.data else []
            if len(key)==2 and not all(k in pathlib.Path(path).name for k in key):
                QtWidgets.QMessageBox.information(self,'Check recording','Choose a video named for the loaded participant and trial to avoid mismatched data.');return
            self.set_video(path)
    def set_video(self,path):
        if pathlib.Path(path).exists():self.player.setSource(QtCore.QUrl.fromLocalFile(str(pathlib.Path(path).resolve())));self.video_name=str(path);self.clock.setText('Video loaded')
    def choose_annotations(self):
        if not self.data:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Choose annotations for the loaded trial','','Annotations (*.csv)')
        if not path:return
        try:
            a=pd.read_csv(path).fillna('')
            required={'participant','trial','video_start_s','video_end_s','label'}
            if not required.issubset(a.columns):raise ValueError('Annotation CSV needs participant, trial, label and video start/end seconds.')
            if set(a.participant.astype(str)+' '+a.trial.astype(str))!={self.data['trial']}:raise ValueError('Annotations belong to a different participant or trial.')
            start=pd.to_numeric(a.video_start_s,errors='raise');end=pd.to_numeric(a.video_end_s,errors='raise')
            if not ((start>=0)&(end>start)).all():raise ValueError('Annotation intervals must have valid increasing times.')
            old_video=self.player.source();updated=dict(self.data);updated['annotations']=a.to_dict('records');self.loaded(updated)
            if not old_video.isEmpty():self.player.setSource(old_video)
        except Exception as e:QtWidgets.QMessageBox.information(self,'Could not load annotations',str(e))
    def play(self):
        if self.player.source().isEmpty():self.choose_video();return
        if self.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState:self.player.pause()
        else:self.player.play()
    def position(self,ms):
        if not self.slider.isSliderDown():self.slider.setValue(ms)
        self.clock.setText(f'Video {ms/1000:.1f} s • recording {ms/1000+self.offset.value():.1f} s')
        if self.data:
            t=ms/1000+self.offset.value()
            w=next((w for w in self.data['windows'] if w['Start_s']<=t<w['Start_s']+5),None)
            if w is not None and w is not self.current:self.select(w,False)
    def annotation_click(self,row,column):
        a=self.data['annotations'][row];self.player.setPosition(int(float(a['video_start_s'])*1000))
    def export(self):
        if not getattr(self,'visible',None):return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save window table',self.data['trial'].replace(' ','_')+'_movement_quality.csv','CSV (*.csv)')
        if path:pd.DataFrame(self.visible).to_csv(path,index=False)
    def help(self):
        QtWidgets.QMessageBox.information(self,'Understanding quality','bSQI asks: do two ECG beat detectors find the same beats?\n\n1 = close agreement. 0 = no agreement. It compares NeuroKit and Hamilton on the SAME ECG, with a ±75 ms tolerance. Belt and BIOPAC bSQIs are calculated separately.\n\nMovement does not enter bSQI. High bSQI is consistency, not proof of accuracy. The repository reliability gate also checks HR and R–R regularity.\n\nPositive correlation: more movement accompanies higher SQI. Negative: more movement accompanies lower SQI. This is one trial, not a causal or population result.\n\nVideo time + offset = recording time. Annotation status reflects the saved review flags; event timing remains approximate.\n\nPPG and skin-temperature quality are not included yet.')
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self,'Analysis running','Please wait for the current trial calculation before closing.');event.ignore();return
        self.player.stop();event.accept()
if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv);app.setStyle('Fusion')
    app.setStyleSheet('QWidget {font-size:13px;} QPushButton {padding:8px 14px;} QComboBox {padding:5px;}')
    win=Explorer();win.show();sys.exit(app.exec())



