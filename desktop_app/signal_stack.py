"""Shared-time raw signal display. Preserve peaks using pyqtgraph peak downsampling."""
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6 import QtWidgets
import pyqtgraph as pg
from imu_processing import process_imu, summary as movement_summary


def signal_payload(belt_path,bio_path=None):
    b=pd.read_csv(belt_path)
    groups=[]
    for col in ['ECG','IR','Red','AccX','AccY','AccZ','GyroX','GyroY','GyroZ']:
        if col in b:groups.append({'name':'Baby belt '+col,'unit':'mV' if col=='ECG' else 'raw units','time':b.time_s.to_numpy(),'values':b[col].to_numpy()})
    end=float(b.time_s.iloc[-1]);start=float(b.time_s.iloc[0]);durations={'Baby belt':(start,end)}
    if bio_path:
        bp=pd.read_csv(bio_path)
        durations['BIOPAC']=(float(bp.time_s.iloc[0]),float(bp.time_s.iloc[-1]))
        for col in bp:
            if 'ECG MODULE' in col or 'AHA' in col:
                groups.append({'name':'BIOPAC '+col,'unit':'mV','time':bp.time_s.to_numpy(),'values':bp[col].to_numpy()})
            elif 'PPG MODULE' in col:
                groups.append({'name':'BIOPAC '+col,'unit':'V' if '[Volts]' in col else 'source units','time':bp.time_s.to_numpy(),'values':bp[col].to_numpy()})
    return {'channels':groups,'durations':durations}


class SignalStack(QtWidgets.QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.plots=[];self.data=None;self.updating=False
        layout=QtWidgets.QVBoxLayout(self)
        self.info=QtWidgets.QLabel('Load a trial to inspect raw ECG and movement together.');self.info.setWordWrap(True);layout.addWidget(self.info)
        controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        controls.addWidget(QtWidgets.QLabel('Signal'))
        self.signal_kind=QtWidgets.QComboBox();self.signal_kind.addItems(['ECG','PPG']);controls.addWidget(self.signal_kind)
        self.ppg_channel=QtWidgets.QComboBox();self.ppg_channel.addItems(['IR','Red']);self.ppg_channel.setToolTip('Baby belt PPG channel');controls.addWidget(self.ppg_channel);self.ppg_channel.hide()
        self.signal_kind.currentTextChanged.connect(self.refresh_waveforms)
        self.ppg_channel.currentTextChanged.connect(self.refresh_waveforms)
        controls.addWidget(QtWidgets.QLabel('Start'));self.start=QtWidgets.QDoubleSpinBox();self.start.setRange(0,100000);controls.addWidget(self.start)
        controls.addWidget(QtWidgets.QLabel('Show seconds'));self.span=QtWidgets.QComboBox();self.span.addItems(['5','10','30','60','Full recording']);controls.addWidget(self.span)
        for name,delta in [('Previous',-1),('Next',1)]:
            btn=QtWidgets.QPushButton(name);btn.clicked.connect(lambda checked=False,d=delta:self.step(d));controls.addWidget(btn)
        self.start.valueChanged.connect(self.range);self.span.currentIndexChanged.connect(self.range)
        self.centre=QtWidgets.QCheckBox('Centre acceleration for display');self.centre.setChecked(True);controls.addWidget(self.centre);self.centre.toggled.connect(self.update_imu_display)
        self.centre.hide()
        self.centre.setToolTip('Subtract each axis trial median for visibility. Original data and movement estimates are unchanged.')
        self.start.setSuffix(' s')
        controls.addWidget(QtWidgets.QLabel('Window'))
        self.window=QtWidgets.QComboBox();controls.addWidget(self.window);self.window.currentIndexChanged.connect(self.choose_window)
        fit=QtWidgets.QPushButton('Full recording');fit.clicked.connect(self.full_range);controls.addWidget(fit);controls.addStretch()
        self.span.setCurrentText('5')
        threshold_row=QtWidgets.QHBoxLayout();layout.addLayout(threshold_row)
        self.rotation_threshold=QtWidgets.QDoubleSpinBox();self.acc_threshold=QtWidgets.QDoubleSpinBox()
        for name,widget in [('Rotation threshold (raw)',self.rotation_threshold),('Acc change threshold (raw)',self.acc_threshold)]:
            widget.setRange(.001,1e9);widget.setDecimals(2);threshold_row.addWidget(QtWidgets.QLabel(name));threshold_row.addWidget(widget);widget.valueChanged.connect(self.threshold_changed)
        self.baseline_button=QtWidgets.QPushButton('Recalculate automatic reference')
        threshold_row.addWidget(self.baseline_button);self.baseline_button.clicked.connect(self.select_baseline)
        self.baseline_status=QtWidgets.QLabel('Quiet reference not selected');self.baseline_status.setWordWrap(True)
        layout.addWidget(self.baseline_status);self.baseline_status.hide()
        for i in range(threshold_row.count()):
            widget=threshold_row.itemAt(i).widget()
            if widget:widget.hide()
        for widget in (self.rotation_threshold,self.acc_threshold):widget.setReadOnly(True);widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        for widget in (self.rotation_threshold,self.acc_threshold):widget.setMaximumWidth(130);widget.setSpecialValueText('Not set')
        self.body=QtWidgets.QWidget();self.stack=QtWidgets.QVBoxLayout(self.body);self.stack.setContentsMargins(0,0,0,0);layout.addWidget(self.body,1)
        self.interval=QtWidgets.QLabel();layout.addWidget(self.interval)
        self.activity_legend=QtWidgets.QLabel();self.activity_legend.setWordWrap(True);self.activity_legend.setTextFormat(pg.QtCore.Qt.TextFormat.RichText);layout.insertWidget(2,self.activity_legend)
        self.shading_key=QtWidgets.QLabel();self.shading_key.setWordWrap(True);self.shading_key.setTextFormat(pg.QtCore.Qt.TextFormat.RichText);layout.insertWidget(3,self.shading_key)
        self.imu_summary=QtWidgets.QLabel();self.imu_summary.setWordWrap(True);layout.insertWidget(4,self.imu_summary)
        self.annotation_timeline=pg.PlotWidget(title='Activity timing')
        self.annotation_timeline.setMinimumHeight(90)
        self.annotation_timeline.setMaximumHeight(230)
        self.annotation_timeline.getAxis('left').setWidth(165)
        self.annotation_timeline.getAxis('right').setWidth(110)
        self.annotation_timeline.showAxis('right');self.annotation_timeline.getAxis('right').setStyle(showValues=False)
        self.annotation_timeline.hideAxis('bottom')
        self.annotation_timeline.showGrid(x=True,y=False,alpha=.15)
        self.annotation_timeline.setMouseEnabled(x=True,y=False)
        layout.insertWidget(layout.indexOf(self.body),self.annotation_timeline)
        note=QtWidgets.QLabel('Green: acceleration · Purple: gyro · Dashed line: movement threshold');layout.addWidget(note)
    def add(self,title,unit):
        p=pg.PlotWidget(title=title);p.setMinimumHeight(75);p.getAxis('left').setWidth(165);p.setLabel('left',unit);p.setLabel('bottom','Recording time (s)');p.showGrid(x=True,y=True,alpha=.15)
        if self.plots:p.setXLink(self.plots[0])
        self.stack.addWidget(p);self.plots.append(p);return p
    def load(self,data):
        self.data=data
        self.reference=None
        self.baseline_status.setText('Selecting the lowest-motion reference from all valid 5-second windows in this trial.')
        for widget in (self.rotation_threshold,self.acc_threshold):
            widget.blockSignals(True);widget.setValue(widget.minimum());widget.blockSignals(False)
        if hasattr(self,'acc_view'):
            self.acc_view.scene().removeItem(self.acc_view);del self.acc_view
        while self.stack.count():
            item=self.stack.takeAt(0)
            if item.widget():item.widget().deleteLater()
        self.annotation_regions=[];self.plots=[];signals=data.get('signals',{});channels=signals.get('channels',[])
        durations=signals.get('durations',{})
        endpoints=[float(v[1]) for v in durations.values()]
        self.end=max(endpoints) if endpoints else max((w['Start_s']+5 for w in data['windows']),default=10)
        durations_text='; '.join(f'{name}: {ends[0]:.2f}–{ends[1]:.2f} s ({(ends[1]-ends[0])/60:.2f} min)' for name,ends in durations.items())
        self.info.setText(f'{data["trial"]} · {self.end:.0f} s');self.info.setToolTip(durations_text)
        w=data['windows']
        imu=self.add('Raw IMU — acceleration solid; gyroscope dashed; X blue / Y orange / Z purple','Raw sensor counts')
        self.annotation_timeline.setXLink(imu)
        self.imu_curves=[];self.motion_time=None;imu.addLegend()
        acc=[next((c for c in channels if c['name']=='Baby belt Acc'+axis),None) for axis in ['X','Y','Z']]
        gyro=[next((c for c in channels if c['name']=='Baby belt Gyro'+axis),None) for axis in ['X','Y','Z']]
        if all(c is not None for c in acc+gyro):
            self.motion_time=acc[0]['time'];av=np.column_stack([c['values'] for c in acc]);gv=np.column_stack([c['values'] for c in gyro])
            self.av=av;self.gv=gv;self.raw_motion_time=self.motion_time.copy()
            self.raw_imu_curves=[]
            for prefix,values,style in [('Acc',av,pg.QtCore.Qt.PenStyle.SolidLine),('Gyro',gv,pg.QtCore.Qt.PenStyle.DashLine)]:
                for i,(axis,color) in enumerate(zip('XYZ',['#257d98','#d88832','#7255a0'])):
                    curve=imu.plot(self.motion_time,values[:,i],pen=pg.mkPen(color,width=2.5,style=style),name=prefix+' '+axis)
                    curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True);self.raw_imu_curves.append(curve)
            self.acc_motion=np.full(len(av),np.nan);self.rotation_motion=np.full(len(gv),np.nan)
            self.rotation_curve=imu.plot([],[],pen=pg.mkPen('#7255a0',width=2.5),name='Gyro')
            self.acc_curve=imu.plot([],[],pen=pg.mkPen('#398649',width=2.5),name='Acc')
            for curve in [self.rotation_curve,self.acc_curve]:curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
            self.threshold_line=pg.InfiniteLine(1,angle=0,movable=False,pen=pg.mkPen('#aa3333',width=2,style=pg.QtCore.Qt.PenStyle.DashLine))
            imu.addItem(self.threshold_line);self.threshold_line.hide()
        else:
            imu.setTitle('IMU unavailable — one or more Acc/Gyro XYZ channels are missing')
        self.baseline_button.setEnabled(self.motion_time is not None)
        self.waveform_plots={};self.waveform_curves={}
        for device in ['Baby belt','BIOPAC']:
            self.waveform_plots[device]=self.add(device,'')
        self.refresh_waveforms()
        self.annotation_regions=[]
        self.window.blockSignals(True);self.window.clear()
        for i,r in enumerate(w):self.window.addItem(f"{i+1}: {r['Start_s']:.0f}–{r['Start_s']+5:.0f} s",r['Start_s'])
        self.window.blockSignals(False)
        self.plots[0].getViewBox().sigXRangeChanged.connect(self.describe_range)
        for plot in self.plots[:-1]:plot.hideAxis('bottom')
        # Reserve the same right margin on every row to align both ends of the shared time axis.
        for plot in self.plots:
            plot.getAxis('right').setWidth(110)
        for plot in self.plots[1:]:plot.showAxis('right');plot.getAxis('right').setStyle(showValues=False)
        self.update_imu_display()
        self.annotations();self.start.setMaximum(max(0,self.end));self.start.setValue(0);self.range()
        for plot in self.plots:
            for r in w:
                line=pg.InfiniteLine(r['Start_s'],angle=90,movable=False,pen=pg.mkPen((140,140,140,65),style=pg.QtCore.Qt.PenStyle.DotLine));line.setZValue(-15);plot.addItem(line)
        if self.motion_time is not None:self.select_baseline()
    def refresh_waveforms(self,*_):
        ppg=self.signal_kind.currentText()=='PPG';self.ppg_channel.setVisible(ppg)
        if not self.data or not getattr(self,'waveform_plots',None):return
        channels=self.data.get('signals',{}).get('channels',[])
        for device,color in [('Baby belt','#257d98'),('BIOPAC','#d88832')]:
            plot=self.waveform_plots[device]
            old=self.waveform_curves.pop(device,None)
            if old is not None:plot.removeItem(old)
            if ppg:
                eligible=[c for c in channels if (c['name']=='Baby belt '+self.ppg_channel.currentText() if device=='Baby belt' else c['name'].startswith('BIOPAC ') and 'PPG MODULE' in c['name'])]
            else:
                eligible=[c for c in channels if c['name'].startswith(device) and 'ECG' in c['name']]
            if not eligible:
                plot.setTitle(device+' '+self.signal_kind.currentText()+' — unavailable');plot.setLabel('left','');continue
            channel=eligible[0]
            title=(device+' PPG — '+self.ppg_channel.currentText()) if ppg and device=='Baby belt' else channel['name']
            plot.setTitle(title);plot.setToolTip(channel['name']+' — saved CSV values, no filtering or resampling')
            plot.setLabel('left',channel['unit'])
            curve=plot.plot(channel['time'],channel['values'],pen=pg.mkPen(color,width=2))
            curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
            self.waveform_curves[device]=curve
            plot.enableAutoRange(axis='y');plot.setAutoVisible(y=True)
    def choose_window(self,*_):
        t=self.window.currentData()
        if t is None:return
        self.span.setCurrentText('5');self.start.setValue(float(t));self.range()
    def describe_range(self,*_):
        if not self.plots:return
        lo,hi=self.plots[0].viewRange()[0]
        self.draw_annotation_timeline(lo,hi)
        self.scale_imu(lo,hi)
        selected=self.window.findData(round(lo)) if abs((hi-lo)-5)<.01 and abs(lo-round(lo))<.01 else -1
        self.window.blockSignals(True);self.window.setCurrentIndex(selected);self.window.blockSignals(False)
        import html
        visible=[f'<span style="color:{color}">■</span> {html.escape(label)} ({start:.1f}–{end:.1f} s)' for start,end,label,color in getattr(self,'active_annotations',[]) if end>lo and start<hi]
        summaries=[]
        if hi-lo<=60.01:
            for start in np.arange(max(0,np.floor(lo/5)*5),hi-.0001,5):
                end=start+5;coverage={}
                for label in {a[2] for a in getattr(self,'active_annotations',[])}:
                    intervals=sorted((max(start,a),min(end,b)) for a,b,l,c in self.active_annotations if l==label and b>start and a<end)
                    total=0.;last=start
                    for a,b in intervals:
                        total+=max(0,b-max(last,a));last=max(last,b)
                    if total:coverage[label]=total
                if coverage:
                    colors={label:color for a,b,label,color in self.active_annotations}
                    detail=' &nbsp; | &nbsp; '.join(f'<span style="color:{colors[label]}">■</span> {html.escape(label)} ({value:.1f}s)' for label,value in sorted(coverage.items(),key=lambda x:-x[1]))
                    summaries.append(f'<b>{start:.0f}–{end:.0f}s</b> · {detail}')
                else:
                    summaries.append(f'<b>{start:.0f}–{end:.0f}s</b> · Not annotated')
                off=0 if self.data.get('annotations_synced') else self.owner.offset.value()
                rows=[a for a in getattr(self,'visible_annotation_rows',[]) if float(a['video_end_s'])+off>start and float(a['video_start_s'])+off<end]
                def crying_value(value):
                    value=str(value).strip().lower()
                    if value in {'yes','true','1','crying'}:return 'Yes'
                    if value in {'no','false','0','not_crying','not crying'}:return 'No'
                    return 'Unknown'
                states={crying_value(a.get('crying','')) for a in rows}
                crying=next(iter(states)) if len(states)==1 else ('Mixed / uncertain' if states else 'Not annotated')
                summaries[-1]+=f' &nbsp; | &nbsp; <b>Crying: {crying}</b>'
        self.activity_legend.setText('Activity: '+'<br>'.join(summaries) if summaries else 'Activity: zoom in to see labels')
        self.shading_key.setText(' · '.join(f'<span style="color:{color}">■</span> {html.escape(label)}' for label,color in sorted({(label,color) for start,end,label,color in getattr(self,'active_annotations',[]) if end>lo and start<hi})))
        self.shading_key.hide()
        summaries=[]
        if self.reference is not None and hi-lo<=60.01:
            for start in np.arange(max(0,np.floor(lo/5)*5),hi-.0001,5):
                mask=(self.motion_time>=start)&(self.motion_time<start+5)
                a=self.acc_motion[mask];g=self.rotation_motion[mask]
                valid=np.isfinite(a)&np.isfinite(g)
                if valid.sum()<8:
                    text='Insufficient IMU data'
                else:
                    moving=(a[valid]>self.acc_threshold.value())|(g[valid]>self.rotation_threshold.value())
                    label='Movement detected' if moving.any() else 'No movement detected'
                    text=f'{label} ({moving.mean()*100:.0f}%) · Acc RMS {np.sqrt(np.mean(a[valid]**2)):.1f} · Gyro RMS {np.sqrt(np.mean(g[valid]**2)):.1f} raw units'
                summaries.append(f'{start:.0f}–{start+5:.0f}s · {text}')
        self.imu_summary.setText('IMU: '+('\n'.join(summaries) if summaries else ('Zoom in for movement summary' if self.reference is not None else 'Raw signals only — movement estimate unavailable')))
    def select_baseline(self):
        if self.motion_time is None:return
        try:
            reference=process_imu(self.raw_motion_time,self.av,self.gv)
        except ValueError as e:
            self.baseline_status.setText(str(e)+' Raw IMU remains visible.');return
        self.reference=reference
        for curve in self.raw_imu_curves:self.plots[0].removeItem(curve)
        self.raw_imu_curves=[]
        self.plots[0].setTitle('IMU movement strength')
        self.plots[0].setLabel('left','Threshold ratio');self.threshold_line.show()
        self.motion_time=reference['time'];self.acc_motion=reference['acc'];self.rotation_motion=reference['gyro']
        for widget,key in [(self.acc_threshold,'acc_threshold'),(self.rotation_threshold,'gyro_threshold')]:
            widget.blockSignals(True);widget.setDecimals(6);widget.setValue(reference[key]);widget.blockSignals(False)
        self.baseline_status.setText(f'Automatic reference: {len(reference["reference_starts"])*.5:.1f}s of low-motion intervals pooled from this trial. Acc + gyro RMS every 0.5s; results every 5s. Exploratory: low motion is not verified rest.')
        self.threshold_changed()
    def draw_annotation_timeline(self,lo,hi):
        timeline=self.annotation_timeline
        timeline.clear()
        visible=[a for a in getattr(self,'active_annotations',[]) if a[1]>lo and a[0]<hi]
        labels=sorted({a[2] for a in visible})
        short={'upper limb movement':'Upper limbs','lower limb movement':'Lower limbs',
               'head movement':'Head','whole body trunk movement':'Trunk',
               'rolling lateral movement':'Rolling','caregiver induced movement':'Caregiver',
               'still no movement':'Still','minimal visible movement':'Minimal movement'}
        if not labels:
            timeline.getAxis('left').setTicks([])
            timeline.setTitle('Activity timing — not annotated')
            timeline.setFixedHeight(75)
            return
        timeline.setTitle('Activity timing')
        timeline.setFixedHeight(min(230,45+27*len(labels)))
        timeline.getAxis('left').setTicks([[(i,short.get(label,label)) for i,label in enumerate(labels)]])
        timeline.setYRange(-.6,len(labels)-.4,padding=0)
        for i,label in enumerate(labels):
            for start,end,_,color in (a for a in visible if a[2]==label):
                bar=pg.BarGraphItem(x0=[start],x1=[end],y=[i],height=.55,brush=color,pen=pg.mkPen(color))
                bar.setToolTip(f'{label}: {start:.2f}–{end:.2f} s')
                timeline.addItem(bar)
    def annotations(self):
        if not self.data:return
        for plot,region in getattr(self,'annotation_regions',[]):plot.removeItem(region)
        self.annotation_regions=[];off=0 if self.data.get('annotations_synced') else self.owner.offset.value()
        ann=[a for a in self.data.get('annotations',[]) if not self.owner.reviewed.isChecked() or a.get('review_status')=='human_reviewed']
        self.visible_annotation_rows=ann
        labels=sorted({a['label'] for a in ann});self.active_annotations=[]
        for a in ann:
            start=float(a['video_start_s'])+off;end=float(a['video_end_s'])+off
            color=pg.intColor(labels.index(a['label']),max(1,len(labels)))
            self.active_annotations.append((start,end,a['label'].replace('_',' '),color.name()))
        self.describe_range()
    def range(self,*_):
        if not self.plots:return
        span=self.end if self.span.currentText()=='Full recording' else float(self.span.currentText());start=0 if self.span.currentText()=='Full recording' else min(self.start.value(),max(0,self.end-span))
        self.plots[0].setXRange(start,min(self.end,start+span),padding=0)
    def step(self,direction):
        if self.span.currentText()=='Full recording':self.span.setCurrentText('5')
        self.start.setValue(max(0,min(self.end-float(self.span.currentText()),self.start.value()+direction*float(self.span.currentText()))))
    def full_range(self):
        self.span.setCurrentText('Full recording');self.range()
    def update_imu_display(self,*_):
        pass
    def threshold_changed(self,*_):
        if not self.data or self.motion_time is None or self.reference is None:return
        self.rotation_curve.setData(self.motion_time,self.rotation_motion/self.rotation_threshold.value())
        self.acc_curve.setData(self.motion_time,self.acc_motion/self.acc_threshold.value())
        self.describe_range()
    def scale_imu(self,lo,hi):
        if self.motion_time is None:return
        mask=(self.motion_time>=lo)&(self.motion_time<=hi)
        if self.reference is None:
            values=np.r_[self.av[mask].ravel(),self.gv[mask].ravel()]
            values=values[np.isfinite(values)]
            if len(values):self.plots[0].setYRange(float(values.min())-1,float(values.max())+1,padding=.08)
            return
        values=np.r_[self.rotation_motion[mask]/self.rotation_threshold.value(),self.acc_motion[mask]/self.acc_threshold.value()]
        values=values[np.isfinite(values)]
        if len(values):self.plots[0].setYRange(0,max(1.2,float(values.max())*1.1),padding=0)
