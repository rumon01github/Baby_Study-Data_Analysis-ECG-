"""Shared-time raw signal display. Preserve peaks using pyqtgraph peak downsampling."""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6 import QtWidgets
import pyqtgraph as pg
from imu_processing import process_imu, summary as movement_summary
from imu_posture import tilt as belt_tilt, activity as belt_activity, window_activity
from ppg_processing import process_ppg, display_values, DESCRIPTION as PPG_PROCESSING
from data_validity import mask_imu
from breathing import STEP as RATE_STEP

RATE_RANGE=(0,100)  # fixed breaths/min axis for rate plots

# BIOPAC PPG sensor site per participant (user-provided). P7 has no BIOPAC recording.
# Belt capacitive respiration sensors (user-confirmed).
RESP_SIDE={'Resp0':'left','Resp1':'right'}
BIOPAC_PPG_SITE={1:'leg',2:'leg',3:'leg',4:'leg',5:'leg',6:'leg',8:'leg',9:'hand',10:'hand'}


def biopac_ppg_site(trial):
    match=re.match(r'P0*(\d+)',str(trial))
    return BIOPAC_PPG_SITE.get(int(match.group(1))) if match else None


def signal_payload(belt_path,bio_path=None,b=None,bp=None):
    if b is None:b=pd.read_csv(belt_path)
    groups=[]
    for col in ['ECG','IR','Red','Resp0','Resp1','AccX','AccY','AccZ','GyroX','GyroY','GyroZ']:
        if col in b:groups.append({'name':'Baby belt '+col,'unit':'mV' if col=='ECG' else 'raw units','time':b.time_s.to_numpy(),'values':b[col].to_numpy()})
    end=float(b.time_s.iloc[-1]);start=float(b.time_s.iloc[0]);durations={'Baby belt':(start,end)}
    if bio_path:
        if bp is None:bp=pd.read_csv(bio_path)
        durations['BIOPAC']=(float(bp.time_s.iloc[0]),float(bp.time_s.iloc[-1]))
        for col in bp:
            if 'ECG MODULE' in col or 'AHA' in col:
                groups.append({'name':'BIOPAC '+col,'unit':'mV','time':bp.time_s.to_numpy(),'values':bp[col].to_numpy()})
            elif 'PPG MODULE' in col:
                groups.append({'name':'BIOPAC '+col,'unit':'V' if '[Volts]' in col else 'source units','time':bp.time_s.to_numpy(),'values':bp[col].to_numpy()})
            elif 'Respiration Rate' in col:
                groups.append({'name':'BIOPAC '+col,'unit':'breaths/min','time':bp.time_s.to_numpy(),'values':bp[col].to_numpy()})
    for c in groups:
        if c['name'] in ('Baby belt IR','Baby belt Red'):
            col=c['name'].split()[-1];v=np.asarray(c['values'],float).copy()
            bad=(v==0)
            if {'SpO2','HR'}<=set(b):bad &= (b.SpO2.to_numpy()==0)&(b.HR.to_numpy()==0)
            v[bad]=np.nan;c['values']=v
    return {'channels':groups,'durations':durations}


def prepare_signals(data):
    """Run in trial worker, cache results with the trial; no GUI calls."""
    channels=data['signals']['channels']
    from breathing import derive
    data['breathing_rates']=derive(channels)
    data['ppg_cache']={c['name']:process_ppg(c['time'],c['values']) for c in channels if c['name'] in ('Baby belt IR','Baby belt Red') or 'PPG MODULE' in c['name']}
    acc=[next((c for c in channels if c['name']=='Baby belt Acc'+axis),None) for axis in 'XYZ']
    gyro=[next((c for c in channels if c['name']=='Baby belt Gyro'+axis),None) for axis in 'XYZ']
    if all(c is not None for c in acc+gyro):
        t=acc[0]['time'];a,g=mask_imu(t,np.column_stack([c['values'] for c in acc]),np.column_stack([c['values'] for c in gyro]))
        try:data['imu_reference']=process_imu(t,a,g)
        except ValueError as e:data['imu_error']=str(e)
        data['tilt']=belt_tilt(t,a);data['belt_activity']=belt_activity(t,a)
    return data


class SignalStack(QtWidgets.QWidget):
    def __init__(self,owner):
        super().__init__();self.owner=owner;self.plots=[];self.data=None;self.updating=False
        layout=QtWidgets.QVBoxLayout(self)
        self.info=QtWidgets.QLabel('Load a trial to inspect raw ECG and movement together.');self.info.setWordWrap(True);layout.addWidget(self.info)
        controls=QtWidgets.QHBoxLayout();layout.addLayout(controls)
        controls.addWidget(QtWidgets.QLabel('Signal'))
        self.signal_kind=QtWidgets.QComboBox();self.signal_kind.addItems(['ECG','PPG','Breathing']);controls.addWidget(self.signal_kind)
        self.breathing_help=QtWidgets.QPushButton('How calculated?');controls.addWidget(self.breathing_help);self.breathing_help.hide()
        self.breathing_help.clicked.connect(self.explain_breathing)
        self.signal_kind.currentTextChanged.connect(self.refresh_waveforms)
        controls.addWidget(QtWidgets.QLabel('Start'));self.start=QtWidgets.QDoubleSpinBox();self.start.setRange(0,100000);controls.addWidget(self.start)
        controls.addWidget(QtWidgets.QLabel('Show seconds'));self.span=QtWidgets.QComboBox();self.span.addItems(['5','10','30','60','Full recording']);controls.addWidget(self.span)
        for name,delta in [('Previous',-1),('Next',1)]:
            btn=QtWidgets.QPushButton(name);btn.clicked.connect(lambda checked=False,d=delta:self.step(d));controls.addWidget(btn)
        self.start.valueChanged.connect(self.range);self.span.currentIndexChanged.connect(self.range)
        self.show_raw=QtWidgets.QCheckBox('Show raw IMU');controls.addWidget(self.show_raw);self.show_raw.toggled.connect(self.update_imu_display)
        self.show_raw.setToolTip('Add a plot of the six saved accelerometer and gyro channels in sensor counts, e.g. to check for dropouts. Shown automatically when the movement estimate is unavailable.')
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
        note=QtWidgets.QLabel('Movement strength — green: acceleration · purple: gyro · dashed line: movement threshold. Belt tilt — hover the plot for definitions.');layout.addWidget(note)
    def add(self,title,unit):
        p=pg.PlotWidget(title=title);p.setMinimumHeight(75);p.getAxis('left').setWidth(165);p.setLabel('left',unit);p.setLabel('bottom','Recording time (s)');p.showGrid(x=True,y=True,alpha=.15)
        if self.plots:p.setXLink(self.plots[0])
        self.stack.addWidget(p);self.plots.append(p);return p
    def load(self,data):
        self.data=data
        self.breathing_rates=data.get('breathing_rates');self.ppg_cache=data.setdefault('ppg_cache',{});self.resp_vb2=None
        self.reference=None
        self.baseline_status.setText('Selecting the lowest-motion reference from all valid 5-second windows in this trial.')
        for widget in (self.rotation_threshold,self.acc_threshold):
            widget.blockSignals(True);widget.setValue(widget.minimum());widget.blockSignals(False)
        if hasattr(self,'acc_view'):
            self.acc_view.scene().removeItem(self.acc_view);del self.acc_view
        while self.stack.count():
            item=self.stack.takeAt(0)
            # Hide and detach immediately: a deferred delete alone can leave the old plots painting behind the new ones.
            if item.widget():old=item.widget();old.hide();old.setParent(None);old.deleteLater()
        self.annotation_regions=[];self.plots=[];signals=data.get('signals',{});channels=signals.get('channels',[])
        durations=signals.get('durations',{})
        endpoints=[float(v[1]) for v in durations.values()]
        self.end=max(endpoints) if endpoints else max((w['Start_s']+5 for w in data['windows']),default=10)
        durations_text='; '.join(f'{name}: {ends[0]:.2f}–{ends[1]:.2f} s ({(ends[1]-ends[0])/60:.2f} min)' for name,ends in durations.items())
        self.info.setText(f'{data["trial"]} · {self.end:.0f} s · Saved time axes; belt/BIOPAC lag is not corrected');self.info.setToolTip(durations_text)
        w=data['windows']
        imu=self.add('IMU movement strength','Threshold ratio')
        self.annotation_timeline.setXLink(imu)
        self.imu_curves=[];self.motion_time=None;self.belt_activity=None;imu.addLegend()
        # Belt tilt replaces the raw counts as the default IMU view; raw counts stay available on their own plot.
        self.tilt_plot=self.add('Belt tilt from accelerometer gravity direction (0.1 Hz low-pass)','degrees');self.tilt_plot.addLegend(colCount=3)
        self.tilt_plot.setToolTip('From flat: angle between belt Z and gravity; 0° = belt lying flat facing up; '
                                  'sign-free, so comparable between trials. Roll (about X) and pitch (about Y) give the direction, as in Chung et al., '
                                  'Nat Med 2020; their signs depend on how the belt was put on, and roll is undefined when X points along gravity. '
                                  'Angles describe the belt, not the infant’s body; segments under 30 s have no angle.')
        self.raw_plot=self.add('Raw IMU — acceleration solid; gyroscope dashed; X blue / Y orange / Z purple','Raw sensor counts');self.raw_plot.addLegend()
        acc=[next((c for c in channels if c['name']=='Baby belt Acc'+axis),None) for axis in ['X','Y','Z']]
        gyro=[next((c for c in channels if c['name']=='Baby belt Gyro'+axis),None) for axis in ['X','Y','Z']]
        if all(c is not None for c in acc+gyro):
            self.motion_time=acc[0]['time'];av=np.column_stack([c['values'] for c in acc]);gv=np.column_stack([c['values'] for c in gyro])
            raw_av,raw_gv=av.copy(),gv.copy();av,gv=mask_imu(self.motion_time,av,gv)
            self.av=av;self.gv=gv;self.raw_motion_time=self.motion_time.copy()
            for prefix,values,style in [('Acc',raw_av,pg.QtCore.Qt.PenStyle.SolidLine),('Gyro',raw_gv,pg.QtCore.Qt.PenStyle.DashLine)]:
                for i,(axis,color) in enumerate(zip('XYZ',['#257d98','#d88832','#7255a0'])):
                    curve=self.raw_plot.plot(self.motion_time,values[:,i],pen=pg.mkPen(color,width=2.5,style=style),name=prefix+' '+axis)
                    curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
            try:
                angles=data['tilt'] if 'tilt' in data else belt_tilt(self.motion_time,av);self.belt_activity=data['belt_activity'] if 'belt_activity' in data else belt_activity(self.motion_time,av)
                for key,name,color,width in [('from_flat','From flat','#257d98',2.6),('roll','Roll (X)','#d88832',1.4),('pitch','Pitch (Y)','#a0522d',1.4)]:
                    curve=self.tilt_plot.plot(self.motion_time,angles[key],pen=pg.mkPen(color,width=width),name=name,connect='finite')
                    curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
            except ValueError as e:
                self.tilt_plot.setTitle('Belt tilt unavailable — '+str(e))
            self.acc_motion=np.full(len(av),np.nan);self.rotation_motion=np.full(len(gv),np.nan)
            self.rotation_curve=imu.plot([],[],pen=pg.mkPen('#7255a0',width=2.5),name='Gyro')
            self.acc_curve=imu.plot([],[],pen=pg.mkPen('#398649',width=2.5),name='Acc')
            for curve in [self.rotation_curve,self.acc_curve]:curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
            self.threshold_line=pg.InfiniteLine(1,angle=0,movable=False,pen=pg.mkPen('#aa3333',width=2,style=pg.QtCore.Qt.PenStyle.DashLine))
            imu.addItem(self.threshold_line);self.threshold_line.hide()
        else:
            imu.setTitle('IMU unavailable — one or more Acc/Gyro XYZ channels are missing')
            self.tilt_plot.setTitle('Belt tilt unavailable — one or more Acc XYZ channels are missing');self.raw_plot.setTitle('Raw IMU unavailable')
        for plot in (self.tilt_plot,self.raw_plot):plot.enableAutoRange(axis='y');plot.setAutoVisible(y=True)
        self.baseline_button.setEnabled(self.motion_time is not None)
        self.waveform_plots={};self.waveform_curves={}
        for device in ['Respiration channels','IMU breathing','Baby belt','BIOPAC']:
            self.waveform_plots[device]=self.add(device,'')
            self.waveform_plots[device].addLegend(colCount=2)  # two columns so four sources fit a short plot
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
        ppg=self.signal_kind.currentText()=='PPG'
        breathing=self.signal_kind.currentText()=='Breathing';self.breathing_help.setVisible(breathing)
        if not self.data or not getattr(self,'waveform_plots',None):return
        channels=self.data.get('signals',{}).get('channels',[])
        for device,curves in self.waveform_curves.items():
            for curve in (curves if isinstance(curves,list) else [curves]):self.waveform_plots[device].removeItem(curve)
        self.waveform_curves={};self.ppg_series=[]
        self.clear_capacitive()
        self.waveform_plots['Respiration channels'].setVisible(breathing);self.waveform_plots['IMU breathing'].setVisible(breathing)
        if breathing:
            if self.breathing_rates is None:
                from breathing import derive
                self.breathing_rates=self.data.setdefault('breathing_rates',derive(channels))
            self.plot_breathing(channels)
            return
        for device,color in [('Baby belt','#257d98'),('BIOPAC','#d88832')]:
            plot=self.waveform_plots[device]
            if ppg:
                eligible=[c for c in channels if (c['name'] in ('Baby belt IR','Baby belt Red') if device=='Baby belt' else c['name'].startswith('BIOPAC ') and 'PPG MODULE' in c['name'])]
            else:
                eligible=[c for c in channels if c['name'].startswith(device) and 'ECG' in c['name']][:1]
            site=biopac_ppg_site(self.data.get('trial')) if device=='BIOPAC' else None
            if not eligible:
                plot.setTitle(device+' '+self.signal_kind.currentText()+' — unavailable');plot.setLabel('left','');continue
            if ppg:
                title=device+' PPG — IR and Red (filtered)' if device=='Baby belt' else eligible[0]['name']+' — '+(site or 'site unknown')+' (filtered)'
                plot.setTitle(title);plot.setToolTip(', '.join(c['name'] for c in eligible)+' — '+PPG_PROCESSING)
                plot.setLabel('left','z-score')
            else:
                plot.setTitle(eligible[0]['name']);plot.setToolTip(eligible[0]['name']+' — saved CSV values, no filtering or resampling')
                plot.setLabel('left',eligible[0]['unit'])
            self.waveform_curves[device]=[]
            for channel in eligible:
                pen=pg.mkPen('#c0392b' if channel['name']=='Baby belt Red' else color,width=2)
                name=channel['name'].replace('Baby belt ','') if ppg and device=='Baby belt' else None
                if ppg:
                    if channel['name'] not in self.ppg_cache:self.ppg_cache[channel['name']]=process_ppg(channel['time'],channel['values'])
                    values=self.ppg_cache[channel['name']]
                else:
                    values=channel['values']
                curve=plot.plot(channel['time'],values,pen=pen,name=name,connect='finite')
                curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
                self.waveform_curves[device].append(curve)
                if ppg:self.ppg_series.append((curve,channel['time'],values,channel['values']))
            plot.enableAutoRange(axis='y');plot.setAutoVisible(y=True);plot.setMouseEnabled(y=True)
        if ppg and self.plots:self.scale_ppg(*self.plots[0].viewRange()[0])
    def scale_ppg(self,lo,hi):
        for curve,time,values,raw in getattr(self,'ppg_series',[]):
            curve.setData(time,display_values(time,values,raw,lo,hi))
    def plot_breathing(self,channels):
        self.plot_capacitive(channels)
        bio=[dict(c,name='From BIOPAC ECG') for c in self.breathing_rates if c['name']=='BIOPAC ECG']
        belt_names={'ECG':'From ECG','Resp0':'From left capacitive','Resp1':'From right capacitive'}
        imu=[dict(c,name='From IMU (chest tilt)') for c in self.breathing_rates if c['name']=='IMU']
        line=imu[0].get('interference') if imu else None
        imu_title='IMU-derived Resp — fused gyro + accelerometer chest tilt (solid: 30 s median, dotted: each breath)'+(f'; {60*line:.0f}/min interference removed' if line else '')
        specs=[('IMU breathing',imu_title,'breaths/min',imu),
               ('Baby belt','Belt-derived Resp — ECG, left and right capacitive (solid: 30 s median, dotted: each breath)','breaths/min',[dict(c,name=belt_names[c['name']]) for c in self.breathing_rates if c['name'] in belt_names]),
               ('BIOPAC','BIOPAC ECG-derived Resp — same method as belt (solid: 30 s median, dotted: each breath)','breaths/min',bio)]
        for device,title,unit,series in specs:
            p=self.waveform_plots[device];p.setLabel('left',unit)
            usable=any(np.isfinite(c['values']).any() for c in series)
            p.setTitle(title+('' if usable else ' — unavailable'))
            p.setToolTip('Line: each belt step covers one 5-second window and is the median rate over 30 seconds centred on it (shifted inside the recording at the start and end). Dots: breath-by-breath rate, 60 / time since the previous breath, drawn at each breath; breaths are found in the same 30 s spans. Gaps mean no estimate. Belt and BIOPAC use the same ECG method.' if device!='Respiration channels' else 'Original Resp0 (left) and Resp1 (right) capacitive respiration values, each shifted by its median over the visible time range so both fit one axis; units and calibration not specified in the source.')
            self.waveform_curves[device]=[]
            for c in series:
                color={'From IMU (chest tilt)':'#7255a0','From left capacitive':'#398649','From right capacitive':'#a0522d'}.get(c['name'],{'Baby belt':'#257d98','BIOPAC':'#d88832'}.get(device,'#7255a0'))
                x,y=c['time'],c['values']
                if device in ('IMU breathing','Baby belt','BIOPAC'):
                    # Each estimate is a flat step across its 5 s window. Breaks go in y, not x: pyqtgraph's
                    # clip-to-view needs time to be non-decreasing, and NaN times made it drop whole steps.
                    e=np.asarray(x,dtype=float);r=np.asarray(y,dtype=float)
                    x=np.column_stack([e-RATE_STEP,e,e]).ravel();y=np.column_stack([r,r,np.full(len(r),np.nan)]).ravel()
                curve=p.plot(x,y,name=c['name'],pen=pg.mkPen(color,width=2.6),connect='finite')
                if len(c.get('breath_time',[])):
                    # Breath-by-breath: 60 / interval since the previous breath, drawn at the breath and joined by a
                    # dotted line. The line breaks where breaths are over 5 s apart (longest accepted interval), so
                    # it never bridges a gap; breaks go in y so time stays non-decreasing for clip-to-view.
                    bt=np.asarray(c['breath_time'],dtype=float);br=np.asarray(c['breath_rate'],dtype=float)
                    gap=np.flatnonzero(np.diff(bt)>5)+1
                    bt=np.insert(bt,gap,bt[gap]);br=np.insert(br,gap,np.nan)
                    dots=p.plot(bt,br,pen=pg.mkPen(color,width=1.6,style=pg.QtCore.Qt.PenStyle.DotLine),connect='finite',symbol='o',symbolSize=6,symbolBrush=color,symbolPen=color)
                    dots.setClipToView(True);self.waveform_curves[device].append(dots)
                curve.setDownsampling(auto=True,method='peak');curve.setClipToView(True)
                self.waveform_curves[device].append(curve)
            p.disableAutoRange(axis='y');p.setYRange(*RATE_RANGE,padding=0);p.setMouseEnabled(y=False)  # scrolling must not zoom the rate axis
    def clear_capacitive(self):
        # Items on the second (right-axis) view box are not owned by the plot item, so remove them here; the legend
        # entries for both sensors were added by hand, so clear those too.
        pi=self.waveform_plots['Respiration channels'].plotItem if getattr(self,'waveform_plots',None) else None
        if pi is None:return
        if pi.legend is not None:
            for item in [sample.item for sample,_ in list(pi.legend.items)]:pi.legend.removeItem(item)
        vb2=getattr(self,'resp_vb2',None)
        if vb2 is not None:
            # addedItems omits items added with ignoreBounds (the faint saved trace), so use our own list.
            for item in getattr(self,'resp_vb2_items',[]):vb2.removeItem(item)
        self.resp_vb2_items=[]
    def plot_capacitive(self,channels):
        # Saved units, no shifting: Resp0 (left sensor) on the left axis, Resp1 (right sensor) on a second view box tied to
        # the right axis, so each fills the plot even when their baselines differ. Saved values faint (excluded from
        # autoscale, so spikes do not squash the view); cleaned trace bold; counted dips marked.
        p=self.waveform_plots['Respiration channels'];pi=p.plotItem
        if getattr(self,'resp_vb2',None) is None:
            vb2=pg.ViewBox();pi.scene().addItem(vb2);pi.getAxis('right').linkToView(vb2);vb2.setXLink(pi.vb)
            def sync():vb2.setGeometry(pi.vb.sceneBoundingRect());vb2.linkedViewChanged(pi.vb,vb2.XAxis)
            pi.vb.sigResized.connect(sync);sync();self.resp_vb2=vb2
        vb2=self.resp_vb2
        p.setTitle('Belt capacitive respiration — saved units (faint: saved values; bold: spikes removed; ▼: counted breaths)')
        p.setLabel('left','Resp0 left',color='#398649');p.setLabel('right','Resp1 right',color='#a0522d')
        pi.getAxis('right').setStyle(showValues=True)
        p.setToolTip('Resp0 (left sensor, left axis) and Resp1 (right sensor, right axis), in the saved units; units are not specified in the source. Spikes/dropouts more than 0.3 units from the 1 s running median are replaced, then a 2 Hz low-pass removes 0.01-step chatter. Each dip (at least 0.02 units, or half the 30 s SD, and at least 0.67 s apart) is counted as a breath.')
        self.waveform_curves['Respiration channels']=[]
        rates={c['name']:c for c in self.breathing_rates}
        for col,side in RESP_SIDE.items():
            raw=next((c for c in channels if c['name']=='Baby belt '+col),None)
            if raw is None or col not in rates:continue
            color={'Resp0':'#398649','Resp1':'#a0522d'}[col];view=pi.vb if col=='Resp0' else vb2
            t=np.asarray(raw['time'],dtype=float);clean=np.asarray(rates[col]['clean'],dtype=float);bt=np.asarray(rates[col]['breath_time'],dtype=float)
            faint=pg.PlotDataItem(t,raw['values'],pen=pg.mkPen(pg.mkColor(color).lighter(170),width=1),connect='finite')
            bold=pg.PlotDataItem(t,clean,pen=pg.mkPen(color,width=2.2),connect='finite')
            marks=pg.PlotDataItem(bt,np.interp(bt,t,clean),pen=None,symbol='t',symbolSize=9,symbolBrush=color,symbolPen=color)
            for item,ignore in ((faint,True),(bold,False),(marks,False)):
                # Left sensor goes through the plot item so the normal removal in refresh_waveforms handles it.
                if col=='Resp0':pi.addItem(item,ignoreBounds=ignore)
                else:view.addItem(item,ignoreBounds=ignore);self.resp_vb2_items.append(item)
            if pi.legend is not None:pi.legend.addItem(bold,f'{col} ({side})')
            for item in (faint,bold):item.setDownsampling(auto=True,method='peak');item.setClipToView(True)
            marks.setClipToView(True)
            self.waveform_curves['Respiration channels']+=[faint,bold,marks]
        for view in (pi.vb,vb2):view.enableAutoRange(axis='y');view.setAutoVisible(y=True);view.setMouseEnabled(y=True)
    def explain_breathing(self):
        QtWidgets.QMessageBox.information(self,'Breathing calculation',
            'BIOPAC: breathing is calculated from the BIOPAC ECG module channel with exactly the same method as the belt ECG below. The rate saved by AcqKnowledge is not shown: it never exceeds 20 breaths/min, below normal infant rates, and its settings are not in the CSVs.\n\n'
            'Belt capacitive Resp: Resp0 (left sensor, left axis) and Resp1 (right sensor, right axis), plotted in their saved units without shifting. Breathing moves them by about 0.02–0.1 units. Spikes/dropouts (more than 0.3 units from the 1 s running median) are removed, a 2 Hz low-pass removes 0.01-step chatter, and each dip is counted as one breath (at least 0.02 units or half the 30 s SD, and at least 0.67 s apart). Spans with more than 20% spikes, fewer than 4 dips or a gap over 5 s get no estimate. Units are not specified in the source. Belt ECG-derived Resp: ECG beat amplitudes provide a breathing-related signal, which is detrended and band-pass filtered before respiratory peaks are detected. Rate = 60 / median time between respiratory peaks.\n\n'
            'Belt IMU-derived Resp: the thorax tilts slightly with each breath. Gyro and accelerometer are fused within valid segments (complementary filter, 5 s). Motion can still affect the estimate. Each window is band-pass filtered (0.2–2 Hz), projected onto its main direction of change, then processed like the other sources. The gyro scale is fitted per valid segment. Periodicity and minimum-modulation checks reject unsupported estimates.\n\n'
            'Each belt estimate covers one 5-second window and uses 30 seconds centred on it, shifted inside the recording at the start and end. Missing data, weak/irregular modulation, unsupported fast breathing or too few peaks produce gaps, not zero. The search band is 12–90 breaths/min, reduced for beat-derived signals by beat sampling limits.\n\n'
            'This is an exploratory peak-interval method, similar in principle to rate detection, not a verified reproduction of BIOPAC. Motion may produce false breathing peaks. Source curves are shown separately; no combined rate is claimed.\n\n'
            'Full details: Breathing_algorithm.md in the app folder.')
    def choose_window(self,*_):
        t=self.window.currentData()
        if t is None:return
        self.span.setCurrentText('5');self.start.setValue(float(t));self.range()
    def describe_range(self,*_):
        if not self.plots:return
        lo,hi=self.plots[0].viewRange()[0]
        if not self.updating:
            self.start.blockSignals(True);self.start.setValue(max(0,lo));self.start.blockSignals(False)
        self.draw_annotation_timeline(lo,hi)
        self.scale_imu(lo,hi)
        self.scale_ppg(lo,hi)
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
        draft=any('draft' in str(a.get('review_status','')).lower() for a in self.data.get('annotations',[]))
        self.activity_legend.setText(('AI draft annotations · ' if draft else '')+'Activity: '+'<br>'.join(summaries) if summaries else 'Activity: zoom in to see labels')
        self.shading_key.setText(' · '.join(f'<span style="color:{color}">■</span> {html.escape(label)}' for label,color in sorted({(label,color) for start,end,label,color in getattr(self,'active_annotations',[]) if end>lo and start<hi})))
        self.shading_key.hide()
        summaries=[];activity=getattr(self,'belt_activity',None)
        if (self.reference is not None or activity is not None) and hi-lo<=60.01:
            for start in np.arange(max(0,np.floor(lo/5)*5),hi-.0001,5):
                text='Movement estimate unavailable'
                if self.reference is not None:
                    text=movement_summary(self.reference,start)
                if activity is not None:
                    value=window_activity(activity,start,start+5)
                    text+=f' · Activity {value:.3f} g' if np.isfinite(value) else ' · Activity unavailable (IMU dropout or edge)'
                summaries.append(f'{start:.0f}–{start+5:.0f}s · {text}')
        self.imu_summary.setText('IMU: '+('\n'.join(summaries) if summaries else ('Zoom in for movement summary' if self.reference is not None or activity is not None else 'Raw signals only — movement estimate unavailable')))
        self.imu_summary.setToolTip('Activity: RMS of the three-axis acceleration band-passed 1–8 Hz, in g, per second (after Chung et al., Nat Med 2020, who used 1–10 Hz; '
                                    'the belt accelerometer updates ~24 times/s). Seconds inside held-value dropouts are excluded rather than counted as still.')
    def select_baseline(self):
        if self.motion_time is None:return
        try:
            if 'imu_error' in self.data:raise ValueError(self.data['imu_error'])
            reference=self.data['imu_reference'] if 'imu_reference' in self.data else process_imu(self.raw_motion_time,self.av,self.gv)
        except ValueError as e:
            self.baseline_status.setText(str(e)+' Raw IMU shown instead.')
            self.plots[0].setTitle('IMU movement strength unavailable — '+str(e));self.update_imu_display();return
        self.reference=reference
        self.plots[0].setTitle('IMU movement strength');self.threshold_line.show();self.update_imu_display()
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
        span=self.end if self.span.currentText()=='Full recording' else float(self.span.currentText())
        last=max(0,5*(np.ceil(self.end/5)-1))
        start=0 if self.span.currentText()=='Full recording' else min(self.start.value(),last)
        self.updating=True
        self.plots[0].setXRange(start,min(self.end,start+span),padding=0)
        self.start.blockSignals(True);self.start.setValue(start);self.start.blockSignals(False)
        self.updating=False
    def step(self,direction):
        if not self.plots:return
        lo=self.plots[0].viewRange()[0][0]
        if self.span.currentText()=='Full recording':self.span.setCurrentText('5')
        span=float(self.span.currentText());target=5*np.floor((lo+direction*span)/5+1e-8)
        last=max(0,5*(np.ceil(self.end/5)-1))
        self.start.setValue(max(0,min(last,target)));self.range()
    def full_range(self):
        self.span.setCurrentText('Full recording');self.range()
    def update_imu_display(self,*_):
        # Raw counts on request, or as the fallback whenever the movement estimate is unavailable.
        if getattr(self,'raw_plot',None) is None:return
        self.raw_plot.setVisible(self.motion_time is not None and (self.show_raw.isChecked() or self.reference is None))
    def threshold_changed(self,*_):
        if not self.data or self.motion_time is None or self.reference is None:return
        self.rotation_curve.setData(self.motion_time,self.rotation_motion/self.rotation_threshold.value())
        self.acc_curve.setData(self.motion_time,self.acc_motion/self.acc_threshold.value())
        self.describe_range()
    def scale_imu(self,lo,hi):
        if self.motion_time is None:return
        if self.reference is None:return  # tilt and raw plots auto-range on their visible data
        mask=(self.motion_time>=lo)&(self.motion_time<=hi)
        values=np.r_[self.rotation_motion[mask]/self.rotation_threshold.value(),self.acc_motion[mask]/self.acc_threshold.value()]
        values=values[np.isfinite(values)]
        if len(values):self.plots[0].setYRange(0,max(1.2,float(values.max())*1.1),padding=0)
