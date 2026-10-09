"""Exploratory breathing-rate estimates; not a reproduction of BIOPAC presets."""
import numpy as np
from scipy import signal, ndimage
from data_validity import mask_imu

WINDOW = 30.0
STEP = 5.0

MAX_MOVING = 0.2  # IMU estimate: skip windows with more than 20% of 0.5 s bins above the movement thresholds

# Capacitive (Resp0/Resp1) breath counting. Breathing moves these channels by roughly 0.02-0.1 raw units; separate
# sharp spikes/dropouts of 1-6 units (down to a ~16.4 rail, or 0) are not breaths and are removed first.
CAP_SPIKE = 0.3       # raw units away from the 1 s running median = spike/dropout
CAP_MAX_SPIKE = 0.2   # skip a 30 s span if more than 20% of it was spikes
CAP_MIN_DIP = 0.02    # smallest counted dip (two 0.01 sensor steps); otherwise half the span's SD
CAP_MIN_GAP = 0.67    # s between breaths (at most 90 breaths/min)


def _windows(time):
    """5 s display windows (returned by their end time), the 30 s analysis span for each, and the median
    sample interval. The span is centred on the 5 s window and shifted inside the recording at its start
    and end, so every 5 s window of a recording of at least 30 s has a span. Empty if timestamps are unusable."""
    t = np.asarray(time, dtype=float)
    if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0) or t[-1]-t[0] < WINDOW:
        return np.array([]), np.empty((0, 2)), None
    ends = np.arange(np.floor(t[0] / STEP) * STEP + STEP, np.ceil(t[-1] / STEP) * STEP + 1e-8, STEP)
    lo = np.clip(ends - STEP/2 - WINDOW/2, t[0], t[-1] - WINDOW)
    return ends, np.column_stack([lo, lo + WINDOW]), np.median(np.diff(t))


def _usable(tw, y, dt):
    if len(y) < 30 or not np.all(np.isfinite(y)) or np.ptp(y) == 0:
        return False
    return not (tw[-1]-tw[0] < WINDOW-2*dt or np.max(np.diff(tw)) > 3*dt)


def _run(t, window_breaths, with_breaths):
    """Loop over 5 s windows. window_breaths(mask, lo, hi) returns breath (respiratory peak) times found in that
    window's 30 s span, or None. Rate = 60 / median breath interval in the span. Breath-by-breath values are
    60 / each interval, kept only for intervals ending inside the 5 s window, so no breath is reported twice."""
    ends, spans, dt = _windows(t)
    rates = np.full(len(ends), np.nan)
    bt, br = [], []
    for i, (lo, hi) in enumerate(spans):
        p = window_breaths((t >= lo-1e-9) & (t <= hi+1e-9), lo, hi, dt)
        if p is None:
            continue
        intervals = np.diff(p)
        rates[i] = 60/np.median(intervals)
        inside = (p[1:] > ends[i]-STEP) & (p[1:] <= ends[i])
        bt.extend(p[1:][inside])
        br.extend(60/intervals[inside])
    if with_breaths:
        return ends, rates, np.array(bt), np.array(br)
    return ends, rates


def estimate(time, values, source, with_breaths=False):
    """One estimate per 5 s window from a 30 s span centred on it. Never interpolate across missing samples/gaps.
    with_breaths=True also returns breath times and breath-by-breath rates."""
    t = np.asarray(time, dtype=float)
    x = np.asarray(values, dtype=float)
    def window_breaths(mask, lo, hi, dt):
        tw, y = t[mask], x[mask]
        return _window_breaths(tw, y, source, dt) if _usable(tw, y, dt) else None
    return _run(t, window_breaths, with_breaths)


IMU_TAU = 5.0  # s; complementary filter: gyro leads below this time scale, accelerometer corrects slower drift

# Fixed-frequency IMU interference: in P01 T01/T02 and P06 T01-T03 the gyro carries a narrow line at 0.56-0.64 Hz
# (34-38 /min) that keeps exactly the same frequency for most of the recording while ECG and capacitive rates vary;
# breathing does not do that. Such a line (and harmonics) is notched out before fusion.
LINE_SHARP = 20      # line peak at least 20x the median 0.2-2 Hz power of its 30 s window
LINE_SHARE = 0.5     # ...within 0.02 Hz of the same frequency in at least half of the 30 s windows


def stationary_line(time, x):
    """Frequency (Hz) of a fixed-frequency interference line in x (samples x axes), or None."""
    t = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    fs = 1/np.median(np.diff(t))
    n = int(WINDOW*fs)
    peaks = []
    for s in range(0, len(x)-n+1, n//2):
        fr, P = signal.welch(x[s:s+n], fs=fs, nperseg=n, axis=0)
        P = P.reshape(len(fr), -1).sum(1)
        band = (fr >= .2) & (fr <= 2)
        k = np.argmax(P[band])
        peaks.append((fr[band][k], P[band][k]/np.median(P[band])))
    if not peaks:
        return None
    pk = np.array(peaks)
    sharp = pk[pk[:, 1] >= LINE_SHARP, 0]
    if len(sharp) == 0:
        return None
    values, counts = np.unique(np.round(sharp/.02)*.02, return_counts=True)
    f0 = values[np.argmax(counts)]
    share = np.mean((np.abs(pk[:, 0]-f0) <= .02) & (pk[:, 1] >= LINE_SHARP))
    return float(f0) if share >= LINE_SHARE else None


def remove_line(time, x, f0):
    """Zero-phase notch at f0 and its harmonics below 0.45 fs."""
    fs = 1/np.median(np.diff(np.asarray(time, dtype=float)))
    x = np.asarray(x, dtype=float)
    for h in range(1, 4):
        if h*f0 < .45*fs:
            b, a = signal.iirnotch(h*f0, Q=8, fs=fs)
            x = signal.filtfilt(b, a, x, axis=0)
    return x


def gyro_scale(time, acc, gyro):
    """Gyro units are raw counts with an unconfirmed range, so fit the scale (rad/s per count) per recording: in
    calm samples (|acc| within 5% of its median) the gravity direction from the accelerometer must turn as
    d(g)/dt = -omega x g. Least squares on 2 Hz low-passed signals. gyro must already have its bias removed."""
    t = np.asarray(time, dtype=float)
    a = np.asarray(acc, dtype=float)
    w = np.asarray(gyro, dtype=float)
    fs = 1/np.median(np.diff(t))
    lp = signal.butter(2, min(2.0, .4*fs), fs=fs, output='sos')
    ga = signal.sosfiltfilt(lp, a, axis=0)
    ga /= np.maximum(np.linalg.norm(ga, axis=1, keepdims=True), 1e-9)
    x = -np.cross(signal.sosfiltfilt(lp, w, axis=0), ga)
    dg = np.gradient(ga, axis=0)*fs
    norm = np.linalg.norm(a, axis=1)
    calm = np.abs(norm-np.median(norm)) < .05*np.median(norm)
    x, dg = x[calm], dg[calm]
    keep = np.ones(len(x), bool)
    for _ in range(3):
        # Refit without the 20% worst-fitting samples: brief jolts below the 5% |acc| limit still bend the fit.
        den = np.sum(x[keep]*x[keep])
        if den <= 0:
            return np.nan
        k = np.sum(dg[keep]*x[keep])/den
        resid = np.linalg.norm(dg-k*x, axis=1)
        keep = resid <= np.percentile(resid, 80)
    return k


def fused_gravity(time, acc, gyro, scale, tau=IMU_TAU):
    """Gravity direction in sensor axes from a complementary filter: rotate the previous estimate by the gyro,
    then pull it towards the accelerometer direction with time constant tau. Unlike the raw accelerometer it is
    not thrown by short linear accelerations (jolts, handling)."""
    t = np.asarray(time, dtype=float)
    a = np.asarray(acc, dtype=float)
    dt = np.median(np.diff(t))
    alpha = dt/tau
    norm = np.linalg.norm(a, axis=1)
    ok = np.isfinite(norm) & (norm > 0)
    # Dropout rows (all-zero or missing accelerometer) give no gravity direction: coast on the gyro there,
    # otherwise one bad row would turn the estimate into NaN for the rest of the recording.
    ga = np.where(ok[:, None], a/np.where(ok, norm, 1)[:, None], 0.0)
    turn = np.nan_to_num(np.asarray(gyro, dtype=float))*scale*dt
    g = np.empty_like(ga)
    g[0] = ga[np.argmax(ok)] if ok.any() else (0, 0, 1)
    for i in range(1, len(g)):
        p = g[i-1]-np.cross(turn[i], g[i-1])
        if ok[i]:
            p = (1-alpha)*p + alpha*ga[i]
        g[i] = p/np.linalg.norm(p)
    return g


def imu_tilt(time, acc, gyro=None):
    """Gravity direction for the IMU estimate: fused with the gyro when available, else the raw accelerometer."""
    t = np.asarray(time, dtype=float)
    a = np.asarray(acc, dtype=float)
    if gyro is not None:
        scale = gyro_scale(t, a, gyro)
        if np.isfinite(scale) and scale > 0:
            return fused_gravity(t, a, gyro, scale)
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    return a/np.where(norm > 0, norm, 1)


def estimate_imu(time, acc, moving=None, with_breaths=False, gyro=None):
    """Thoracic tilt. With gyro (bias removed), the tilt signal is the fused gravity direction (fused_gravity),
    otherwise the raw accelerometer. Per window: band-pass the three axes and project onto their main direction
    (first principal component), then use the same steps as Resp0/Resp1. moving(start, end) -> fraction of
    movement-flagged time; windows above MAX_MOVING are skipped (None = no movement gate)."""
    t = np.asarray(time, dtype=float)
    a = imu_tilt(t, acc, gyro) if gyro is not None else np.asarray(acc, dtype=float)
    def window_breaths(mask, lo, hi, dt):
        tw, aw = t[mask], a[mask]
        if not _usable(tw, aw, dt) or 1/dt < 10:
            return None
        if moving is not None and not moving(lo, hi) <= MAX_MOVING:
            return None
        aw = signal.sosfiltfilt(signal.butter(2, (.2, min(2, .4/dt)), btype='bandpass', fs=1/dt, output='sos'), aw-aw.mean(0), axis=0)
        if np.sqrt(np.mean(np.sum(aw**2,axis=1))) < (.001 if np.nanmedian(np.linalg.norm(a,axis=1))<2 else 4.096):return None
        direction = np.linalg.svd(aw, full_matrices=False)[2][0]
        return _window_breaths(tw, aw @ direction, 'IMU', dt)
    return _run(t, window_breaths, with_breaths)


def clean_capacitive(time, values):
    """Spike/dropout removal and smoothing for a capacitive respiration channel. Returns (cleaned, spike_mask)."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(t) < 3:
        return v.copy(), np.zeros(len(v), bool)
    fs = 1/np.median(np.diff(t))
    missing=~np.isfinite(v)
    v = np.where(~missing, v, np.nanmedian(v))
    med = ndimage.median_filter(v, size=int(fs)+1, mode='nearest')
    spike=np.abs(v-med)>CAP_SPIKE
    jumps=np.flatnonzero(np.abs(np.diff(v))>CAP_SPIKE)+1
    until=0
    for start in jumps:
        if start<until:continue
        baseline=np.median(v[max(0,start-int(fs)):start])
        back=np.flatnonzero(np.abs(v[start:]-baseline)<=CAP_SPIKE)
        end=start+int(back[0]) if len(back) else len(v)
        spike[start:end]=True;until=end
    spike = ndimage.binary_dilation(spike|missing, iterations=max(1, int(.1*fs)))
    valid=np.flatnonzero(~spike)
    if len(valid)<2:return np.full(len(v),np.nan),np.ones(len(v),bool)
    y=v.copy();y[spike]=np.interp(np.flatnonzero(spike),valid,v[valid])
    # 2 Hz low-pass removes the 0.01-step quantisation chatter, keeps breaths (<=1.5 Hz).
    y = signal.sosfiltfilt(signal.butter(2, min(2.0, .4*fs), fs=fs, output='sos'), y)
    edges=np.flatnonzero(np.r_[True,spike[1:]!=spike[:-1],True])
    for i,j in zip(edges[:-1],edges[1:]):
        if spike[i] and (j-i)/fs>.5:y[i:j]=np.nan
    return y, spike


def estimate_capacitive(time, values, with_breaths=False):
    """Count breaths as dips in a capacitive respiration channel (after clean_capacitive), per 30 s span."""
    t = np.asarray(time, dtype=float)
    y, spike = clean_capacitive(t, values)
    def window_breaths(mask, lo, hi, dt):
        tw, w = t[mask], y[mask]
        if not _usable(tw, w, dt) or spike[mask].mean() > CAP_MAX_SPIKE:
            return None
        fs = 1/dt
        if not periodic(w,fs):return None
        w = signal.sosfiltfilt(signal.butter(2, .15, btype='highpass', fs=fs, output='sos'), w-np.median(w))
        dips, _ = signal.find_peaks(-w, prominence=max(CAP_MIN_DIP, .5*np.std(w)), distance=max(1, int(CAP_MIN_GAP*fs)))
        bt = tw[dips]
        if len(bt) < 4 or np.any(np.diff(bt) > 5):
            return None
        return bt
    return _run(t, window_breaths, with_breaths)


def periodic(y, fs, high=1.5):
    """Engineering acceptance gate, not a clinical quality score."""
    y=signal.detrend(np.asarray(y,float))
    if len(y)<100 or not np.isfinite(y).all() or np.std(y)<1e-8:return False
    f,p=signal.periodogram(y,fs=fs)
    band=(f>=.2)&(f<=2.0)
    if not band.any() or p[band].sum()<=0:return False
    k=np.flatnonzero(band)[np.argmax(p[band])];freq=f[k]
    if freq>=high*.95:return False  # reject fast modulation, do not count every other breath
    share=p[np.abs(f-freq)<=max(.07,fs/len(y))].sum()/p[band].sum()
    lag=int(round(fs/freq))
    if lag>=len(y)//2:return False
    corr=np.corrcoef(y[:-lag],y[lag:])[0,1]
    return bool(share>=.55 and corr>=.65)

def _window_breaths(tw, y, source, dt):
    """Breath (respiratory peak) times for one validated span, or None if any check fails."""
    fs = 1/dt
    tt = np.arange(tw[0], tw[-1]+dt*.1, dt)
    y = np.interp(tt, tw, y)
    high = 1.5
    if source in ('ECG', 'IR', 'Red'):
        if fs < 25:
            return None
        band = (5, min(25, fs*.4)) if source == 'ECG' else (.5, min(8, fs*.4))
        filtered = signal.sosfiltfilt(signal.butter(2, band, btype='bandpass', fs=fs, output='sos'), y)
        # ECG polarity is selected by stronger excursions; optical pulses likewise.
        if abs(np.percentile(filtered, 1)) > abs(np.percentile(filtered, 99)):
            filtered = -filtered
        peaks, _ = signal.find_peaks(filtered, distance=max(1, int(.25*fs)), prominence=.4*np.std(filtered))
        if len(peaks) < 12:
            return None
        bt = tt[peaks]
        if np.max(np.diff(bt)) > 2 or bt[-1]-bt[0] < 25:
            return None
        # Breath-related beat/pulse amplitude modulation, not pulse frequency.
        amplitudes = filtered[peaks]
        if np.std(amplitudes)<.05*np.median(np.abs(amplitudes)):return None
        high = min(1.5, .4/np.median(np.diff(bt)))
        rt = np.arange(bt[0], bt[-1], .1)
        envelope = np.interp(rt, bt, amplitudes)
    else:
        if fs < 10:
            return None
        y = signal.sosfiltfilt(signal.butter(4, min(4, fs*.4), fs=fs, output='sos'), y)
        rt = np.arange(tt[0], tt[-1], .1)
        envelope = np.interp(rt, tt, y)
    if high <= .2 or len(envelope) < 200 or np.std(envelope) < 1e-10:
        return None
    envelope = signal.detrend(envelope)
    if not periodic(envelope,10,high):return None
    resp = signal.sosfiltfilt(signal.butter(2, (.2, high), btype='bandpass', fs=10, output='sos'), envelope)
    # Discard filter-edge regions; rate uses intervals between respiratory peaks.
    resp = resp[20:-20]
    if np.std(resp) < 1e-10:
        return None
    peaks, _ = signal.find_peaks(resp, prominence=.5*np.std(resp))
    intervals = np.diff(peaks)/10
    if len(intervals) < 3 or np.any(intervals < 1/high) or np.any(intervals > 5):
        return None
    return rt[20:-20][peaks]


def derive(channels):
    result = []
    for name in ('ECG',):
        c = next((c for c in channels if c['name'] == 'Baby belt '+name), None)
        if c is not None:
            t, rate, bt, br = estimate(c['time'], c['values'], name, with_breaths=True)
            result.append(dict(name=name, time=t, values=rate, breath_time=bt, breath_rate=br))
    for col in ('Resp0', 'Resp1'):
        c = next((c for c in channels if c['name'] == 'Baby belt '+col), None)
        if c is not None:
            t, rate, bt, br = estimate_capacitive(c['time'], c['values'], with_breaths=True)
            result.append(dict(name=col, time=t, values=rate, breath_time=bt, breath_rate=br,
                               clean=clean_capacitive(c['time'], c['values'])[0]))
    # BIOPAC: same ECG method on its ECG module channel (the saved AcqKnowledge rate tops out at 20 breaths/min).
    c = next((c for c in channels if c['name'].startswith('BIOPAC ') and 'ECG MODULE' in c['name']), None)
    if c is not None:
        t, rate, bt, br = estimate(c['time'], c['values'], 'ECG', with_breaths=True)
        result.append(dict(name='BIOPAC ECG', time=t, values=rate, breath_time=bt, breath_rate=br))
    acc = [next((c for c in channels if c['name']=='Baby belt Acc'+axis),None) for axis in 'XYZ']
    gyro = [next((c for c in channels if c['name']=='Baby belt Gyro'+axis),None) for axis in 'XYZ']
    if all(c is not None for c in acc+gyro):
        t=np.asarray(acc[0]['time']);a=np.column_stack([c['values'] for c in acc]);g=np.column_stack([c['values'] for c in gyro])
        a,g=mask_imu(t,a,g);valid=np.isfinite(a).all(1)&np.isfinite(g).all(1)
        dt=np.median(np.diff(t));edges=np.flatnonzero(np.r_[True,~valid[:-1]|~valid[1:]|(np.diff(t)>3*dt),True])
        parts=[]
        for i,j in zip(edges[:-1],edges[1:]):
            if not valid[i:j].all() or t[j-1]-t[i]<30:continue
            av=a[i:j];gv=g[i:j]-np.median(g[i:j],axis=0)
            # A stable spectral line alone cannot establish interference; do not notch real breathing automatically.
            tilt=imu_tilt(t[i:j],av,gv)
            parts.append(estimate_imu(t[i:j],tilt,with_breaths=True))
        arrays=[np.concatenate([r[k] for r in parts]) if parts else np.array([]) for k in range(4)]
        result.append(dict(name='IMU',time=arrays[0],values=arrays[1],breath_time=arrays[2],breath_rate=arrays[3],interference=None))
    return result
