"""Exploratory, trial-local quiet-reference IMU analysis in raw sensor units."""
import numpy as np


def quiet_reference(time, acc, gyro, start):
    time = np.asarray(time)
    if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError('IMU timestamps must be finite and increasing.')
    dt = float(np.median(np.diff(time)))
    mask = (time >= start) & (time < start + 5)
    t = time[mask]
    a, g = np.asarray(acc)[mask], np.asarray(gyro)[mask]
    if (len(t) < 10 or t[0] > start + 2 * dt or
            t[-1] + dt < start + 5 - dt or np.max(np.diff(t)) > 3 * dt or
            not np.all(np.isfinite(a)) or not np.all(np.isfinite(g))):
        raise ValueError('Choose a complete 5-second IMU window without gaps or missing values.')
    ac, gc = np.median(a, axis=0), np.median(g, axis=0)
    def threshold(v):
        median = np.median(v)
        mad = np.median(np.abs(v - median))
        # Robust exploratory rule; one raw count prevents a zero denominator.
        return float(max(1.0, median + 3 * 1.4826 * mad))
    return dict(start=float(start), acc_center=ac, gyro_center=gc,
                acc_threshold=threshold(np.linalg.norm(a - ac, axis=1)),
                gyro_threshold=threshold(np.linalg.norm(g - gc, axis=1)))


def motion_values(acc, gyro, reference):
    return (np.linalg.norm(acc - reference['acc_center'], axis=1),
            np.linalg.norm(gyro - reference['gyro_center'], axis=1))


def automatic_reference(time, acc, gyro):
    """Select lowest combined rank of acceleration variation and rotation RMS.

    Whole-trial gyro median is used only for ranking candidate windows.
    Final reference centers and thresholds use the selected window alone.
    """
    time, acc, gyro = np.asarray(time), np.asarray(acc), np.asarray(gyro)
    candidates=[];scores=[]
    gyro_center=np.nanmedian(gyro,axis=0)
    for start in np.arange(0, time[-1], 5):
        try: ref=quiet_reference(time,acc,gyro,start)
        except ValueError: continue
        mask=(time>=start)&(time<start+5)
        a=acc[mask]-np.median(acc[mask],axis=0)
        g=gyro[mask]-gyro_center
        candidates.append(ref)
        scores.append([np.sqrt(np.mean(np.sum(a*a,axis=1))),np.sqrt(np.mean(np.sum(g*g,axis=1)))])
    if not candidates:raise ValueError('No complete valid 5-second IMU reference window.')
    scores=np.asarray(scores)
    # Equal values receive equal ranks. Equal combined ranks favour earlier time.
    ranks=np.column_stack([np.searchsorted(np.sort(scores[:,i]),scores[:,i],side='left') for i in range(2)])
    return candidates[int(np.argmin(ranks.sum(axis=1)))]


def window_summary(time, acc_motion, gyro_motion, start, acc_threshold, gyro_threshold):
    mask = (time >= start) & (time < start + 5)
    valid = mask & np.isfinite(acc_motion) & np.isfinite(gyro_motion)
    if not np.any(valid):
        return 'IMU unavailable'
    a, g = acc_motion[valid], gyro_motion[valid]
    return (f'Acc above reference: {100*np.mean(a > acc_threshold):.0f}% · '
            f'Rotation: {100*np.mean(g > gyro_threshold):.0f}% · '
            f'Either: {100*np.mean((a > acc_threshold) | (g > gyro_threshold)):.0f}% of valid samples')
