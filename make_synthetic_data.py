"""Generate a small synthetic data set in the expected folder layout so the pipeline can be exercised without real recordings.
Two 'infants', two trials each: belt (100 Hz, first-difference ECG + IMU) and BIOPAC (1000 Hz) built from NeuroKit2's ECG simulator,
with injected motion bursts and one trial with poor belt contact. Numbers obtained on this data are meaningless; it exists to check the code path."""
import os, numpy as np, pandas as pd, neurokit2 as nk
rng = np.random.default_rng(3); ROOT = 'synthetic_data/Organized Study Data/'
for p, hr, contact in [('P11', 140, 1.0), ('P12', 125, 0.15)]:
    for t in ['T01', 'T02']:
        T = 120; fs_b = 1000
        ecg = nk.ecg_simulate(duration=T, sampling_rate=fs_b, heart_rate=hr, noise=0.02, random_state=int(rng.integers(1e6)))
        tb = np.arange(len(ecg)) / fs_b
        # motion bursts: baseline wander + gyro activity
        gyro = np.full(len(tb), 60.0); bursts = rng.uniform(5, T - 15, 4)
        for s in bursts:
            m = (tb >= s) & (tb < s + 8); ecg[m] += 0.8 * np.sin(2 * np.pi * 1.5 * tb[m]) * rng.uniform(0.3, 1); gyro[m] += rng.uniform(400, 900)
        bio = pd.DataFrame({'sample_index': np.arange(len(tb)), 'time_s': tb, 'ECG MODULE - ECG A, X, ECG2-R [mV]': ecg + 0.02 * rng.standard_normal(len(tb))})
        # belt: first-difference ECG at 100 Hz, contact factor, IMU at ~24 Hz sample-and-hold
        t100 = np.arange(0, T, 0.01); e100 = np.interp(t100, tb, ecg); belt_ecg = contact * np.r_[0, np.diff(e100)] * 40 + 0.03 * rng.standard_normal(len(t100))
        g = np.interp(t100, tb, gyro); idx = (np.arange(len(t100)) // 4) * 4; g = g[idx]
        acc = np.c_[np.full(len(t100), -64.0), np.full(len(t100), 336.0), np.full(len(t100), 4048.0)] + rng.standard_normal((len(t100), 3)) * 20 * (g[:, None] > 300)
        belt = pd.DataFrame({'sample_index': np.arange(len(t100)), 'time_s': t100, 'ECG': belt_ecg, 'AccX': acc[:, 0], 'AccY': acc[:, 1], 'AccZ': acc[:, 2],
                             'GyroX': g / np.sqrt(3), 'GyroY': g / np.sqrt(3), 'GyroZ': g / np.sqrt(3)})
        d = f'{ROOT}{p}/{t}/'; os.makedirs(d, exist_ok=True)
        belt.to_csv(f'{d}{p}_synthetic_{t}_BELT_M_Independent.csv', index=False); bio.to_csv(f'{d}{p}_synthetic_{t}_BIOPAC_M_Independent.csv', index=False)
        print('wrote', d)
