# Review fixes — 8 October 2026

Only valid segments enter IMU fusion: zero/missing/stale input is excluded before filtering. No automatic interference notch is applied.

Before accepting a respiratory peak sequence, detrended pre-bandpass data must have at least 55% of 0.2–2 Hz power within ±max(0.07 Hz, one FFT bin) of its strongest frequency, plus lag-one-cycle correlation ≥0.65. A peak at or above 95% of the supported upper rate is rejected before filtering to avoid reporting every other fast breath. Respiratory peak detection no longer imposes a minimum distance that conceals fast intervals; unsupported intervals reject the window.

ECG beat-amplitude modulation SD must be at least 5% of median absolute beat amplitude. IMU tilt RMS must exceed 0.001 in normalised gravity units (4.096 counts when using acceleration). These conservative engineering gates reject many noise-only inputs, but are not a validated breathing detector. Some true irregular or weak breathing will be unavailable.

Capacitive steps over 0.3 source units are tracked until return near the preceding baseline. Long flagged episodes (>0.5 s) remain missing in the cleaned output, and affected analysis spans are unavailable. Short spikes are interpolated for smoothing. No arbitrary plateau is counted as a breath. Missing intervals are not restored as valid observations.

Rates use 30 s centred analysis spans every 5 s. Computation runs in the trial worker and is cached per loaded trial. The UI keeps the existing time axes. Annotation timestamps, offsets and files are unchanged.

# Current display

Capacitive plot: green = Resp0 (left sensor), brown = Resp1 (right sensor). Faint = saved values; bold = spikes removed and smoothed; ▼ = dips counted as breaths. Values are in the saved units, not shifted: Resp0 uses the left y-axis and Resp1 the right y-axis, each scaled to its own visible range. IMU rate plot (second): purple = IMU (chest tilt) breathing rate in breaths/min. Belt rate plot (third): blue = belt ECG, green/brown = left/right capacitive. BIOPAC rate plot (last): orange = BIOPAC ECG, calculated by the app with the same method as the belt ECG. In rate plots the solid line is the 30 s median per 5 s window and the dotted line joins breath-by-breath values. No PPG-derived rate is calculated. Resp0/Resp1 units and calibration are not specified in the source files.

# Breathing view — ECG-PPG_Neotex

The view contains activity timing, IMU, original belt Resp0/Resp1, belt estimated breathing rates, and BIOPAC ECG-derived rates calculated by the app. All share recording time. No rate is computed from just the displayed 5 seconds.

## BIOPAC

The view calculates BIOPAC breathing itself from the `ECG MODULE - ECG A, X, ECG2-R [mV]` channel (1000 Hz), using exactly the belt ECG method below (same windows, filters, checks, median and breath-by-breath rates). The algorithm is shared; comparison still requires usable signals and verified temporal alignment.

The saved AcqKnowledge channels `Respiration Rate (ECG) [BPM]` and `Respiration Rate (PPG) [BPM]` are no longer plotted. Across all 42 BIOPAC files the ECG-based saved rate never exceeds 20.0 breaths/min (median about 19; 35% of non-zero values at 19.5–20), below normal infant rates, which points to a maximum-rate limit in the recording preset. The CSV does not identify that preset, its filters, averaging or delay.

BIOPAC describes rate detection from periodic signals, and confirms ECG-derived respiration is possible:
- https://www.biopac.com/application-note/acqknowledge-rate-detector-algorithm/
- https://www.biopac.com/wp-content/uploads/Respiration-QA.pdf

These sources do not establish which preset produced the saved channels. The app's own calculation is an exploratory peak-interval algorithm, not a reproduction of a BIOPAC preset.

## Belt algorithm

Active sources: belt ECG, BIOPAC ECG, Resp0, Resp1 and IMU. The optional IR/Red helper is not called by derive(). Sources are separate: ECG, Resp0, Resp1 (capacitive sources use their own dip-counting steps, below). Resp0 and Resp1 are the left and right capacitive respiration sensors (user-confirmed); units and calibration are not specified, so their plots retain raw units. The ECG display remains saved values; the PPG view has its separately documented filtering.

1. Make one estimate per 5-second display window (0–5 s, 5–10 s, …). Each uses 30 seconds of data centred on that window (e.g. 50–55 s uses 37.5–67.5 s); at the start and end of the recording the 30 s span is shifted inside it (0–5 s uses 0–30 s). Recordings shorter than 30 s get no estimate. In the view, each estimate is drawn as a flat step across its 5 s window. Belt and BIOPAC rate plots use a fixed 0–100 breaths/min axis.
2. Require finite data, increasing times, coverage within two sample intervals of 30 s, and no gap over three median sample intervals. Resample linearly onto the median sampling interval within that window only.
3. ECG: second-order Butterworth bandpass 5–min(25,0.4 fs) Hz. The unused PPG helper uses 0.5–min(8,0.4 fs) Hz. Apply forward/backward zero-phase filtering. Orient pulses by the stronger 1st/99th percentile excursion. Detect peaks with minimum separation 0.25 s and prominence 0.4 times filtered standard deviation. Require at least 12 peaks, at least 25 s coverage and no inter-peak gap over 2 s. Use their filtered amplitudes as the respiration surrogate. This is amplitude modulation, not counting heartbeats as breaths.
4. Interpolate those amplitudes at 10 Hz. Resp0/Resp1 use the separate capacitive pipeline below, not this helper. Detrend linearly. Apply second-order zero-phase Butterworth bandpass 0.2–1.5 Hz (12–90 breaths/min). For beat-derived surrogates, reduce the upper limit to min(1.5, 0.4 / median beat interval) Hz to limit aliasing. Fast breathing outside that range will not be represented reliably.
5. Discard 2 s at each filtered surrogate edge. Detect respiratory peaks with prominence 0.5 times surrogate standard deviation without a minimum-distance constraint; reject intervals outside the supported band rather than suppressing fast peaks. Require at least three intervals, all between 1/upper-band-limit and 5 s.
6. Rate = 60 / median(diff(respiratory peak times)), in breaths/minute. Breath-by-breath rate = 60 / each individual interval, plotted at the later breath; only intervals ending inside the 5 s window are kept, so each breath is reported once. In the view, breath-by-breath values are dots joined by a dotted line (broken where breaths are over 5 s apart) and the solid flat line is the 30 s median. Single intervals are noisier than the median: one missed or extra peak shows up as an outlying dot. Flat, missing, short or failed windows produce NaN gaps, never a substituted zero.

## Belt capacitive breathing (Resp0 left, Resp1 right)

Breathing moves these channels by roughly 0.02–0.1 raw units (sensor resolution 0.01). Separately they contain sharp spikes/dropouts of 1–6 units, often to a fixed ~16.4 rail or to 0; these are not breaths and dominated earlier attempts.

1. Spike removal on the whole channel: samples more than 0.3 units from the 1 s running median are flagged (widened by 0.1 s each side) and replaced by that running median.
2. Second-order zero-phase 2 Hz low-pass to remove the 0.01-step quantisation chatter.
3. Same 5 s windows and centred 30 s spans as above. Skip a span if more than 20% of it was spikes.
4. Within the span: subtract the median, 0.15 Hz zero-phase high-pass for slow drift, then count dips (local minima) with prominence at least max(0.02 units, 0.5 × span SD) and at least 0.67 s apart (≤90 breaths/min).
5. Require at least 4 dips and no gap over 5 s. Rate = 60 / median dip interval; breath-by-breath = 60 / each interval, as for the other sources.

Historical check of the pre-fix algorithm (not evidence for the revised method) against the BIOPAC ECG-derived rate (T01 unless noted): P08 T06 right sensor gave a rate in 90% of windows, median 30 vs 27, within 5/min in 72%. Elsewhere results are mixed: within 5/min in 30–55% of windows for P01, P04, P06 and P09 (left); the right sensor reads high on P08 T01 and P10 T01 (median about 47–48 vs 29–33), likely counting more than one dip per breath; P09 right and P10 left are mostly spikes and rarely give a rate. The thresholds are engineering choices, not validated infant settings.

## Belt IMU-derived breathing (chest tilt, fused orientation)

Shown in breaths/min in its own plot (second), below the capacitive signals.

The belt DAQ, including its IMU, sits on the thorax. Each breath tilts the chest slightly, which turns the gravity direction seen by the sensor.

1. Gyro bias: per-axis median over the low-motion reference bins from Movement_algorithm.md.
2. Automatic notch removal is disabled. A persistent spectral peak alone cannot establish interference. The detector remains available for inspection/testing, but derive() does not notch a possible breathing component.
3. Gyro scale is fitted per valid recording segment from accelerometer/gyro consistency. It is an estimated effective scale, not verified BNO085 calibration. Previous pre-notch range figures are withdrawn; no numerical range is claimed for the current method.
4. Complementary filter (time constant 5 s): rotate the previous gravity estimate by the gyro, then pull it towards the accelerometer direction (skipped on all-zero/missing accelerometer rows, where the filter coasts on the gyro). Fusion can still be affected by linear acceleration; it does not establish breathing validity.
5. Same 5 s windows and centred 30 s spans as above. Per span: remove the mean, second-order zero-phase 0.2–min(2, 0.4 fs) Hz bandpass, project the three axes onto their first principal component, then continue as for the other sources (10 Hz resampling, 0.2–1.5 Hz bandpass, respiratory peaks, 60 / median interval, breath-by-breath values).

Validation status: exploratory. Previous claims that every source pair agreed no better than a 60 s shift were overstated. The prior review reported left/right capacitive Spearman correlation about +0.49 versus +0.29 after a 60 s shift. These are historical audit results, not validation of the revised estimator. No accuracy claim follows from agreement between surrogate estimates.

Background: Beck et al. 2020 (Curr Dir Biomed Eng 6(3):20203060) measured adult respiratory rate from the relative orientation of two IMUs (ventral and dorsal) with a 0.1–1.5 Hz Butterworth bandpass and FFT per ~30 s segment; the second IMU cancels whole-body movement, which the single belt IMU cannot do. Rahman & Morshed 2021 (IEEE EIT, doi:10.1109/EIT51626.2021.9491900) used one thorax-abdomen accelerometer at 10 Hz with moving-average smoothing and peak counting against a pressure belt. Both studied still adults with deliberate or paced breathing; their accuracy does not transfer to moving infants.

All numeric settings above are initial engineering choices, not validated infant thresholds. Zero-phase filtering is offline. No annotation or IMU is used to force a rate. Motion can generate periodic artefacts; passing these checks does not establish physiological validity. ECG/PPG amplitude-derived respiration may miss fast respiration or confuse motion with breathing. No averaging of different sources into a single consensus rate is performed.

Use these plots for exploratory comparison. Both ECG rates use the same implementation, but input sampling, artifacts and clock alignment can affect their difference. In a first check on T01 for P01, P04 and P09 they were within 5 breaths/min in 31–41% of windows. Agreement statistics require checking source validity and temporal support first. No signal-quality grading or clinical classification is added.