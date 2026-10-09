# Review fixes — 8 October 2026 (current rules)

Missing IMU: all-zero accelerometer rows, non-finite rows, and unchanged accelerometer or gyro XYZ runs longer than 0.5 s are excluded BEFORE filtering, tilt, activity or reference selection. These rules are conservative: a genuinely stationary quantised sensor can be marked unavailable, never silently confirmed as still. Source CSVs and optional raw traces are unchanged.

Movement uses the existing 0.5 s vector RMS and low-motion reference, but each threshold now also includes a whole-trial robust noise guard:

T = max(reference median + 6 × 1.4826 × reference MAD, whole-trial median + 6 × 1.4826 × whole-trial MAD, floor).

Floors: acceleration 40.96 counts (0.01 g at 4096 counts/g); gyro 20 raw counts. These are engineering settings, not infant-validated cutoffs. The whole-trial guard prevents low-tail reference selection from labelling ordinary noise as movement, but can miss sustained activity in trials with no quiet period. Do not interpret no detected movement as proof of rest.

A 5 s window needs at least 80% valid bins. No bins = unavailable. Classification requires two consecutive valid 0.5 s bins above either threshold; gaps break consecutiveness. Percent time above threshold is still reported separately with coverage. One-second activity values are computed only in valid segments with filter-edge exclusions.

Tilt is the orientation of the sensor relative to gravity. There is no Independent-versus-ParentHeld angle classifier. The classes overlap. Shared time axes are saved recording times, not evidence of belt/BIOPAC synchronization; no trace or annotation is shifted by this update.

The detailed processing steps and literature context follow.

# Label-free IMU movement processing

This is a custom exploratory signal-processing detector. It is not InfantMotion2Vec, a trained model, or a validated clinical classifier. Annotations are optional and do not influence detection. Each participant/trial is processed independently.

## Inputs and validity

Use AccX/Y/Z, GyroX/Y/Z and time_s. Keep raw sensor units: gyro physical scaling is not verified, so no physical activity cutoff is claimed. Magnetometer data are unused. Input timestamps must be finite, strictly increasing, with a median sampling rate of at least 10 Hz.

Split at nonfinite axis values or gaps greater than three median sampling intervals. Process contiguous segments at least ten seconds long. Resample within each segment to its recording-wide median interval with linear interpolation; never bridge gaps between segments.

## Filtering and interval measurements

Acceleration: second-order Butterworth high-pass at 0.3 Hz, then fourth-order low-pass at min(20 Hz, 0.4*sampling rate). Gyro: same low-pass. Apply filters forward/backward offline, axis by axis. Exclude one second at both ends of each segment to reduce filter-edge effects (not a guarantee that all transients disappear).

High-pass acceleration suppresses slow components including gravity/posture projection; it is not orientation-based gravity compensation. Very slow movement can be attenuated. Gyro preserves sustained rotation except for bias subtraction described below. All ECG displays remain unchanged/raw.

Form nonoverlapping 0.5-second bins. Require at least 90% of expected resampled samples. Calculate XYZ vector RMS for acceleration and gyro. Preliminary gyro RMS subtracts the whole-valid-recording median of each gyro axis.

## Automatic reference

Require at least 20 valid half-second bins. Rank bins by acceleration RMS and preliminary gyro RMS, assigning equal values equal ranks. Add ranks with equal weights, and choose the lowest 20% of bins, with a minimum of ten bins. Ties choose earlier bins.

Pool selected samples and estimate per-axis gyro bias using their medians. Recalculate gyro RMS for all valid bins against this bias. No repeated selection is performed. Selected bins can be noncontiguous; their start times are retained in reference_starts in the processing result.

For acceleration and gyro separately, derive threshold from selected-bin RMS values:

threshold = max(sensor floor, reference median + 6*1.4826*reference MAD, whole-trial median + 6*1.4826*whole-trial MAD)

MAD = median absolute deviation from the median. The floor avoids zero thresholds. The 20% fraction, minimum reference duration, filter choices, MAD multiplier and reporting criteria are exploratory engineering choices, not parameters validated by the shared papers. Whole-trial bias estimation can hide persistent rotation. A low-ranked subset is not proof of rest, and thresholds can be unreliable for uniformly active/noisy trials.

## Detection and five-second reporting

A valid half-second bin is movement-positive when acceleration OR gyro RMS is strictly above its threshold. Thus the minimum resolved event is 0.5 seconds; events shorter than that may influence bin RMS but do not have finer reported timing. Adjacent positive bins form event intervals; invalid bins interrupt events. No gap merging is applied; classification requires the persistence rule below.

For each five-second window report movement when at least two consecutive valid bins are positive, otherwise no sustained movement detected. Below 80% coverage report insufficient IMU coverage; with no valid bins report unavailable. Movement percentage is positive-bin duration divided by valid-bin duration; show coverage separately. RMS summaries combine valid half-second RMS values by root-mean-square. No physiological signal energy/quality analysis is added by this update.

Top plot: acceleration RMS/threshold in green and gyro RMS/threshold in purple, with a line at 1. These are processed measures in half-second bins. The raw six-channel fallback remains visible when processing/reference estimation is unavailable. Annotation intervals and five-second label summaries remain independent.

## Belt tilt and activity (display)

Implementation: imu_posture.py, adapted from Chung et al., Nature Medicine 2020, DOI 10.1038/s41591-020-0792-9, who used a chest accelerometer for body orientation and activity in NICU infants. It is independent of the movement detector above and does not change it.

Units: acceleration counts / 4096 = g (±8 g range; resting magnitude in the study data is ~4,100 counts).


Activity: third-order Butterworth band-pass 1–8 Hz on each axis, then the root mean square of the three-axis vector per second, in g. Chung et al. used 1–10 Hz on a 100 Hz accelerometer; the belt accelerometer updates about 24 times per second (saved on a 100 Hz grid by repeating values), so the band stops at 8 Hz. One second at each segment end is excluded. Samples inside runs of identical XYZ values longer than 0.5 s are dropouts and are excluded; a second needs 90% of its samples or it is unavailable. Example: P01 T01 is frozen for 85% of samples after ~550 s and so has no activity there. The movement detector and tilt also exclude these runs before filtering. The caption reports, per 5-second window, the RMS of the valid per-second values. Chung et al. reported 0.07 ± 0.02 g at rest and 0.24 ± 0.05 g during hands-on care (3 neonates); these are context, not thresholds used here.

## Relation to shared literature

- Marin-Palma et al., Scientific Reports 2025, DOI 10.1038/s41598-025-85621-y: gravity-removed acceleration, 0.1–20 Hz filtering and a 0.2 m/s² noise deadband. Our raw-data high-pass adaptation and threshold rule differ.
- Rihar et al., 2014, DOI 10.1186/1743-0003-11-133: orientation fusion plus reference/pressure information. That orientation pipeline is not implemented here.
- InfantMotion2Vec, IEEE BSN 2024, DOI 10.1109/BSN63547.2024.10780750: unlabelled representation learning followed by labelled posture classification. No model from that paper is implemented here.

The detector quantifies sensor activity, not certainty that an infant moved, nor pickup, off-body placement or physiological signal degradation. Accuracy remains unknown without independent evaluation. Older optional analysis panels retain their explicitly labelled legacy calculations.

Implementation: imu_processing.py, called from signal_stack.py. This supersedes the earlier single-five-second-reference method in imu_baseline.py (retained for historical compatibility, not used by the main viewer).