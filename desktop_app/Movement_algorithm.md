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

threshold = max(1 raw unit, median + 3*1.4826*MAD)

MAD = median absolute deviation from the median. The floor avoids zero thresholds. The 20% fraction, minimum reference duration, filter choices, MAD multiplier and reporting criteria are exploratory engineering choices, not parameters validated by the shared papers. Whole-trial bias estimation can hide persistent rotation. A low-ranked subset is not proof of rest, and thresholds can be unreliable for uniformly active/noisy trials.

## Detection and five-second reporting

A valid half-second bin is movement-positive when acceleration OR gyro RMS is strictly above its threshold. Thus the minimum resolved event is 0.5 seconds; events shorter than that may influence bin RMS but do not have finer reported timing. Adjacent positive bins form event intervals; invalid bins interrupt events. No additional gap merging or duration filtering is applied.

For each five-second window report movement when at least one valid bin is positive, otherwise no movement detected. Below 80% coverage report insufficient IMU coverage; with no valid bins report unavailable. Movement percentage is positive-bin duration divided by valid-bin duration; show coverage separately. RMS summaries combine valid half-second RMS values by root-mean-square. No physiological signal energy/quality analysis is added by this update.

Top plot: acceleration RMS/threshold in green and gyro RMS/threshold in purple, with a line at 1. These are processed measures in half-second bins. The raw six-channel fallback remains visible when processing/reference estimation is unavailable. Annotation intervals and five-second label summaries remain independent.

## Relation to shared literature

- Marin-Palma et al., Scientific Reports 2025, DOI 10.1038/s41598-025-85621-y: gravity-removed acceleration, 0.1–20 Hz filtering and a 0.2 m/s² noise deadband. Our raw-data high-pass adaptation and threshold rule differ.
- Rihar et al., 2014, DOI 10.1186/1743-0003-11-133: orientation fusion plus reference/pressure information. That orientation pipeline is not implemented here.
- InfantMotion2Vec, IEEE BSN 2024, DOI 10.1109/BSN63547.2024.10780750: unlabelled representation learning followed by labelled posture classification. No model from that paper is implemented here.

The detector quantifies sensor activity, not certainty that an infant moved, nor pickup, off-body placement or physiological signal degradation. Accuracy remains unknown without independent evaluation. Older optional analysis panels retain their explicitly labelled legacy calculations.

Implementation: imu_processing.py, called from signal_stack.py. This supersedes the earlier single-five-second-reference method in imu_baseline.py (retained for historical compatibility, not used by the main viewer).
