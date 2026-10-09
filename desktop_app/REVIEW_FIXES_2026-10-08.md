# App review fixes — 8 October 2026

Annotation timestamps, offsets and source annotation files were intentionally left unchanged, as requested. Review issue 4 (shifting annotation timing) is excluded.

## Changes

- PPG: preserve missing-data gaps, reject too-short segments, and avoid magnifying flat signals into apparent pulses (issues 1, 2, 22).
- IMU: exclude zero, missing and stale data; use conservative noise guards and sustained threshold crossings; report valid coverage. Thresholds remain engineering heuristics, not validated movement labels (3, 5, 7, 20).
- Breathing: require periodic evidence, reject noise and unsupported fast estimates, mask sustained capacitive dropouts, and avoid automatic notch removal based solely on a stable spectral peak (10, 13, 14).
- Annotations: identify AI drafts, use latest annotations consistently in summaries, preserve manually imported overrides and show their presence in the inventory (9, 11, 12, 18).
- Navigation and inventory: follow the visible time range, keep five-second navigation on its grid, prevent summary selection being reset, inventory channels from headers, and flag/exclude marked trials from aggregate summaries (15–19, 33).
- Loading: choose the organized-data child folder where appropriate, reject stale worker results after folder changes, reuse loaded data, compute derived signals in the worker, retain unavailable folder settings, and accept BOM-prefixed JSON (26–31).
- Launching: report missing Python and startup errors instead of silently failing (32).
- Documentation: remove unsupported posture ranges and stale calibration claims, identify historical agreement figures, describe current processing, and state that saved belt/BIOPAC time axes have not been independently lag-corrected (6, 8, 23–25).

## Verification (issue 21)

URI Unity Slurm job **65440396** completed with exit code **0:0**. Regression tests covered PPG gaps/flat/short data, IMU noise and dropouts, noise-only breathing, fast ECG modulation, and sustained capacitive dropouts. Existing breathing and IMU tests also passed.

All **48 trial extracts** completed numerical processing without exceptions. This is a processing smoke test, not a validation of physiological accuracy; unavailable breathing estimates remain missing rather than being forced into a number. Only numeric extracts were used on Unity.

Windows offscreen UI checks covered signal switching, navigation, folder selection, annotation overrides, and stale-result rejection. Launcher syntax was checked. Restart the application to load the updated files.

No raw dataset files, annotation timing, commits or remote branches were changed. The paper-tracker workbook was not modified. See Movement_algorithm.md and Breathing_algorithm.md for methods and limitations.
