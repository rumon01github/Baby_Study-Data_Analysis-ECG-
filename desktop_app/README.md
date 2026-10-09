# ECG/PPG App

The actual application source is saved in this folder, not only a shortcut:

- neotex_desktop.py: PyQt main app, participant/trial selection and optional analysis backend.
- signal_stack.py: shared-time ECG/motion plots, thresholds and timed motion events.
- study_summary.py: optional window and study summaries.
- study_overview.py: local study-file inventory.
- imu_posture.py: belt tilt angles and activity in g from the accelerometer (test_imu_posture.py: synthetic checks).
- Movement_algorithm.md: current motion algorithm, formulas, event rules and limitations.

## Run locally

Double-click ECG-PPG_Neotex.cmd. Its PowerShell launcher runs neotex_desktop.py from this folder. Local-runtime.json points to the installed Python/dependencies on this computer and is ignored by Git. The .lnk shortcut is optional.

For another computer, install Python and run from the repository root:

    python -m pip install -r desktop_app/requirements.txt
    python desktop_app/neotex_desktop.py

Choose the full Organized Study Data folder once. The app remembers its location locally; Change study folder reconnects relocated recordings. Data stays in its original folder. No raw recordings or installed dependencies are included in tracked app source.

## Current main view


Use 5-second windows, Previous/Next, drag/zoom and reset. Saved annotations appear as colour-keyed shaded intervals with window summaries. The shared recording-time axis appears only below the bottom plot. No ECG quality is plotted in the main view. Optional analysis tools still reuse repository ECG methods via ../common.py.

See Movement_algorithm.md for the current algorithm. Local recordings, settings, runtime paths, dependencies, caches and shortcuts are excluded from Git. Source files, launch scripts, requirements, README and the algorithm document can be committed on your branch.

Latest video-derived annotation snapshot: annotations_latest/ contains 12 CSV files across 7 participants, fetched from Drive folder 1GfyUbeC9SjSGbGTYEP0OQbUYj0AF5syE on 2026-10-06. These files override older annotations for matching trials. Per user confirmation, video_start_s/video_end_s are synchronized recording times and are used directly without offsets. P03 T01/T02 remain marked as drafts and are shown with a draft warning. Other files are marked human_reviewed. CSVs are git-ignored study data. This is a local snapshot, not live Drive synchronization.

## Breathing view
Select Breathing in the Signal dropdown. Four plots, in order: belt capacitive respiration (Resp0 left, Resp1 right; saved values faint, spikes removed bold, counted breaths marked ▼), the IMU (chest tilt) breathing rate, exploratory belt breathing rates from ECG and the left/right capacitive dips, and BIOPAC ECG-derived breathing rate calculated by the app with the same method as the belt (the saved AcqKnowledge rate is not shown: it never exceeds 20 breaths/min). Belt rates give one value per 5-second window, each estimated from 30 seconds centred on it; gaps mean unavailable. Use How calculated? or read Breathing_algorithm.md for methods and limits.

## PPG view
Select PPG in the Signal dropdown. The belt plot overlays IR (blue) and Red (red); BIOPAC PPG is shown below, with its sensor site in the title: leg for P1–P6 and P8, hand for P9–P10 (P7 has no BIOPAC recording). Sites are listed in BIOPAC_PPG_SITE in signal_stack.py. Belt IR/Red and BIOPAC PPG are filtered before plotting with ppg_processing.py, which follows MINDER-V1 biomedical_filters.filter_ppg_robust (the version used by its reviewer and Stress vs Relaxation tools): missing samples bridged (short gaps interpolated, longer gaps mean-filled for filtering and then shown as gaps), 9-sample median despike, linear detrend, order-4 zero-phase Butterworth 0.5–10 Hz bandpass. The whole trial is filtered once; each visible time window is then z-scored when shown, as in MINDER, so artifacts elsewhere do not compress its scale. The y-axis is therefore unitless (z-score of the visible window). The median filter is in samples: 90 ms on the 100 Hz belt, 9 ms on 1000 Hz BIOPAC. Parameters come from adult MINDER data and have not been tuned for infants. No -99 sentinel handling or quality classes are applied. ECG and breathing traces are unchanged.

## Review update — 8 October 2026
Launch ECG-PPG_Neotex.cmd. Errors remain visible. Use Organized Study Data; choosing its parent automatically selects that child. Scanning and trial calculations run in background workers. A folder is remembered only after a trial loads successfully. An unavailable saved folder is retained for reconnection. Excluded trials are marked and omitted from combined summaries.

Source recordings and annotation times have not been changed. Saved-time axes do not correct device drift. AI draft annotation status is shown briefly; it is not an activity or signal-quality label. Manually loaded annotations take precedence for that trial in the current session. Latest annotation snapshots are included in overview and summary.

PPG: zero readings accompanied by zero HR and SpO2 are missing. Filtering never crosses missing segments; two-second edges are blanked. Flat/negligible windows are not amplified. IMU: zero/stale samples excluded before processing; robust noise guard and persistence replace the old permissive classification. Breathing: periodicity/modulation gates reject unsupported windows instead of presenting every periodic-looking trace as breathing. See the current rules at the top of the algorithm documents. Rates remain exploratory, with gaps when evidence is insufficient.
