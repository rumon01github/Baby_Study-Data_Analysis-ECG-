# ECG/PPG App

The actual application source is saved in this folder, not only a shortcut:

- neotex_desktop.py: PyQt main app, participant/trial selection and optional analysis backend.
- signal_stack.py: shared-time ECG/motion plots, thresholds and timed motion events.
- study_summary.py: optional window and study summaries.
- study_overview.py: local study-file inventory.
- Movement_algorithm.md: current motion algorithm, formulas, event rules and limitations.

## Run locally

Double-click Launch ECG-PPG App.cmd. Its PowerShell launcher runs neotex_desktop.py from this folder. Local-runtime.json points to the installed Python/dependencies on this computer and is ignored by Git. The .lnk shortcut is optional.

For another computer, install Python and run from the repository root:

    python -m pip install -r desktop_app/requirements.txt
    python desktop_app/neotex_desktop.py

Choose the full Organized Study Data folder once. The app remembers its location locally; Change study folder reconnects relocated recordings. Data stays in its original folder. No raw recordings or installed dependencies are included in tracked app source.

## Current main view

Three aligned plots: measured motion, raw baby-belt ECG and raw BIOPAC ECG. Motion shows gyro-vector magnitude and acceleration-vector change divided by editable thresholds, initialized at the trial 95th percentile. Above 1 crosses the threshold; the caption lists start/end times. This is an exploratory sensor-event display, not validated prediction of infant pickup, left/right bending or module placement.

Use 5-second windows, Previous/Next, drag/zoom and reset. Saved annotations appear as colour-keyed shaded intervals with window summaries. The shared recording-time axis appears only below the bottom plot. No ECG quality is plotted in the main view. Optional analysis tools still reuse repository ECG methods via ../common.py.

See Movement_algorithm.md for the current algorithm. Local recordings, settings, runtime paths, dependencies, caches and shortcuts are excluded from Git. Source files, launch scripts, requirements, README and the algorithm document can be committed on your branch.

Latest video-derived annotation snapshot: annotations_latest/ contains 12 CSV files across 7 participants, fetched from Drive folder 1GfyUbeC9SjSGbGTYEP0OQbUYj0AF5syE on 2026-10-06. These files override older annotations for matching trials. Per user confirmation, video_start_s/video_end_s are synchronized recording times and are used directly without offsets. P03 T01/T02 remain marked as drafts and are shown with a draft warning. Other files are marked human_reviewed. CSVs are git-ignored study data. This is a local snapshot, not live Drive synchronization.
