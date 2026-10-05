# NeoTex ECG analysis

Reproducible analysis code for the ECG / heart-rate (HR) part of the NeoTex paper: a textile chest belt for infants, validated
against a wired BIOPAC reference in 10 infants (48 trials) with expert-labelled beats (21.9 k R peaks) and video-annotated events.

The pipeline turns raw belt ECG into HR with a **reference-free reliability flag**, and the scripts here reproduce every number,
table and figure in the paper's ECG results:

* a lightweight belt-specific R-peak detector (`neotex_qrs_v2.py`, ~15 multiply–accumulates per sample, no external library)
* a reliability gate (two-detector agreement + HR range + R–R regularity) validated against the reference
* expert-in-the-loop labels, a learned beat verifier and its learning curve (leave-one-infant-out)
* reliability versus motion, belt fit and video-annotated infant events
* seven publication figures (ACM two-column width, Plotly, vector PDF)

> **Data are not included.** The recordings are infant physiological data collected under an IRB protocol and are shared only
> through the study team. The code runs on the folder layout described below, and `--synthetic` generates a small artificial
> data set so the code path can be exercised by anyone.

---

## 1. How to use it

There are three ways to run the same pipeline; pick one.

**A. Desktop window (easiest).** Double-click `gui.py` (or run `python gui.py`). Choose the *BABY DATA* folder, optionally the
manual-review handoff folder, tick the steps you want, press **Run**. The log shows progress; **Open output folder** shows the
tables and figures. **Demo on synthetic data** runs without any recordings. The window needs nothing beyond the standard
Python install (Tkinter) plus the packages in `requirements.txt`.

**B. One command.**
```bash
python run_pipeline.py --data "C:/path/to/BABY DATA" --handoff "C:/path/to/NeoTex manual review handoff 20261004T093331Z"
```

**C. Script by script.** Set the paths in `config.py`, then run `01_…py`, `02_…py`, … in order (see section 4).

Outputs: `work/*.csv` (tables), `work/figures/Fig*.pdf|png`, `work/NeoTex_Quality_Quadrants_Events.xlsx`.

## 2. Quick start

```bash
git clone <this repo>
cd neotex-ecg-analysis
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# smoke test without real data (≈ 20 s): generates synthetic recordings, runs the core steps, draws the pipeline figure
python run_pipeline.py --synthetic

# full analysis on the study data
python run_pipeline.py --data "C:/path/to/BABY DATA" --handoff "C:/path/to/NeoTex manual review handoff 20261004T093331Z"
```

Outputs go to `work/` (tables as CSV, figures in `work/figures/`). `run_pipeline.py --help` lists the options; `--steps 1,2,5`
runs a subset, `--no-figures` / `--figures-only` split analysis and plotting.

**Windows note.** Windows 11 *Smart App Control* blocks unsigned compiled files in user folders (error: *"An Application Control policy has blocked this file"* when importing scikit-learn). Turn it off in Windows Security → App & browser control → Smart App Control, or run the pipeline under WSL. Recommended Windows setup is a private environment: `uv venv --python 3.12 .venv && .venv\Scripts\activate && uv pip install -r requirements.txt`.

Python 3.10–3.13. Tested on Linux with the versions pinned in `requirements.txt`; `environment.yml` is provided for conda.
Figures need a Chromium for Plotly's `kaleido` exporter; if it is not found automatically set `BROWSER_PATH` to the Chrome /
Chromium executable.

---

## 3. Expected data layout

```
BABY DATA/
├── Organized Study Data/                 # one folder per infant and trial
│   └── P03/T01/
│       ├── P03_20260204_T01_BELT_F_Independent.csv      # belt, 100 Hz
│       └── P03_20260204_T01_BIOPAC_F_Independent.csv    # reference, 1000 Hz
└── Trials with Videos/                   # optional: video-annotated trials
    └── P03/P03_..._T01_.../P03_T01_annotations.csv

NeoTex manual review handoff .../         # optional: expert beat labels
├── Manual labels/ALL_MANUAL_BEATS.csv
└── Review report/strip_summary.csv
```

| file | required columns | notes |
|---|---|---|
| `*BELT*.csv` | `time_s, ECG, AccX, AccY, AccZ, GyroX, GyroY, GyroZ` | ECG in mV (first-difference front end), IMU raw counts (4096 counts/g); file name carries the placement (`Independent` / `ParentHeld`) |
| `*BIOPAC*.csv` | `time_s` + at least one column whose name contains `ECG MODULE` or `AHA` | the ECG channel is chosen automatically by beat plausibility |
| `*_annotations.csv` | `participant, trial, video_start_s, video_end_s, label` and, when known, `ecg_start_s, ecg_end_s` | when `ecg_*` are empty the offset in `video_offsets_final.csv` is applied |
| `ALL_MANUAL_BEATS.csv` | `trial, strip_start_s, peak_time_s` | one row per expert R marker |
| `strip_summary.csv` | `trial, start_s, quality, confidence` | quality ∈ readable / partly_readable / uncertain / unreadable |

Steps that need video annotations or expert labels are skipped automatically when those folders are absent.

---

## 4. What each script does

Scripts are numbered in run order. Every step reads its inputs from `work/` and writes its outputs there; `run_pipeline.py`
only sets the paths and runs them in sequence.

| step | script | produces | used by |
|---|---|---|---|
| 1 | `01_build_beats_cache.py` | `beats_cache.pkl` — NeuroKit2 and Hamilton–Tompkins beats for belt (250 Hz) and reference (500 Hz) | everything |
| 2 | `02_windows_and_trials.py` | `quadrants.csv` (per 10-s window: gate decision for both devices after per-window lag alignment, HR, SQIs, motion level), `trial_table_v2.csv` (inclusion rule, belt contact class), `contact_first60.csv` (non-circular 60-s contact check) | Figs 1, 4, 5, 7; Tables |
| 3 | `03_video_offsets.py` | `video_offsets_12.csv` — video-to-data offset by annotation–gyroscope cross-correlation | step 4 (via the curated `video_offsets_final.csv`) |
| 4 | `04_events.py` | `events12.csv` (all resolved videos), `events.csv` (trials with validated alignment), each event gated per device | Figs 3, 4c, 5 |
| 5 | `05_event_stats.py` | prints every statistic quoted in the Results (McNemar, trial bootstrap, per-trial Wilcoxon, contact splits, Bland–Altman); `s5_failure_attribution.csv` | Fig 7c, text |
| 6 | `06_manual_handoff.py` | `manual_beats_clean.csv` (boundary duplicates merged), `strip_summary_hr.csv`, `strips_gate_vs_human.csv` (run-time detector + gate vs expert grades) | 7d–8b, Fig 2 |
| 7a | `07a_tune_qrs.py` | detector filter grid on the first 23 labelled strips (odd strips tune, even test) → `neotex_qrs_params.json` | — |
| 7b, 7c | `07b_val_qrs_v1.py`, `07c_val_qrs_v2.py` | window-level validation of the run-time detector + gate against clean reference windows → `val_qrs_windows*.csv` | Fig 7b |
| 7d | `07d_eval_handoff.py` | per-strip F1 of detector v1 and v2 against the expert → `handoff_eval.csv` | Fig 2, Table 2 |
| 7e | `07e_grid_guards.py` | artifact-guard grid (clip, τ, dead time); odd infants choose, even infants check → `grid_v2.csv` | text |
| 8a, 8b | `08a_scorer_features.py`, `08b_scorer_train.py` | candidate beats + features; leave-one-infant-out gradient-boosted verifier; learning curve → `s3_*.csv` | Fig 2, Fig 7a |
| 8c | `08c_sqi_tables.py` | descriptive SQIs per window → `windows_sqi_ext.csv` | step 9 |
| 9 | `09_motion_context.py` | motion-class clustering / classifier from IMU + ECG features (reported as a negative result) | text |
| 10 | `10_tables.py` | `detector_table_strips.csv`, `tbl_bsqi_sens.csv`, `tbl_per_infant.csv` | Tables 1–3 |
| 11 | `11_workbook.py` | `NeoTex_Quality_Quadrants_Events.xlsx` — all intermediate tables in one workbook | supplementary |
| figures | `fig1_…py` … `fig7_…py` | `work/figures/Fig*.pdf` + `.png` (600 dpi) | paper |

`08c_sqi_tables.py` adds descriptive signal-quality indices (kurtosis, QRS power ratio, baseline SQI) per window; step 9 and the
supplementary SQI panels need it.

### Core modules

| module | content |
|---|---|
| `config.py` | data roots and **all analysis constants** (window 10 s, ±75 ms tolerance, bSQI ≥ 0.90, HR 80–220 bpm, R–R CV < 0.25, motion thresholds, inclusion rule, contact classes). These were fixed before the full manual review and are not tuned to the results. |
| `common.py` | beat matching (`f1`), the reliability gate (`sqi`), belt CSV loading, per-window lag alignment (`local_lag`), motion level |
| `neotex_qrs.py`, `neotex_qrs_v2.py` | the belt R-peak detector: band-pass 5–25 Hz → derivative → square → 50-ms integration → adaptive threshold with refractory period and search-back → R refinement. v2 adds two artifact guards (clipped signal-level update, lock-up decay). `rr_quality()` is the run-time HR-range / R–R-regularity check. Parameters in the two `*.json` files. |
| `gui.py` | desktop launcher (Tkinter) around `run_pipeline.py` |
| `run_pipeline.py` | command-line runner: sets paths, runs the numbered steps in order, skips steps whose data are absent |
| `make_synthetic_data.py` | generates artificial recordings in the expected layout for the demo / smoke test |
| `pstyle.py` | Plotly style for ACM sigconf figures: 504 pt width, Liberation Sans 6.8–9 pt, Okabe–Ito colours, PDF export at 1 px = 1 pt, automatic QA for font size and overlapping labels |

### Bundled inputs (small, non-identifying)

`trial_metadata.csv` (trial → infant, placement), `segment_key.csv` + `round1_labels.json` (the first 23 labelled strips used
for detector tuning), `label_eval.csv` (first-round detector evaluation), `video_offsets_final.csv` (curated video→data offsets
with provenance notes).

---

## 5. Using the detector on your own belt data

```python
import numpy as np, json
from neotex_qrs_v2 import detect, rr_quality
P = json.load(open('neotex_qrs_params_v2.json'))
x = np.loadtxt('belt_ecg_100hz.csv')          # 1-D ECG at 100 Hz
r = detect(x, fs=100, **P)                     # R-peak sample indices
for a in range(0, len(x) - 1000, 1000):       # 10-s windows
    q = rr_quality(r[(r >= a) & (r < a + 1000)] - a, fs=100)
    print(a / 100, 's', round(q['hr'], 1), 'bpm', 'reliable' if q['ok'] else 'not reliable')
```
Only NumPy and SciPy are needed for this part.

---

## 6. Method summary

* **Gate.** A 10-s window reports HR only if two independent detectors agree (F1 ≥ 0.90 at ±75 ms), the median HR is 80–220 bpm
  and the R–R coefficient of variation is < 0.25. Morphology indices are descriptive only.
* **Alignment.** The belt clock drifts; every belt–reference comparison uses a per-window lag found by beat matching (±1 s).
* **Trial inclusion.** A trial enters the belt-vs-reference comparison when the reference itself has ≥ 3 reliable windows.
* **Belt fit.** The belt's own reliable-window share (good ≥ 60 %, poor ≤ 20 %); a non-circular variant classifies from the first
  60 s only and evaluates the rest of the trial.
* **Expert labels.** 1,819 strips of 8 s, graded readable / partly readable / uncertain / unreadable; markers duplicated at strip
  boundaries are merged. Labels were proposal-assisted (detector proposals visible to the reviewer).
* **Evaluation.** Beat level: sensitivity / PPV / F1 at ±75 ms, leave-one-infant-out for anything learned. Window level: reliable
  share, MAE and within-5-bpm share against clean reference windows, Bland–Altman. Event level: exact McNemar on discordant
  events, trial-cluster bootstrap, per-trial Wilcoxon.

---

## 7. Reproducibility notes

* Re-running step 2 reproduces the paper's window table to within lag-search ties (99.6 % of window decisions identical;
  36 trials in comparison; 18 good / 22 poor / 2 partial contact).
* Steps 7d, 8a, 8b and 9 take a few minutes each on a laptop; everything else runs in seconds.
* Figure scripts display named study trials (e.g. `P03 T01`); edit the trial IDs at the top of each `fig*.py` to show others.
* Randomness: bootstraps and the scorer use fixed seeds (`numpy` `default_rng(0/1)`, `random_state=0`).

## 8. Citing

See `CITATION.cff`. Please cite the paper when using the detector or the evaluation protocol.

## 9. License

MIT (see `LICENSE`). The license covers the code only; the study data are not part of this repository.
