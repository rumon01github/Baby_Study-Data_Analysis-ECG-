"""Central configuration for the NeoTex ECG analysis.
Set NEOTEX_DATA to the BABY DATA folder (run_pipeline.py / gui.py do this for you). The sub-folders are found automatically
in either layout:  'Organized Study Data' / 'Trials with Videos' / <handoff folder>   or   data/trials, data/videos, data/handoff.
Each can also be overridden with NEOTEX_BELT, NEOTEX_VIDEO, NEOTEX_HANDOFF; outputs go to NEOTEX_WORK (default ./work)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
def _norm(p): return p.replace('\\', '/').rstrip('/') + '/'
def _first(*cands):
    for c in cands:
        if c and os.path.isdir(c): return _norm(c)
    return _norm(next(c for c in cands if c))   # none exists: return the first named one (steps that need it are skipped)
DATA = _norm(os.environ.get('NEOTEX_DATA', os.path.join(_HERE, 'data')))
BELT = _first(os.environ.get('NEOTEX_BELT'), DATA + 'Organized Study Data', DATA + 'trials', DATA)             # <P>/<T>/*BELT*.csv (+ *BIOPAC*.csv)
VIDEO = _first(os.environ.get('NEOTEX_VIDEO'), DATA + 'Trials with Videos', DATA + 'videos')                  # <P>/<anything>/<P>_<T>_annotations.csv
HANDOFF = _first(os.environ.get('NEOTEX_HANDOFF'), DATA + 'handoff', DATA + '../NeoTex HR Lab/exports/NeoTex manual review handoff 20261004T093331Z')
WORK = _norm(os.environ.get('NEOTEX_WORK', os.path.join(_HERE, 'work')))
FIGS = WORK + 'figures/'
os.makedirs(WORK, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
# --- analysis constants (fixed before the full manual review; do not re-tune) --
WIN = 10.0          # s, reliability window
TOL = 0.075         # s, beat-matching tolerance (ANSI/AAMI EC57 style)
BSQI_MIN = 0.90     # two-detector agreement
HR_MIN, HR_MAX = 80, 220
RRCV_MAX = 0.25
MOVE_OFFSET = 200   # gyro counts above the trial's 20th percentile = "moving"
STILL_MAX, LIGHT_MAX = 0.10, 0.50   # share of samples moving: still < 10 %, light 10-50 %, active > 50 %
MIN_REF_WINDOWS = 3                 # trial enters comparison if BIOPAC has >= 3 reliable windows
CONTACT_GOOD, CONTACT_POOR = 0.60, 0.20
