#!/usr/bin/env python3
"""One-command runner for the NeoTex ECG analysis.

    python run_pipeline.py --data "<BABY DATA folder>"                 # full pipeline + figures
    python run_pipeline.py --data ... --steps 1,2,5 --no-figures       # subset
    python run_pipeline.py --synthetic                                 # smoke test on generated data (no real recordings needed)

Paths can also come from the environment (NEOTEX_DATA, NEOTEX_BELT, NEOTEX_HANDOFF, NEOTEX_WORK). Each step is a plain script;
this runner only sets the paths, runs them in order and stops at the first failure.
"""
import argparse, os, subprocess, sys, time

STEPS = {  # id: (script, needs)   needs: 'base' = belt+BIOPAC CSVs, 'video' = annotation files, 'handoff' = manual-review export
    '1': ('01_build_beats_cache.py', 'base'), '2': ('02_windows_and_trials.py', 'base'),
    '3': ('03_video_offsets.py', 'video'), '4': ('04_events.py', 'video'), '5': ('05_event_stats.py', 'video'),
    '6': ('06_manual_handoff.py', 'handoff'), '7a': ('07a_tune_qrs.py', 'base'), '7b': ('07b_val_qrs_v1.py', 'base'), '7c': ('07c_val_qrs_v2.py', 'base'),
    '7d': ('07d_eval_handoff.py', 'handoff'), '7e': ('07e_grid_guards.py', 'handoff'), '8a': ('08a_scorer_features.py', 'handoff'), '8b': ('08b_scorer_train.py', 'handoff'), '8c': ('08c_sqi_tables.py', 'base'),
    '9': ('09_motion_context.py', 'video'), '10': ('10_tables.py', 'handoff'), '11': ('11_workbook.py', 'video'),
}
FIGS = ['fig6_pipeline.py', 'fig1_ecg_morphology.py', 'fig2_beat_detectors.py', 'fig3_event_snapshots.py', 'fig4_reliability_vs_motion_and_events.py',
        'fig5_event_and_window_quality.py', 'fig7_pipeline_validation.py']
FIG_NEEDS = {'fig6_pipeline.py': 'none', 'fig1_ecg_morphology.py': 'base', 'fig2_beat_detectors.py': 'handoff', 'fig3_event_snapshots.py': 'video',
             'fig4_reliability_vs_motion_and_events.py': 'video', 'fig5_event_and_window_quality.py': 'video', 'fig7_pipeline_validation.py': 'handoff'}
ORDER = ['1', '2', '3', '4', '5', '6', '7a', '7b', '7c', '7d', '7e', '8a', '8b', '8c', '9', '10', '11']

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', help='BABY DATA folder (holds Organized Study Data, Trials with Videos)')
    ap.add_argument('--belt', help='folder with <P>/<T>/*BELT*.csv (default: <data>/Organized Study Data)')
    ap.add_argument('--handoff', help='manual-review handoff folder (ALL_MANUAL_BEATS.csv + strip_summary.csv)')
    ap.add_argument('--work', help='output folder (default: ./work)')
    ap.add_argument('--steps', default='all', help="comma list from 1,2,3,4,5,6,7a,7b,7c,7d,7e,8a,8b,8c,9,10,11 or 'all'")
    ap.add_argument('--no-figures', action='store_true'); ap.add_argument('--figures-only', action='store_true')
    ap.add_argument('--synthetic', action='store_true', help='generate synthetic belt/BIOPAC recordings into ./synthetic_data and run the core steps on them')
    a = ap.parse_args()
    env = dict(os.environ)
    if a.synthetic:
        subprocess.run([sys.executable, 'make_synthetic_data.py'], check=True)
        env['NEOTEX_DATA'] = os.path.abspath('synthetic_data') + os.sep; env['NEOTEX_BELT'] = env['NEOTEX_DATA'] + 'Organized Study Data' + os.sep
        env['NEOTEX_WORK'] = a.work or os.path.abspath('work_synthetic')
        if a.steps == 'all': a.steps = '1,2,7b,7c'
        figs = ['fig6_pipeline.py']   # the other figure scripts show named study trials
    else:
        if a.data: env['NEOTEX_DATA'] = os.path.abspath(a.data) + os.sep
        if a.belt: env['NEOTEX_BELT'] = os.path.abspath(a.belt) + os.sep
        elif a.data: env['NEOTEX_BELT'] = env['NEOTEX_DATA'] + 'Organized Study Data' + os.sep
        if a.handoff: env['NEOTEX_HANDOFF'] = os.path.abspath(a.handoff) + os.sep
        if a.work: env['NEOTEX_WORK'] = os.path.abspath(a.work)
        figs = FIGS
    # resolve the folders exactly as config.py will, so the skip logic agrees with the scripts
    os.environ.update(env); import importlib, config; importlib.reload(config)
    have = {'none': True, 'base': True,
            'video': any(f.endswith('_annotations.csv') for _, _, fs in os.walk(config.VIDEO) for f in fs) if os.path.isdir(config.VIDEO) else False,
            'handoff': os.path.isfile(os.path.join(config.HANDOFF, 'Manual labels', 'ALL_MANUAL_BEATS.csv'))}
    print('data   :', config.DATA); print('belt   :', config.BELT, '' if os.path.isdir(config.BELT) else '(missing!)')
    print('video  :', config.VIDEO, 'found' if have['video'] else '(no annotation files -> video steps skipped)')
    print('handoff:', config.HANDOFF, 'found' if have['handoff'] else '(not found -> expert-label steps skipped)')
    steps = ORDER if a.steps == 'all' else [s.strip() for s in a.steps.split(',')]
    if a.figures_only: steps = []
    def run(script):
        t = time.time(); print(f'\n=== {script}', flush=True)
        r = subprocess.run([sys.executable, script], env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode: sys.exit(f'{script} failed (exit {r.returncode})')
        print(f'    done in {time.time() - t:.0f} s')
    for s in steps:
        script, need = STEPS[s]
        if not have[need]: print(f'--- skipping {script}: needs {need} data'); continue
        run(script)
    if not a.no_figures:
        for f in figs:
            if have[FIG_NEEDS[f]]: run(f)
            else: print(f'--- skipping {f}: needs {FIG_NEEDS[f]} data')
    print('\nOutputs are in', env.get('NEOTEX_WORK', os.path.abspath('work')))
if __name__ == '__main__': main()
