#!/usr/bin/env python3
"""Small desktop launcher for the NeoTex ECG analysis (no extra dependencies; uses the Python standard library's Tkinter).

    python gui.py

Pick the data folders, tick the steps you want, press Run. The log appears in the window; outputs go to the chosen output folder.
Everything the GUI does can also be done from the command line with run_pipeline.py.
"""
import os, sys, subprocess, threading, queue, webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [('1', '1  Beat cache (NeuroKit2 + Hamilton on belt and BIOPAC)'), ('2', '2  10-s windows, gate, trial table, contact classes'),
         ('3', '3  Video offsets (needs Trials with Videos)'), ('4', '4  Events (needs Trials with Videos)'), ('5', '5  Event + window statistics'),
         ('6', '6  Expert labels (needs manual-review handoff)'), ('7b', '7b Validate detector v1 vs reference'), ('7c', '7c Validate detector v2 vs reference'),
         ('7d', '7d Detectors vs expert'), ('7e', '7e Artifact-guard grid'), ('8a', '8a Scorer features'), ('8b', '8b Scorer training (leave-one-infant-out)'), ('8c', '8c Descriptive SQI tables'),
         ('9', '9  Motion context'), ('10', '10 Paper tables'), ('11', '11 Excel workbook')]

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('NeoTex ECG analysis'); self.geometry('860x680'); self.minsize(720, 560)
        self.q = queue.Queue(); self.proc = None
        f = ttk.Frame(self, padding=12); f.pack(fill='both', expand=True)
        # paths
        self.v_data = tk.StringVar(value=os.environ.get('NEOTEX_DATA', '')); self.v_handoff = tk.StringVar(value=os.environ.get('NEOTEX_HANDOFF', ''))
        self.v_work = tk.StringVar(value=os.path.join(HERE, 'work'))
        for r, (lab, var, isdir) in enumerate([('BABY DATA folder', self.v_data, True), ('Manual-review handoff folder (optional)', self.v_handoff, True), ('Output folder', self.v_work, True)]):
            ttk.Label(f, text=lab).grid(row=r, column=0, sticky='w', pady=3)
            ttk.Entry(f, textvariable=var, width=70).grid(row=r, column=1, sticky='ew', padx=6)
            ttk.Button(f, text='Browse…', command=lambda v=var: v.set(filedialog.askdirectory() or v.get())).grid(row=r, column=2)
        f.columnconfigure(1, weight=1)
        # steps
        sf = ttk.LabelFrame(f, text='Steps', padding=8); sf.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(10, 4))
        self.vars = {}
        for i, (k, lab) in enumerate(STEPS):
            v = tk.BooleanVar(value=True); self.vars[k] = v
            ttk.Checkbutton(sf, text=lab, variable=v).grid(row=i % 9, column=i // 9, sticky='w', padx=6)
        self.v_figs = tk.BooleanVar(value=True); ttk.Checkbutton(sf, text='Draw the seven figures', variable=self.v_figs).grid(row=8, column=1, sticky='w', padx=6)
        bf = ttk.Frame(f); bf.grid(row=4, column=0, columnspan=3, sticky='ew', pady=6)
        ttk.Button(bf, text='Select all', command=lambda: [v.set(True) for v in self.vars.values()]).pack(side='left')
        ttk.Button(bf, text='Select none', command=lambda: [v.set(False) for v in self.vars.values()]).pack(side='left', padx=4)
        self.b_run = ttk.Button(bf, text='▶ Run', command=self.run); self.b_run.pack(side='right')
        ttk.Button(bf, text='Demo on synthetic data', command=lambda: self.run(synthetic=True)).pack(side='right', padx=6)
        ttk.Button(bf, text='Open output folder', command=self.open_out).pack(side='right', padx=6)
        ttk.Button(bf, text='Install dependencies', command=self.install).pack(side='left', padx=12)
        self.b_stop = ttk.Button(bf, text='Stop', command=self.stop, state='disabled'); self.b_stop.pack(side='right', padx=6)
        # log
        self.log = tk.Text(f, height=20, wrap='word', font=('Consolas' if sys.platform == 'win32' else 'Menlo', 9)); self.log.grid(row=5, column=0, columnspan=3, sticky='nsew')
        sb = ttk.Scrollbar(f, command=self.log.yview); sb.grid(row=5, column=3, sticky='ns'); self.log.configure(yscrollcommand=sb.set)
        f.rowconfigure(5, weight=1)
        self.status = tk.StringVar(value='Ready. Set the BABY DATA folder (or try the synthetic demo), tick steps, press Run.')
        ttk.Label(f, textvariable=self.status).grid(row=6, column=0, columnspan=3, sticky='w', pady=(6, 0))
        self.after(150, self.pump); self.after(300, self.check_deps)

    def check_deps(self):
        missing = [m for m in ['numpy', 'pandas', 'scipy', 'sklearn', 'neurokit2', 'plotly', 'kaleido', 'pdfplumber', 'openpyxl'] if not self._has(m)]
        self.append(f'Python: {sys.executable}\n')
        if missing:
            self.append('Missing packages: ' + ', '.join(missing) + '\nPress "Install dependencies" (installs into the Python shown above), then Run.\n')
            self.status.set('Dependencies missing - press "Install dependencies".')
        else: self.append('All packages found.\n')
    @staticmethod
    def _has(mod):
        import importlib.util; return importlib.util.find_spec(mod) is not None
    def install(self):
        if self.proc: return
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', os.path.join(HERE, 'requirements.txt')]
        self.log.delete('1.0', 'end'); self.append('$ ' + ' '.join(cmd) + '\n'); self.status.set('Installing packages…'); self.b_run.configure(state='disabled'); self.b_stop.configure(state='normal')
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self.reader, daemon=True).start()

    def run(self, synthetic=False):
        if self.proc: return
        cmd = [sys.executable, os.path.join(HERE, 'run_pipeline.py')]
        if synthetic: cmd.append('--synthetic')
        else:
            if not os.path.isdir(self.v_data.get()): messagebox.showerror('Missing data', 'Choose the BABY DATA folder first (or use the synthetic demo).'); return
            cmd += ['--data', self.v_data.get()]
            if self.v_handoff.get(): cmd += ['--handoff', self.v_handoff.get()]
            steps = [k for k, v in self.vars.items() if v.get()]
            if not steps and not self.v_figs.get(): messagebox.showinfo('Nothing to do', 'Tick at least one step or the figures.'); return
            cmd += ['--steps', ','.join(steps) if steps else '1']
            if not steps: cmd.append('--figures-only')
            if not self.v_figs.get(): cmd.append('--no-figures')
        if self.v_work.get(): cmd += ['--work', self.v_work.get()]
        self.log.delete('1.0', 'end'); self.append('$ ' + ' '.join(cmd) + '\n'); self.status.set('Running…'); self.b_run.configure(state='disabled'); self.b_stop.configure(state='normal')
        self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self.reader, daemon=True).start()
    def reader(self):
        for line in self.proc.stdout: self.q.put(line)
        self.q.put(None)
    def pump(self):
        try:
            while True:
                line = self.q.get_nowait()
                if line is None:
                    code = self.proc.wait(); self.proc = None; self.b_run.configure(state='normal'); self.b_stop.configure(state='disabled')
                    self.status.set('Finished.' if code == 0 else f'Stopped with error (exit {code}); see the log.'); break
                if 'Warning' in line or 'warn' in line: continue
                self.append(line)
        except queue.Empty: pass
        self.after(150, self.pump)
    def append(self, s): self.log.insert('end', s); self.log.see('end')
    def stop(self):
        if self.proc: self.proc.terminate()
    def open_out(self):
        p = self.v_work.get() or os.path.join(HERE, 'work')
        if not os.path.isdir(p): messagebox.showinfo('No output yet', 'Run the pipeline first.'); return
        if sys.platform == 'win32': os.startfile(p)
        elif sys.platform == 'darwin': subprocess.Popen(['open', p])
        else: webbrowser.open('file://' + p)

if __name__ == '__main__': App().mainloop()
