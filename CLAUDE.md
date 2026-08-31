# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Windows desktop research app that turns a chatbot's spoken answer into a **synthetic EEG recording**, then derives neurovascular/metabolic variables from it and renders 3D brain visualizations.

The chain is: user question -> RAG chatbot ("LaRoccoGPT") -> text answer -> ARPAbet phonemes -> per-phoneme index -> concatenated slices of **real pre-recorded EEG** -> Welch PSD -> normalized gamma -> CMRO2 -> neurovascular variables -> 3D scatter frames -> MP4.

There is no test suite, linter, or build step. It is a single-process Tkinter/CustomTkinter app plus a set of standalone analysis scripts.

## Commands

Run everything from the repo root.

```bash
python -m src.gui.phonemizer_V2          # main GUI app (the entry point)
python src/gui/phonemizer_V2.py          # equivalent; both work via the sys.path bootstrap

python src/viz/visualizer.py <path.csv>  # MNE source-localization brain videos, standalone
python scripts/test.py                   # smoke script for visualizer.py (hardcoded CSV path)
python scripts/t_test.py                 # paired t-test / Wilcoxon / Cohen's d over outputs/neurovascular/*.json
python scripts/yo.py                     # older standalone chatbot prototype (no RAG, hardcoded summary)
```

`OPENAI_API_KEY` must be set (env var or `.env`). Dependencies: `pip install -r requirements.txt`.

Note: the checked-in `.venv/` contains only pip — the working interpreter is the system `python`, which already has numpy/scipy/mne installed. Don't assume `.venv/Scripts/python.exe` works.

## Repository layout

```
src/eeg/      analysis pipeline modules (pure functions, no I/O beyond reading the EEG file)
src/gui/      phonemizer_V2.py (entry point), video_player.py, theme.json
src/viz/      visualizer.py (MNE/PyVista brain rendering, runs in its own process)
scripts/      standalone one-off scripts, not imported by the app
data/         inputs: corpus/, eeg_recordings/ (DLR_0..43), vector_db/ (FAISS cache)
outputs/      generated: frames/, neurovascular/, brain_videos/, metabolic_videos/,
              eeg_culmination_csv/, eeg_culmination_txt/
research/     figures and spreadsheets for the write-up
assets/       brain.png
```

All runtime paths are anchored to a `PROJECT_ROOT` constant derived from `__file__`, not to the current working directory. When adding a path, follow that pattern rather than a bare relative string.

## Architecture

### Entry point and global state

`src/gui/phonemizer_V2.py` (~1000 lines) is the whole application. It has **no `if __name__ == "__main__"` guard** — the CustomTkinter widget tree and `root.mainloop()` are at module scope, so *importing* this file launches the GUI. Two `CTk()` roots are created: `root` (main window) and `analyze_window` (EEG analysis window).

State flows between GUI callbacks through module-level globals, not return values:

- `gpt_output` — the chatbot's last answer, consumed by `show_phonemes()`
- `last_generated_tsv_path` / `last_generated_eeg_path` — the `.tsv` and mirrored `.txt` written by `show_phonemes()`; read by `analyze_eeg_input()` and `show_eeg_visualization()`
- `microgap_var` — a checkbox that changes both the tokenization *and* the output filename suffix (`_mg1` = microgaps on, `_mg2` = off)

### The phoneme -> EEG splice (the core trick)

`phoneme_to_number` maps 39 ARPAbet phonemes to 0-38, plus `rand1`-`rand5` to 39-43. `data/eeg_recordings/DLR_<n>_1.txt` holds one pre-recorded EEG session per index (44 files). `show_phonemes()` builds a "sentence EEG" by, for each phoneme, seeking the first row whose first column is exactly `0.000000` and appending 256 rows from there. Whitespace/punctuation tokens map to a random `rand*` index and contribute 256 **or** 512 rows — this is the "microgap".

So the generated EEG is a splice of real recordings, ordered by phoneme. Row count therefore scales with utterance length, which matters (see Gotchas).

### EEG data contract

Raw `DLR_*.txt` and the generated `.tsv` share one layout: 32 tab/whitespace-separated columns, sampled at **256 Hz**. Column 0 is the index/time column (dropped); columns 1-16 are the electrodes, in this exact order, used everywhere:

```
Fp1 Fp2 F3 F4 T5 T6 O1 O2 F7 F8 C3 C4 T3 T4 P3 P4
```

`EEG_Implement_Welch` reads the whitespace-split flat file (`.txt`); `convert_eeg_tsv_to_csv` and `visualizer.py` read the `.tsv`/`.csv`. Both derive from the same `show_phonemes()` write.

### Analysis pipeline modules

Each is a single function in its own file under `src/eeg/`, chained inside `analyze_eeg_input()`:

1. `EEG_Implement_Welch.py` — `EEG_Implement_Welch(path)` returns `(spectra, Trials)`. Splits into trials of `Timestep * TestDuration` = 512 rows, then 32 segments of 32 samples each. Output is a dict keyed **`f'{electrode} trial {n}'`** (n starting at 1) holding per-segment `frequencies`, `power`, and `band_power` for Delta/Theta/Alpha/Beta/Gamma.
2. `EEG_normalizedGamma_CMRO2.py` — normalizes gamma band power to a CMRO2 scale (baseline 2.1, range 2.1-3.1), globally or per-trial.
3. `EEG_NeurovascularVariables.py` — pure-math translation of CMRO2 into 8 variables (`CBF`, `OEF`, `ph_V`, `p_CO2_V`, `pO2_cap`, `CMRO2`, `DeltaHCO2`, `DeltaLAC`) using published physiological constants. Preserves the same trial-key dict shape. Results are dumped to `outputs/neurovascular/*.json`.
4. `EEG_Plotting.py` — plots one timestep on a unit hemisphere. Electrode coordinates for the 10-20 system are hardcoded; `NodeNum` extra nodes are generated with a Latin hypercube (`scipy.stats.qmc`) and their values interpolated by `EEG_DistanceFunc.py` (Manhattan-distance weighted average of the `nNearest` neighbours, plus Gaussian noise).

`analyze_eeg_input()` then renders **32 frames** (`for t in range(32)`) into `outputs/frames/<variable>_<phrase>/` and stitches them with moviepy into `outputs/metabolic_videos/`.

### Two independent visualization backends

Do not confuse them:

- **`src/eeg/EEG_Plotting.py`** — matplotlib 3D scatter on an abstract hemisphere, driven by the neurovascular variables. Used by the "Analyze" window.
- **`src/viz/visualizer.py`** — real MNE source localization (`fsaverage` subject, ico4 source space, dSPM inverse) producing 6 camera-view MP4s plus a 3x2 moviepy collage in `outputs/brain_videos/<phrase>/`. It is launched in a **subprocess** (`launch_in_subprocess`) because PyVista/Qt cannot share a process with the Tkinter mainloop. It imports nothing from the rest of the project, so running it by path works.

### RAG chatbot

`data/corpus/larocco_combined.txt` (an interview/autobiography corpus) is chunked at 500/50 and embedded with `text-embedding-3-large` into a FAISS index cached at `data/vector_db/`. The index is loaded with `allow_dangerous_deserialization=True`; delete the directory to force a rebuild. `combine_prompt` constrains answers to under 20 words and forbids apostrophes — this is deliberate, since the answer becomes the phoneme sequence.

## Gotchas

- **`from playsound import playsound` (src/gui/phonemizer_V2.py:46) is unused but will crash on import.** `playsound` was removed from `requirements.txt`, so a clean install cannot start the app. Delete the import or reinstate the dependency. TTS actually goes through `pyttsx3`.
- **Trial indexing is off by one.** `EEG_Implement_Welch` emits keys `trial 1..N` and returns `Trials = N`, but `plot_normalized_gamma_across_channels` iterates `range(Trials)` = `0..N-1`. It silently skips the last trial and looks for a nonexistent `trial 0`. Verified on `data/eeg_recordings/DLR_0_1.txt`: 9 trials in, 8 out (`trial 1`-`trial 8`).
- **Short utterances crash the pipeline.** One trial = 512 rows; each word contributes 256. With N=1 the loop above matches nothing, `all_gamma_values` is empty, and `min()` raises. Downstream code hardcodes `Trial_Select=1`, so you effectively need >= ~4 tokens of output.
- **`num_segments = 32` (src/eeg/EEG_Implement_Welch.py:65) is coupled to `for t in range(32)`** (src/gui/phonemizer_V2.py:584). `src/eeg/EEG_Implement_Welch_alt.py` is an alternative parameterization (64-sample segments, 8 segments) that would silently break the frame loop — it is not imported anywhere.
- **`src/viz/visualizer.py:22` hardcodes `subjects_dir="C:/Users/anik2/mne_data/MNE-fsaverage-data/"`**, another machine's path. Override the constructor argument.
- **Windows-only in places**: `os.startfile` in `make_collage`, plus `pywin32`/`pypiwin32`/PyQt5 deps.
- **`.gitignore` lists `outputs/brain_videos/` and `outputs/metabolic_videos/`, but their contents are already tracked**, so the ignore is inert for existing files and generated media keeps landing in commits.
