# thinktank-clearmind

Turns a sentence into a synthetic EEG recording, derives neurovascular and
metabolic variables from it, and renders the result as a 3D brain animation.

This is the reference implementation for **"Imagined Speech Reconstruction with
3D Neural Metabolism and Large Language Model Integration."** It is a research
prototype, not a clinical or diagnostic tool: the EEG it analyses is
*synthesized* by splicing real pre-recorded phoneme segments, and the metabolic
values it reports are model estimates rather than measurements.

## How it works

```
question -> RAG chatbot -> text answer -> ARPAbet phonemes -> per-phoneme index
         -> spliced real EEG -> Welch PSD -> normalized gamma -> CMRO2
         -> neurovascular variables -> 3D frames -> MP4
```

A GPT-4 agent, grounded by retrieval over an abridged autobiography, produces a
short answer. Each phoneme in that answer indexes into one of 44 pre-recorded
EEG sessions, and one second of real signal is drawn from each. Pseudorandom
"microgaps" of 1 or 2 seconds are inserted between words to mimic natural
speech pauses. The resulting composite is analysed for gamma-band power, mapped
onto an estimate of the cerebral metabolic rate of oxygen consumption (CMRO2),
and expanded into eight neurovascular variables that are painted onto a cortical
surface.

The central experimental result concerns those microgaps: inserting them
significantly changes every modelled neurovascular variable.

## Requirements

- Python 3.12 (developed on 3.12.7 under Anaconda Distribution 2024.10-1)
- An OpenAI API key, as `OPENAI_API_KEY` in the environment or a `.env` file
- Windows, in practice — the GUI depends on `pywin32` and PyQt5, and video
  playback uses `os.startfile`

```bash
pip install -r requirements.txt
```

The 3D source-localization stage downloads MNE's `fsaverage` template on first
use. To reuse an existing copy, set `SUBJECTS_DIR` or pass `subjects_dir` to
`EEGVisualizer`.

## Usage

Run everything from the repository root.

```bash
python -m src.gui.phonemizer_V2          # the application
python src/viz/visualizer.py <path.csv>  # brain videos, standalone
python scripts/t_test.py                 # paired t-test / Wilcoxon / Cohen's d
```

The GUI has two panels: a chat panel for talking to the agent, and a panel that
shows the decomposed phoneme sequence and drives the visualizations. Generated
media is written to `outputs/`.

Note that `src/gui/phonemizer_V2.py` builds its widget tree at module scope, so
importing it launches the GUI.

Utterances need roughly four or more tokens. One analysis trial is 512 samples
and each phoneme contributes 256, so very short answers leave the spectral stage
with nothing to work on.

## Layout

```
src/eeg/     analysis pipeline: Welch PSD, gamma to CMRO2, neurovascular
             variables, hemisphere plotting, distance interpolation
src/gui/     phonemizer_V2.py (entry point), video_player.py, theme.json
src/viz/     visualizer.py -- MNE source localization, runs in its own process
             because PyVista and Qt cannot share the Tkinter main loop
scripts/     standalone utilities, not imported by the application
data/        corpus/, eeg_recordings/ (44 sessions), vector_db/ (FAISS cache)
outputs/     frames/, neurovascular/, brain_videos/, metabolic_videos/
research/    figures and spreadsheets for the manuscript
```

Every runtime path is anchored to a `PROJECT_ROOT` derived from `__file__`, so
the working directory does not matter.

## EEG data contract

Raw `DLR_*.txt` files and the generated `.tsv` share one layout: 32
whitespace-separated columns at **256 Hz**. Column 0 is the time index and is
dropped; columns 1–16 are the electrodes, in this order:

```
Fp1  Fp2  F3  F4  T5  T6  O1  O2  F7  F8  C3  C4  T3  T4  P3  P4
```

`data/eeg_recordings/DLR_<n>_1.txt` holds one session per index: 0–38 are the
ARPAbet phonemes and 39–43 are the microgap segments. The recordings come from a
single participant in a healthy-control study, published separately as
[LaRocco et al. (2023)](https://doi.org/10.3389/fninf.2023.1306277).

## Reproducing the published results

All results in the manuscript were produced by release
[**v1.0.0**](https://github.com/neurotechclubosu/thinktank-clearmind/releases/tag/v1.0.0).
Check that tag out rather than `main` if you are verifying the paper -- `main`
has since received pipeline corrections that change computed output.

```bash
git checkout v1.0.0
```

## Known limitations

These are stated in the manuscript and repeated here so anyone reading the code
encounters them directly.

- **The gamma band is unfiltered.** No notch filter is applied, and at 8 Hz
  frequency resolution the bins centred at 56 and 64 Hz straddle 60 Hz mains.
  Line noise may contribute to the gamma power that drives the whole metabolic
  chain.
- **The spectral estimate is coarse.** A 32-sample window at 256 Hz gives 8 Hz
  resolution, so the delta and theta bands are not resolvable. Only gamma
  informs the published results.
- **The eight neurovascular variables are not independent.** Each is a
  deterministic function of one scalar, the normalized gamma index, and four are
  exact affine transformations of it. They are eight views of one comparison,
  not eight independent tests.
- **`n = 16` is electrodes, not subjects.** The channels are spatially
  correlated and come from a single synthesized recording, so the statistics
  measure consistency across the montage rather than an effect across people.
- **The repetition axis is degenerate at v1.0.0.** The spectral routine indexes
  trials by key but analyses the same leading samples for each, so that
  dimension carries no independent data.

The last point, along with an inverted distance weighting in
`EEG_DistanceFunc` and a trial off-by-one in `EEG_normalizedGamma_CMRO2`, was
fixed in [PR #2](https://github.com/neurotechclubosu/thinktank-clearmind/pull/2)
and is merged into `main`.

**`main` therefore no longer reproduces the published numbers.** Those
corrections change trial segmentation, band-power assignment and the
interpolation weighting. Regenerate `outputs/neurovascular/*.json`, the
statistical tables and the supplementary figures before quoting any result
produced from `main`.

## Citation

> Rajagopal, G., Chaudhari, A., LaRocco, J., Xue, J., Zachariah, E., Varanasi,
> S., & Rothman, D. Imagined speech reconstruction with 3D neural metabolism and
> large language model integration. *bioRxiv*.
> https://doi.org/10.1101/2025.07.30.667805

The manuscript is under peer review; this README will be updated on publication.

## Acknowledgements

Ohio State University Neurotech Club. The authors thank Prof. David Tomasko and
Ron Vlcek.
