# Corrections log — v1.1.0 revision

Software engineer's log of the fixes and re-analyses requested by the editor's
checklist (`thinkTankSteps2`) and Reviewers #3/#4 (`thinkTankComments2`).
Every number below is reproduced by `python scripts/reproduce_all.py` and is
written to `results/results.json` / `results/results_summary.md`. Code
references are to this release; diffs against v1.0.0 are in git history once
this tree is committed and tagged (see "Release tagging" at the end).

**Phoneme-lexicon caveat, read first.** CMUdict and `g2p_en` could not be
installed in the sandbox used to prepare these numbers (no network access).
The three trial sentences were transcribed by hand into ARPAbet
(`data/fixtures/trial_phonemes.json`, `_meta` field) for this run. All numbers
below **must be regenerated** with `python scripts/reproduce_all.py
--phoneme-source cmudict` in an environment with `nltk`+`cmudict`+`g2p_en`
before the manuscript is finalized. The pipeline, statistics and consistency
code are independent of this caveat and are unaffected by which lexicon is
used; only the specific numbers would shift.

---

## C-1 / C-2. Repeated samples and trial-indexing defect

**File:** `src/eeg/EEG_Implement_Welch.py`.

v1.0.0 built the trial loop as `for Trials in range(1, N+1)` while slicing the
signal with `EEG_use[0][Timestep*TestDuration*(Trials-1) : ...]` before the
loop body reassigned `Trials`, so — combined with a stale-index bug in the
enclosing GUI call — every reported "trial" was the Welch spectrum of the
**same first 1024 samples** of the composite recording, and trial indices ran
1..N-1 instead of 1..N (a recording with exactly one trial's worth of data
produced zero trials and crashed downstream code with an unhelpful
`KeyError`). Two consequences follow directly:

1. Every one of the "16 electrodes x N trials" numbers behind a table cell was
   computed from **one 4-second window**, however long the sentence actually
   was.
2. The last partial trial's worth of the composite was silently dropped
   (this part was correct) but so, effectively, was every trial after the
   first.

**Fix.** `EEG_Implement_Welch` now computes `n_trials_available =
len(signal)//1024` once, raises `ValueError` with the recording length in the
message if it is `< 1`, and loops `for Trials in range(1, n_trials_available +
1)` slicing `EEG_use[0][1024*(Trials-1) : 1024*Trials]` — i.e. Welch is run
once per non-overlapping 4-s trial, using the samples that belong to that
trial. Regression: `tests/test_pipeline_equivalence.py::
test_array_pipeline_matches_legacy_file_chain` proves that the corrected
file-based chain (`EEG_Implement_Welch` -> `plot_normalized_gamma_across_channels`
-> `calculate_neurovascular_variables`) and the new array-level
`src/eeg/pipeline.analyse_composite` agree to machine precision (max relative
difference 5.4e-16) on a real composite, and `test_welch_raises_on_short_recording`
checks the new guard.

**Quantified impact** (`results.json["legacy_impact"]`, CMRO2, montage mean,
R=100 seeds; "effect" = microgap minus no-microgap): under the v1.0.0 defect
(which also only ever saw the first 4 s of the *no-microgap* composite, since
that recording is even shorter), trial 1 showed effect +0.007 (P(effect>0)=0.24),
trial 2 +0.096 (P=0.73), trial 3 -0.13 (P=0.09) — noisy, and, because they came
from one repeated window compared across seeds, dominated by which single
gamma sample the RNG happened to place first, not by an aggregate exposure to
the sentence. The corrected pipeline's per-sentence effect sizes are reported
in the statistical-unit section below.

## C-3. Distance-weighting defect

**File:** `src/eeg/EEG_DistanceFunc.py`, used only by the 3-D rendering path
(`EEG_Plotting.py`) to interpolate a value onto a cosmetic "virtual node"
between electrodes; it does not touch any number reported in a table.

v1.0.0 weighted neighbour `i` by `distance_i / sum(distances)` — the
**farther** a neighbour, the **larger** its weight, the opposite of
inverse-distance weighting, and inconsistent with the Methods text, which
always described inverse-distance weighting. It also computed distance as an
L1 (Manhattan) sum while working from 3-D Euclidean coordinates.

**Fix.** Weights are now `(1/d_i) / sum(1/d_j)`; metric defaults to Euclidean
(`metric="manhattan"` is kept, opt-in, to exactly reproduce v1.0.0 renders if
ever needed). An exact-match case (`d_i == 0`) is handled explicitly rather
than dividing by zero. Rendering jitter now takes an explicit
`numpy.random.Generator` so plots are reproducible; `EEG_Plotting.py` gained a
`seed=` parameter threaded through its Latin-hypercube sampling and jitter
draws. Tests: `tests/test_distance_and_source.py`
(`test_nearer_point_weighted_more_than_farther_point`,
`test_exact_match_at_electrode_is_centred_on_that_value`,
`test_seeded_rng_is_reproducible`).

## C-5. Units mismatch feeding the neurovascular model

**File:** `src/eeg/EEG_NeurovascularVariables.py`. Not explicitly named in the
repository's public issue list, but discovered and fixed during this revision
because it is required to regenerate physiologically plausible numbers.

The DiNuzzo-model equations take the **fractional** CMRO2 change,
`delta' = (CMRO2 - CMRO2_0)/CMRO2_0`, as input (`dCBF/CBF_0 = n * delta'`,
n~2.28). v1.0.0's gamma-normalization stage outputs an **absolute** CMRO2
already rescaled onto [2.1, 3.1] umol/g/min (`EEG_normalizedGamma_CMRO2.py`,
unchanged, correct), and that array was passed straight into the model as
`delta'` — i.e., every window was interpreted as a 210%-310% jump in
metabolism over baseline. Symptom: CMRO2 in the *output* tables ran 6.5-8.6
umol/g/min (baseline reported as ~6.6 instead of 2.1) and CBF was
correspondingly inflated 6-8x.

**Fix.** `neurovascular_from_cmro2(values, input_mode="absolute")` now
converts `delta' = CMRO2_abs/2.1 - 1` before applying the model; `CMRO2` in
the output is consequently the same 2.1-3.1 umol/g/min value that was fed in,
and OEF is pinned to the literature baseline (0.42) exactly at CMRO2 = 2.1
(`tests/test_neurovascular.py::test_absolute_mode_baseline_maps_to_baseline_oef`).
`input_mode="fractional_legacy"` reproduces v1.0.0's numbers exactly
(regression test, max abs diff 1.4e-14) and exists only so the impact of this
fix can be quantified; it is never used by the application.

---

## Statistical unit of analysis (Steps2 "Redefine the statistical unit"; Comments2 Reviewer #4 point 2)

**Problem.** v1.0.0 treated the 16 electrodes of one composite as n=16
independent replicates and ran a paired t-test across them. They are one
montage from one synthesised recording: spatially correlated, and, for the
eight neurovascular outputs, deterministic functions of one shared scalar
(next section). Channel count is not a sample size.

**What is actually random in this design.** Nothing about the no-microgap
composite is random. The only stochastic element is which of 5 auxiliary
recordings and which of 2 lengths (1 s / 2 s) fills each microgap
(`src/eeg/synthesis.plan_composite`, seeded `random.Random`). The honest
statistical unit for "does inserting microgaps change the output, and how
sensitive is the answer to the particular draw?" is therefore **one microgap
realisation** (one seed), not one channel.

**New design** (`src/analysis/inference.py`): for each sentence, the
no-microgap composite is analysed once (deterministic); R=2000 seeded
microgap composites are analysed; the montage-mean effect
`D_r = mean_channels(microgap_r - no_microgap)` is computed per seed, giving
an empirical distribution over R draws. We report its mean, SD, a percentile
95% CI, `P(D_r>0)`, and an exact Monte-Carlo p-value (Phipson & Smyth 2010,
`p = (1+min(n<=0,n>=0)) / (R+1)`, which is never exactly 0 or 1, addressing the
`p=0.00` problem directly — see below).

Channels are additionally characterised, *descriptively*, as repeated
measures: `src/analysis/inference.effective_n` computes a participation-ratio
effective number of independent channels from the inter-channel correlation
of the underlying window-level series (Cheverud 1996 / Nyholt 2004
construction), which for the tested sentences came out to **n_eff ~ 1.3-2.1
of 16 nominal channels** (`results.json["mc"][*]["n_eff_channels_*"]`) —
consistent with Reviewer #4's point that the 16 channels carry far less than
16 independent observations' worth of information, because they are all
driven by the same broadband gamma-power trajectory of the composite.
`channel_paired()` is provided for any figure that still wants a
paired-channel comparison; it reports the naive paired t-test **and** a
variance-inflated version using `n_eff` in place of `n`, side by side, so a
reviewer can see exactly how much the naive number overstates precision.

**Results actually obtained** (fixture lexicon; regenerate with `cmudict`
before freezing), montage-mean CMRO2 effect of microgap insertion, R=2000:

| sentence | no-microgap | microgap mean | effect (mean +/- sd across seeds) | 95% CI | P(effect>0) | p (Monte-Carlo) |
|---|---|---|---|---|---|---|
| trial1 | 2.1234 | 2.1328 | +0.0095 +/- 0.0021 | [+0.0054, +0.0136] | 1.000 | 0.0010 |
| trial2 | 2.1249 | 2.1280 | +0.0031 +/- 0.0011 | [+0.0009, +0.0053] | 0.997 | 0.0070 |
| trial3 | 2.1325 | 2.1291 | -0.0034 +/- 0.0021 | [-0.0075, +0.0012] | 0.074 | 0.1499 |

Full tables for all eight variables are in `results/results_summary.md` and
`results.json["mc"]`. Reading: for two of the three tested sentences,
inserting microgaps produces a small but very consistent increase in
composite-level CMRO2 across draws (tight CI excluding 0); for the third
(shortest, most punctuation-dense) sentence the sign is not consistent across
draws and the interval spans 0. The manuscript's Results/Discussion
(`docs/MANUSCRIPT_CHANGES.md`) restate the finding in exactly these terms and
nowhere claim a single, universal direction of effect.

## Dependence among the eight derived neurovascular variables (Steps2; Reviewer #4 point 2)

`src/analysis/inference.dependence_report` pools all (window, channel) pairs
across the three sentences' microgap composites and computes the eigenvalues
of the 8x8 Pearson correlation matrix among CBF, OEF, pH_V, pCO2_V, pO2_cap,
CMRO2, DeltaHCO2, DeltaLAC. Result (`results.json["dependence"]`): the first
principal component explains **99.4%** of variance and the participation-ratio
effective number of independent variables is **m_eff ~ 1.01** — i.e., for
practical purposes these eight quantities carry one degree of freedom between
them, exactly as expected from their closed-form derivation from a single
CMRO2 input (four of the eight — CBF, CMRO2, DeltaHCO2, DeltaLAC — are exact
affine functions of it, `EEG_NeurovascularVariables.AFFINE_VARIABLES`). The
revised manuscript reports CMRO2 as the primary outcome and the remaining
seven as "derived, not independent, quantities provided for physiological
interpretability," a sentence added to Methods and repeated as a Table/Figure
caption note wherever more than one of the eight appears (see
`docs/MANUSCRIPT_CHANGES.md`).

## "t = 0.00, p = 0.00, d = 0.00" internally-inconsistent table rows (Reviewer #4 point 2)

Two independent causes were identified and both are fixed:

1. C-1/C-2 above: analysing the same 1024 samples for every "trial" makes
   every trial identical, so a paired t-test across trials for a fixed
   channel has SD=0, t is undefined (0/0), and downstream formatting rounded
   `nan`/tiny values to `0.00`. Fixed by C-1/C-2.
2. p-values were being formatted to 2 decimal places with no floor, so any p
   below 0.005 printed as `0.00`. `inference.summarise_mc` returns the exact
   Monte-Carlo p together with its attainable floor `p_mc_floor = 2/(R+1)`
   (0.0005 at R=2000) and the manuscript number-formatting convention (see
   `docs/MANUSCRIPT_CHANGES.md`) requires `p < {floor}` instead of `p = 0.00`
   whenever a p-value would round to zero, matching journal convention.

## Trial 2 p-value inconsistency between Abstract/Discussion and the table (Steps2; Reviewer #4 point 2)

This was a transcription drift between a Results-table regeneration and the
Abstract/Discussion text, not a computational bug — the original repository
issue list does not name it and no corresponding code defect was found. Root
cause, structural: with the v1.0.0 pipeline every "trial" was identical
(C-1/C-2), so re-running the tail of the analysis with a slightly different
`Trials` count (e.g. after an edit elsewhere in the notebook) silently changed
which cached number got pasted where, with no numerical assertion to catch
the mismatch. **Prevention, not just correction:** every number in the
revised manuscript is generated by the single script
`scripts/reproduce_all.py` into `results/results.json`, and
`docs/MANUSCRIPT_CHANGES.md` gives the exact JSON path behind every quoted
statistic, so Abstract, Discussion and Results-table numbers are copies of
the same JSON field rather than independently retyped.

## Synthetic-vs-real EEG consistency analysis (Reviewer #3 point 2)

New module `src/analysis/consistency.py`; full write-up with all numbers in
`docs/CONSISTENCY_ANALYSIS.md` (time-frequency domain, source space via a
dependency-free spherical forward model + dSPM inverse since MNE/fsaverage
were not installable here, and metabolic metrics), including a
**leave-one-out real-vs-real baseline** built from the 44 source recordings
so that the synthetic composites' similarity to real EEG can be judged
against how similar real EEG segments are to each other. Headline finding:
synthetic composites match the real-vs-real baseline on the CMRO2 spatial
profile (montage-pattern Spearman rho 0.98-0.99, inside the real-vs-real
range) but fall well outside the real-vs-real range on raw spectral shape and
instantaneous source pattern — i.e., splicing preserves the broadband/
gamma-power-driven metabolic readout much better than it preserves fine
time-frequency or spatial detail. See `docs/CONSISTENCY_ANALYSIS.md` for the
full numbers and the important caveat about what "real" means for a spliced
composite (there is no continuous real recording of the tested sentences).

## Microgap "biological realism" claims (Comments2 Reviewer #4 point 1)

No code change — a claims/writing issue. `docs/MANUSCRIPT_CHANGES.md` marks
every location where the submitted text said microgaps "enhance biological
variability" or similar, and gives the replacement wording limited to the
supported claim: microgap insertion changes the simulated EEG-derived
metabolic outputs (quantified above), and the consistency analysis (previous
section) is now the evidence offered about biological realism, honestly
reported as mixed (good for the metabolic readout, poor for raw
spectral/spatial detail).

## "fractal" quotation marks (Reviewer #3 point 1)

No code change. `docs/MANUSCRIPT_CHANGES.md` gives the corrected
Introduction sentence, replacing `'fractal'` with the same "self-similar
architecture" language already used in the Discussion, for terminological
consistency throughout the manuscript.

---

## Reproducing everything in this document

```bash
pip install -r requirements.txt         # add nltk, cmudict data, g2p_en for the real lexicon
python tests/run_all.py                 # 30 tests, no pytest required (see below)
python scripts/reproduce_all.py --R 2000 --redraw-B 200 --perm 2000 \
    --phoneme-source cmudict            # regenerate results/results.json
```

`pytest` was requested by the checklist's usual conventions but is not
installable in the sandbox used for this revision (no network egress); the
30 tests under `tests/` are plain `assert`-based and both pytest- and
no-pytest-runnable (`pytest tests/` or `python tests/run_all.py`).

## Release tagging (Steps2 "create a tagged code release")

This tree has no `.git` (it was provided as a zip export); **before
submission**, `git init && git add -A && git commit -m "v1.1.0: corrected
pipeline, redefined statistics, consistency analysis" && git tag v1.1.0` must
be run in the maintained repository, and the manuscript's Data Availability
statement, the Methods pipeline-version sentence, and this file's header
should all cite that same tag/commit hash (placeholders are marked
`<COMMIT_HASH>` in `docs/MANUSCRIPT_CHANGES.md`).
