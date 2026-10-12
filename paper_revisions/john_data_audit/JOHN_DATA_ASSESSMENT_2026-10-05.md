# Assessment of John's newly delivered EEG

**Update, 6 October:** [John's author clarification](AUTHOR_CLARIFICATIONS_2026-10-06.md) confirms timing, marker semantics, session suffixes, published electrode order and dummy-file status. Its derivative complete-session manifest selects 13 runs/2,860 recalls and flags the prior source-space channel assignment for regeneration. The 5 October text below remains the original audit, including what was unresolved at that time.

5 October 2026. Descriptive audit of all four archives and supplied protocol attachments. No classifier fitting, new biological hypothesis test, manuscript replacement, public deposit, or collaborator message occurred in this audit.

**The delivery substantially expands the recorded phoneme bank and enables independent-recording tests. It does not provide the continuous imagined sentences or independent metabolic measurements required to validate the paper's stronger biological claims.** A new analysis should have an explicit question and fixed evaluation plan, rather than serve as another search for a positive result after FEIS and Kara.

## Downloads and contents

Original archives, attachments, extracted recordings and detailed file/event receipts are at `D:/ResearchData/thinktank-clearmind/John/2026-10-05`. Archive sizes sum to 386,965,133 bytes. Archive integrity tests passed and local SHA-256 receipts match the download manifest. Extraction produced 705 text files totaling 1,451,302,467 bytes, plus the separately preserved attachments.

| Participant alias reported by John | Recording files | Stimulus indices | File suffixes | Paired task intervals | Distinct full task signals after exact duplicate accounting |
|---|---:|---:|---|---:|---:|
| GTP000 / DLR (John, per supplied message) | 308 | 0–43 | 1–7 | 1,540 | 1,535 |
| GTP003 | 132 | 0–43 | 1–3 | 655 | 650 |
| GTP004 | 132 | 0–43 | 1–3 | 655 | 655 |
| GTP007 (files use GT007) | 132 | 0–43 | 1–3 | 660 | 660 |
| Total | 704 | 44 per alias | — | 3,510 | 3,500 |

These are observations of the delivered files. There are four reported participants, not 3,500 independent participants. Repetitions share participant, acquisition block, cue and instrumentation. A file generally contains five marked recalls; its suffix identifies a block, not a single recall. The extra `mapo.txt` remains unassigned and outside these counts.

The new `DLR_*_1.txt` files match all 44 original repository recordings after LF/CRLF normalization. The delivery contains six additional DLR blocks and three other reported participants; it does not make the old bank independent of this new dataset. John’s supplied message now supports the DLR/GTP000 identity, which the earlier repository hash comparison alone could not establish.

## Recording structure and timing

All 704 conventionally named files parse as finite numeric matrices with 32 columns, including the two truncated files discussed below. The plausible BrainFlow layout is a packet counter at Python column 0, 16 candidate EEG channels at 1–16, auxiliary columns at 17–29, timestamps at 30, and event markers at 31. Numerical layout does not identify the physical montage, reference, gain or verified units.

The acquisition script uses `CYTON_DAISY_WIFI_BOARD`, sends `~6`, and inserts each marker 1–5 both before and after `time.sleep(2)`. The official [Cyton command documentation](https://docs.openbci.com/Cyton/CytonSDK/) identifies sample-rate command 6 as 250 Hz. Across aliases, median timestamp durations of paired intervals are 1.997–2.007 seconds and median row-count/time ratios are about 249.7 samples/s. Thus the recorded endpoints, script and original protocol support **two-second instructed recall intervals at nominal 250 Hz**. John's recollection of a one-second recall likely needs clarification; one-second analysis windows should not be relabeled as the acquisition duration.

An endpoint convention uses samples `[start, end)`; these intervals range from 475 to 531 samples. Rates for individual short intervals range more widely than the medians. Timestamp delivery can be irregular, so this is not a direct precision calibration of the ADC clock. All named files have strictly increasing timestamps. Three paired intervals contain a nonunit packet-counter transition: `GTP003_35_1` marker 4, `GT007_18_1` marker 4 and `GT007_21_1` marker 2. More nonunit transitions occur outside paired intervals. Review them separately; do not equate every transition with confirmed packet loss.

Timestamp dates place John's recordings on two UTC dates (21 and 23 March 2023), with suffixes 1–4 on the first and 5–7 on the second. The other aliases each appear on one UTC date. This offers a possible different-day holdout for John after metadata confirmation. It does not establish that every suffix denotes a separate visit or session, and numeric suffixes alone are unsafe chronology: duplicate files create overlapping block time ranges.

The source paper reports a two-second cued imagined-phoneme task with one-second waits, 16 channels at 250 Hz, and a larger 16-participant cohort. Its demographic summary and ethics statement apply to that cohort; they do not by themselves establish the demographics or secondary-use permissions of these particular four aliases. [LaRocco et al., 2023](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1306277/full).

## Issues that must enter the analysis exclusions and provenance record

1. **Truncated recordings.** `GTP003_18_3.txt` has nine rows and no events; `GTP004_11_2.txt` has thirteen rows and no events. Their five expected recalls cannot be recovered by treating the files as ordinary data.
2. **Duplicate DLR block.** `DLR_27_2.txt` and `DLR_27_3.txt` are identical. All five task intervals duplicate each other. Keep one canonical copy, with a provenance link to the other.
3. **Duplicate GTP003 task intervals.** `GTP003_2_2.txt` and `GTP003_2_3.txt` have identical timestamps and markers. Their five complete task intervals are identical. The files differ in just one numeric cell in column 1 outside those task intervals. They remain the same underlying recording for split ownership; a whole-file hash alone would miss this duplication.
4. **Unassigned recording.** `mapo.txt` contains 3,814 rows and 32 columns. Markers 1–4 occur twice and marker 5 once. It is not identical to a named recording. Its participant, stimulus and incomplete final event are unresolved; retain it as unassigned rather than guessing a label.
5. **Stimulus deck mismatch.** The newly supplied `stimulus.pptx` has 16 slides containing object-picture stimuli, including apples, boats, cats and bowling objects. Its instructions refer to image pairs. The script expects 44 phoneme indices, a wait slide 47, and recall slides 48–91. The deck cannot satisfy those references. The previously recovered public phoneme deck has 91 slides and supplies a candidate mapping; John must identify the actual version used for these recordings.
6. **Script variant.** The supplied script matches the recovered acquisition script after line-ending normalization except for two assignments: `initials` changes from `GT007` to `QT`, and the filename suffix from `_3` to `_4`. The label list and marker logic agree. This supports the candidate index order, while leaving the exact executed per-recording configuration unconfirmed. Comments about two blocks/ten trials conflict with the loop decrement of two: the supplied code runs one five-recall file per index per execution. Do not use its comments as trial-count evidence.
7. **Old application mapping remains unsuitable evidence.** The alphabetical ARPAbet mapping in `src/gui/phonemizer_V2.py` conflicts with the script's stimulus order. The v1.27 replacement analysis already uses a recovered-slide crosswalk. The new files do not justify reverting to the application's historical mapping. Indices 39–43 correspond to NG/R/L/W/Y in that candidate deck and cannot be called recorded noise or natural pauses.

The full 0–43 candidate index table is [stimulus_index_map.csv](stimulus_index_map.csv). Vowel dialect and ARPAbet conversion require the same explicit approximations and author confirmation as before.

## Signal quality: unresolved, rather than automatically rejected

The audit computes unfiltered PSDs after per-window constant detrending, using one-second Welch segments and nominal 250 Hz. It reports the median across candidate channels for each interval, then summarizes those interval values within an alias. No screening threshold or preprocessing choice was optimized against labels.

| Alias | Median channel SD in recorded units | Median common-mode variance ratio | Median 58–62 Hz share of 1–100 Hz power | Median 30–100 Hz share |
|---|---:|---:|---:|---:|
| GTP000 | 5,120.9 | 0.718 | 6.90% | 54.42% |
| GTP003 | 1,178.4 | 0.628 | 8.75% | 54.41% |
| GTP004 | 619.4 | 0.230 | 8.68% | 56.35% |
| GTP007 | 3,480.0 | 0.730 | 8.37% | 58.77% |

The common-mode ratio is variance of the channel-mean demeaned signal divided by mean variance across channels. It diagnoses shared signal and does not isolate its cause. A five-Hz neighborhood power fraction is not proof of a narrow line artifact. No paired task contains an exactly flat candidate EEG channel. Large raw offsets/amplitudes, shared fluctuations, and substantial high-frequency power require verification of scaling, reference, gain and artifacts before interpreting gamma. These measurements do not certify neural quality or justify a physiological conversion.

[Fixed example traces and spectra](raw_examples.pdf) use index 0, block 1, marker 1 for each alias; they were chosen by filename before visual inspection and are not representative performance claims. Inspection shows pronounced periodic waveforms/pulses and repeated spectral peaks near harmonics of roughly 10 Hz in these examples. That pattern warrants checking equipment, reference and acquisition interference; its cause is not established here. A notch at 60 Hz alone would not resolve a broad harmonic pattern. The label remains provisional. Auxiliary columns have not been established as separately recorded eye, facial-muscle, microphone or physiological signals. Request those channels/logs if they exist.

## What this can add to the paper

**Recommended question:** Does composition from labeled cued-phoneme EEG preserve reproducible phoneme-dependent structure across independent recordings, beyond equally sized incorrect-label and recorded-wait controls?

This is a testable component question about recorded EEG and composition. A positive result would justify using the bank as an input to a controlled simulation, while leaving continuous speech, cue confounding and metabolism as separate validation problems. The comparison should establish genuine real-to-real repeatability first. A composite with a known recipe cannot serve as an independent biological reference for that recipe.

For reviewer 3's consistency request, additional real blocks can strengthen comparisons of task spectra, channel covariance and time-frequency features using held-out source recordings. A source-space comparison is conditional on verified montage/reference and a justified common inverse model; sparse template-head estimates should remain diagnostics. Comparing model outputs can describe propagation through assumptions, but cannot establish measured metabolic consistency.

For reviewer 4, independent recordings and four participants address part of the replication problem, provided participant/block ownership is respected. They do not establish that inserted waits recreate natural sentence pauses. The eight model variables remain linked to one gamma-derived driver. New data cannot turn them into independent outcomes or calibrate the imposed gamma-to-CMRO2 mapping.

## Fixed next analysis, after essential metadata confirmation

1. **Resolve labels and acquisition first.** Verify the executed phoneme deck/order, the physical column-to-electrode order, gain, units, reference and timing. Confirm the suffix/date-to-session map and identify duplicate/truncated files. Keep all originals; exclude or group copies in a separate analysis manifest.
2. **Freeze the question and splits before fitting.** Hold out entire recordings/acquisition blocks. Keep five recalls from one file together; group both copies of a duplicated source together. For John, use different recording days when justified by the confirmed log. For the other aliases, call a same-day block holdout what it is. Use subject-equal summaries so John's seven blocks do not dominate the three-block participants.
3. **Establish repeatability and specificity limits.** Compare same-label versus wrong-label real intervals across independent blocks, using fixed spectral/covariance features and blocked permutations that respect file-level labeling. Assess acquisition time/order, baseline/wait and cue-related alternatives. Require a stable real-data baseline before any augmentation comparison. Visual phoneme cues remain confounded with instructed imagination unless matched cue-only recordings are available.
4. **Then test composition utility, only if justified.** Compare the specified composition method against real-only, equal-size resampling and wrong-label/other augmentation controls on genuinely held-out real EEG. Give all conditions the same training-data and tuning budget. Fit scaling/calibration/preprocessing on training data only. Record every outcome, including failure. No test trial, duplicated donor or test-derived normalization may influence training or construction.
5. **Rewrite around the actual result.** Four participants support a small exploratory study, with wide population uncertainty. Even a positive phoneme result does not establish sentence-level reconstruction. Preserve the previous FEIS/Kara failures and identify this dataset's distinct motivation rather than recasting it as confirmatory evidence discovered after success.

The composition utility analysis is a proposal, not a reported result or formal preregistration. Specify the endpoint, effect threshold, uncertainty target, model budget, exclusions and stopping rule before running it. If the real-data feasibility gate fails, stop the composition/augmentation branch under that protocol. The existing failed FEIS/Kara gates remain unchanged.

## What still requires different evidence

- **Biological sentence composition:** matched continuous imagined sequences from the same participants, with the same units in different orders, independent repetitions, verified event timing and artifact/cue controls. The delivered files contain repetitions of one indexed unit per file; concatenating them does not create an independently recorded sentence.
- **Natural pause realism:** genuinely recorded pauses within the corresponding continuous task. Instructed waits between isolated cued recalls provide a limited control condition, not direct natural-pause ground truth.
- **Metabolism/Rothman interpretation:** an independently justified EEG-to-CMRO2 calibration and independent physiological measurements. The current `model.py` explicitly implements an illustrative transform with imposed scaling. More scalp EEG alone cannot calibrate oxygen consumption, CBF, pH, gases or lactate.
- **Population claims:** an adequately designed cohort and precision/power rationale. More recalls from four people improve within-person estimation; they do not enlarge biological N to thousands.

## Information to get from John

This is a checklist for the author team, not a message that has been sent.

- Please send the **actual 44-phoneme deck** used to collect these files and confirm the index order. The attached deck contains object pictures and only 16 slides.
- What do suffixes 1–7 and 1–3 mean: runs, sessions, days, or copies? Can you provide the acquisition log and explain the duplicated and almost empty files? What is `mapo.txt`?
- Please confirm the two-second recall/one-second wait schedule, board/firmware/BrainFlow versions, **electrode order for columns 1–16**, physical units, gain, reference/ground and any processing already applied.
- Which four pseudonyms are included, and what are their relevant cohort demographics? Please provide the subset summary; the 16-person paper's mean age/gender breakdown cannot simply be copied into a four-person analysis.
- Please confirm these particular recordings' linkage to the source study, consent/reuse/public-sharing permissions and the appropriate ethics statement. The original paper's reported IRB/consent establishes its own study context, not automatic authorization for every later release.
- Are there recordings of actual imagined words or continuous sequences, cue-only sessions, eye/facial-muscle channels, audio or task logs in the original dataset? If yes, send those plus their labels/timing. They would change what biological questions can be answered.

## Audit artifacts and reproduction

On D: `download_manifest.json`, `extraction_manifest.json`, `metadata/`, `extracted/`, and `audit/{recordings.csv,events.csv,exceptions.json,duplicates.json,extra_files.json,summary.json}`. Archive/file hashes are local provenance receipts, not publisher-supplied authentication.

In this folder: `summary.json`, `stimulus_provenance.json`, `stimulus_index_map.csv`, `raw_examples.png`, `raw_examples.pdf`, and `verification.json`. The audit implementations are `research/john_data_audit/audit.py` and `verify_and_plot.py`.

```powershell
# Extraction is already complete; prepare refuses to overwrite existing data.
python research/john_data_audit/audit.py audit
python research/john_data_audit/verify_and_plot.py
```

These commands rewrite local descriptive audit outputs, not the archives or recordings. The verifier checks extraction hashes, counts, index coverage, original-data correspondence, and fixed example selection against the saved event receipts. Full signal data stays on D:. No manuscript claims have been upgraded automatically.
