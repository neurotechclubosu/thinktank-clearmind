# ThinkTank / ClearMind revision handoff

Prepared **12 October 2026**, America/New_York, for the colleague taking over the PLOS ONE revision. This records the manuscript work, code corrections, data searches, experiments, negative results, author clarifications, and unfinished decisions. It includes the earlier first-round repairs for context and the work following the second-round comments in detail.

This is a handoff, not a new experiment or an approved submission. It was assembled from the current local artifacts, saved verification records, Git history, and the author messages relayed in this conversation. No models were rerun to prepare it. Results below are the recorded results of the identified runs; prior checks are not represented as fresh full-data checks on 12 October.

All `C:/Personal_Coding/thinktank-clearmind/...` and `D:/ResearchData/thinktank-clearmind/...` paths refer to Aniket's workstation. They are absolute for traceability. Copy the required directories and adjust paths when moving to another machine. This file contains the main findings itself; linked evidence supplies the details needed for reproduction.

## 1. Read this first

**The most defensible immediate revision is a reproducible text-guided EEG composition and illustrative neurovascular-modeling demonstration.** The known text chooses recorded EEG segments; the resulting signal is processed through assumed transformations. We have not demonstrated decoding of unknown speech, biological sentence reconstruction, more realistic neural variability, a reliable augmentation benefit, or measured metabolism.

The work produced useful corrections and a complete local replacement manuscript/response package. It also produced negative or inconclusive external-data results. Those are different kinds of outcome:

| Work | What worked | What was not established |
|---|---|---|
| Repository repairs | Cleaner layout, reproducibility fixes, corrected legacy trial/band/interpolation behavior | Corrected code alone does not validate the historical manuscript numbers |
| Round-two replacement analysis | Auditable event-based assembly, seeded simulation, controls, dependent-output interpretation, new tables and figures | Biological sentence realism or calibrated neurovascular activity |
| FEIS augmentation | Training/test isolation, matched augmentation budgets and complete receipts | A reliable splicing advantage over feature mixup |
| Stronger FEIS baselines | Fixed spatial/temporal models and quality/order diagnostics | A useful imagined-phoneme decoding baseline |
| Kara One | Verified native labels/timing, a fresh 13-person four-word evaluation | A robust decoding effect satisfying the fixed feasibility gate |
| John's expanded delivery | Four archives acquired; data/markers/duplicates audited; independent sessions evaluated | Reliable 44-index alignment with the two fixed baselines, continuous sentences, or metabolism |

**Current manuscript starting point:** [Manuscript v1.27 author-review draft](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Manuscript_v1.27_author_review.docx), with its [Markdown source](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Manuscript_v1.27_author_review.md) and [proposed reviewer response](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Response_round2_author_review.md).

**Important later correction:** those drafts were generated before John confirmed the physical electrode order. Fourteen of sixteen inherited source-space assignments differ from that order. The affected spatial calculations, figures, table cells and prose still need regeneration. The later FEIS/Kara/John-data experiments were not automatically inserted into v1.27. Its participant/protocol/declaration placeholders also need updating. Do not submit that draft or the older ZIP unchanged.

For a quick pass, read Sections **4** (reviewer requirements), **6** (manuscript analysis), **8–12** (experiments and failures), **14** (John's scope clarification) and **17–19** (next steps and transfer/reproduction). The rest explains why the older approaches were replaced.

## 2. Which version is which

| Artifact | Role and current status |
|---|---|
| [ISR.pdf](C:/Personal_Coding/thinktank-clearmind/ISR.pdf) | Original paper supplied for review; historical evidence, not the replacement manuscript |
| [First reviewer notes](<C:/Personal_Coding/thinktank-clearmind/Imagined Speech Reconstruction peer review notes (1).pdf>) | Initial citation, design, software and presentation requests |
| [John's v1.23](C:/Personal_Coding/thinktank-clearmind/thinkTankPaper.v.1.23.docx) | Earlier colleague revision supplied by the user |
| [Local v1.24](C:/Personal_Coding/thinktank-clearmind/thinkTankPaper.v.1.24.docx) | Earlier local revision; still contains paired-channel inference and older acquisition assumptions. Superseded as the proposed numerical/interpretive basis |
| [Round-two comments](C:/Personal_Coding/thinktank-clearmind/paper_revisions/thinkTankComments2.doc), [extracted text](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_work/thinkTankComments2.txt) | Actual second-round reviewer requirements |
| [John's supplied v1.26](C:/Personal_Coding/thinktank-clearmind/paper_revisions/thinkTankPaper.v1.26.doc), [reply](C:/Personal_Coding/thinktank-clearmind/paper_revisions/thinkTankReply2.doc) | Supplied second-round working material, preserved unchanged |
| [Supplied v1.26 code](<C:/Personal_Coding/thinktank-clearmind/paper_revisions/thinktank-clearmind-v1.26 (2)/thinktank-clearmind-v1.26/docs/CORRECTIONS.md>) | Historical corrected implementation and documentation reviewed during the audit. Some inference and interpretation in it were rejected; do not treat it as final evidence |
| [round2_revision](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/README.md) | Local replacement analysis, inputs, outputs, figures and four author-review documents; working label `round2-v1.27-working` |
| [Author-review ZIP](C:/Personal_Coding/thinktank-clearmind/paper_revisions/Round2_revision_author_review.zip) | Snapshot of the earlier local replacement package. It predates later metadata/analysis updates; it is not a current submission-ready bundle |
| FEIS, Kara and John analysis reports below | Separate exploratory studies with their own protocols, datasets and verification. They do not silently replace the manuscript's original donor-bank experiment |

The user reported that John had already sent his prior revision before our local changes. Therefore **do not assume the journal has our replacement files**. Ask the corresponding author for the exact manuscript and response currently on file, the editor's decision letter, and the current deadline. Prepare any marked copy against that exact submitted version.

### Repository state at handoff

- Local branch: `main`; HEAD: `68e15e4` (full README commit).
- Earlier repair commits: `81244e21b8dad2feb94051c99a0a1375bb836b42` and `785b9951f4ed30ae35737cd42030257f1bd3628f`, dated 31 August 2026 in local Git history.
- Local history contains merge `a371c71` for [PR #2](https://github.com/neurotechclubosu/thinktank-clearmind/pull/2).
- Local tag `v1.0.0` resolves to `81244e21b8dad2feb94051c99a0a1375bb836b42`. It preserves historical computational behavior; it is not the later event-based replacement pipeline.
- `paper_revisions/`, `research/feis_validation/`, `research/john_data_audit/`, `research/john_phoneme_analysis/` and `AGENTS.md` are untracked in the current local status. **A Git clone alone will not deliver this work.** Preserve and transfer these files separately before cleaning or moving the checkout.
- The remote is `https://github.com/neurotechclubosu/thinktank-clearmind.git`. No fresh remote-state check or GitHub write was performed for this handoff.
- `README.md`, `AGENTS.md` and `CLAUDE.md` contain historical assumptions such as 256 Hz for John's bank, the old channel order and auxiliary-noise labels. They are not the authority for the audited recording metadata. Some known-bug descriptions are also stale after earlier repairs.

For future GitHub writes to this repository, the workspace account policy requires `anik1617`; verify both CLI and Git credential identity first. Do not publish human EEG or manuscript/reviewer material merely because it is locally available.

## 3. Work sequence and why the direction changed

| Stage | Work performed | Decision/result |
|---|---|---|
| Earlier first-round response | Reviewed original paper, reviewer notes and v1.23; reorganized repository; revised documentation and an earlier manuscript draft | Software clarity/reproducibility improved. Historical statistical and biological claims were not fully resolved |
| Legacy numerical fixes | Corrected repeated-trial samples, segment/trial mismatch, band boundaries, normalization trial range and inverted distance weighting; fix branch/PR merged in local history | Outputs changed, making complete numerical regeneration necessary |
| Second-round audit, 3 October | Read actual comments, John's v1.26/response and supplied code; traced recordings back to upstream acquisition/stimulus material | Found deeper event, sampling, mapping, units and inference issues |
| Replacement manuscript analysis, 3 October | Rebuilt event-based assembly and simulations; generated tables, figures and four editable documents | A narrower, auditable working package; author verification still needed |
| Significance/data search, 3 October | Separated software demonstration from biological/utility questions; verified six public dataset candidates | Selected FEIS first, then a separately motivated Kara evaluation |
| First FEIS experiment, 3 October | Tested adapted same-phoneme augmentation with matched controls | No convincing augmentation advantage; preserved negative result and timing sensitivity |
| Bounded FEIS/Kara follow-ups, 3 October New York / 4 October UTC | Tried two stronger model families under fixed budgets; audited acquisition and nuisance controls | Both datasets failed their signal gates; no further splicing experiment followed |
| John's delivery, 5–6 October | Downloaded four archives to D:, audited all files, recovered session/marker information and exclusions; incorporated author clarifications | More independent sessions became available, but no full-word/sentence or physiological reference |
| John's independent-session evaluation, 6 October | Fixed two-family task/wait/timing comparisons, sensitivities and a John different-day diagnostic | Both task families remained near 44-class chance; no augmentation follow-on |
| John's scope clarification, 7 October | Compared his substitution/wait suggestion with the reviewers' actual requests | Refocused immediate revision on the existing computational demonstration rather than requiring a new biological study |

The publication question and the stronger research question were kept separate at the end. A failed decoding gate limits a biological or augmentation claim; it does not prevent executing a known-text lookup. Conversely, functioning lookup software does not demonstrate biological reconstruction.

## 4. What the reviewers actually require

The first notes requested better references/software versions, precise language about cognition and consciousness, clarification of operators versus research participants, justified stress handling, measured support for smoothness claims, explanation of covariance/display choices, variability across gap realizations, the LLM's role, a stable release, cleaner repository structure, and readable consistent figures/tables. The earlier revision addressed parts of this, but retained problematic significance and biological wording.

The second round is the immediate revision target:

| Comment | Required response | Work done | Remaining issue |
|---|---|---|---|
| R3.1: “fractal” | Define/justify the geometry or remove the claim | Unsupported fractal/self-similar language removed from replacement draft | Keep terminology consistent everywhere |
| R3.2: consistency in time-frequency, source space and metabolic metrics | Add explicit comparisons with real EEG | Descriptive comparisons with held-out within-session task trials; spectra, model maps and channel-index differences generated | Source assignment correction pending; no matched continuous sentence or measured metabolism. Explain those limits rather than claiming equivalence |
| R4.1: biological interpretation | Output changes do not demonstrate biological realism; clarify text-to-EEG/model simulation | Title/abstract/discussion/conclusion narrowed; recorded waits and duration control introduced | Final author approval and consistency across all materials; unsupported main claims cannot be rescued by a limitations paragraph alone |
| R4.2: statistical validity | Correlated channels are not biological N; dependent modeled outputs are not independent evidence; inconsistent statistics must be verified | Paired-channel significance tables and invalid simulation-tail p-values withdrawn; descriptive simulation summaries generated directly from results | Keep simulation intervals distinct from participant intervals; verify final numbers after any changed processing |
| R4.3: corrected pipeline and one reproducible version | Regenerate affected calculations, figures and tables; align manuscript/code/release | Local replacement calculations and document generation completed with saved inputs/checks | Later electrode correction, author verification, approved stable deposit and final marked copy still pending |

Evidence: [reviewer coverage](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/reviewer_coverage.csv), [proposed response](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Response_round2_author_review.md), [change/evidence record](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Changes_and_evidence.md).

The reviewers did **not** require a successful new neural decoder as a condition of describing this software accurately. Reviewer 4 explicitly offers narrower simulated-output conclusions. The paper still needs a worthwhile, clearly stated contribution; narrowing claims does not guarantee acceptance.

## 5. Earlier software repairs and deeper audit findings

### Repairs that were implemented

- Moved the flat source tree into `src/eeg/`, `src/gui/`, `src/viz/`, with `scripts/`, `data/`, `outputs/`, `research/` and `assets/`; rewrote imports and anchored runtime paths to the repository root. Tracked media was moved rather than deliberately untracked.
- Removed obsolete bytecode and a dead path to a differently named checkout; removed the unused `playsound` import; made MNE subject-path resolution configurable instead of hardcoded to another machine.
- Corrected the legacy Welch routine to use each trial's own samples, derive segment size consistently, and avoid double assignment at adjacent band boundaries.
- Corrected 1-based trial iteration in gamma normalization.
- Corrected inverse-distance weighting and exact electrode-hit handling.
- Added documentation, a release reference and a substantial README. The later metadata audit supersedes parts of those documents.

### Findings that required replacement analysis, not just those fixes

| Finding | Consequence |
|---|---|
| In all 44 original files, the first packet-counter-zero selection finished before the first marked task onset | The old assembly anchor did not select a verified instructed-imagination interval |
| Acquisition script and timestamp diagnostics support nominal **250 Hz**, not 256 Hz, for this bank | Sample/time and frequency interpretation had to change. FEIS remains a separate native 256-Hz dataset |
| Alphabetical ARPAbet application mapping conflicts with acquisition-slide order | Recovered a candidate crosswalk; did not assume file identity from alphabetical labels |
| Candidate indices 39–43 are NG/R/L/W/Y stimuli, not five verified noise recordings | Replaced the gap pool with recorded instructed-wait intervals |
| Earlier synthesis behavior and description disagreed about final-only versus all phonemes and token handling | Froze pronunciations and shared tokenization; made all-phoneme primary/final-only sensitivity explicit |
| Absolute CMRO2 was used where the model expected fractional change | Used `d = C/2.1 - 1`; checked dimensional CBF conversion |
| Old channel-based tests and treatment-ensemble tail counts did not establish biological effects | Removed the inferential claims and reported conditional algorithmic variation |
| Stored signals contain large offsets, shared fluctuations and substantial high-frequency/periodic structure | No neural gamma/metabolism interpretation is justified without acquisition/quality review |

The supplied v1.26 documentation attempted to explain old zero statistics and the Trial 2 mismatch as formatting/transcription consequences. The audit could not establish the cause of every historical cell. **Do not repeat a single confident historical explanation without evidence.** The remedy was replacement traceable calculations, not guessed repairs to individual table entries.

## 6. Replacement manuscript experiment: what was run and found

This analysis uses the original 44-file bank and three frozen text responses. It is separate from the external classifier experiments below.

### Fixed construction and analysis

- Verified 44/44 original files against upstream recordings after line-ending normalization; recovered 220 paired task intervals.
- Trial 1 supplies calibration and the common source covariance; trial 2 supplies construction task/wait blocks; trials 3–5 supply a disjoint **within-session** reference. These are not new participants or independent sessions.
- Used recorded one-second task blocks and a declared CMUdict-to-stimulus crosswalk, with frozen pronunciations and explicit overrides/dialect approximations. Exact executed-deck identity still needs confirmation.
- Froze the three text responses instead of regenerating them with an LLM. The application uses retrieval/prompting, not demonstrated autobiography weight fine-tuning. The recorded responses exceed the intended 20-word target; the replacement draft does not claim that target was enforced. Earlier temperature-zero wording should not substitute for exact saved inputs.
- Inserted one- or two-second recorded waits between words. A two-second insertion combines two sampled one-second blocks; it is not a measured natural pause.
- Primary spectral summaries use nonoverlapping 125-sample windows at 250 Hz, i.e. 2-Hz bins; no window crosses a segment join.
- The primary outcome is a dimensionless normalized gamma index `u` on a fixed trial-1 calibration scale. The physiological scale is imposed, not measured.
- Ran 2,000 seeded primary insertion realizations per text, plus 500 per text for each of five alternative configurations. Total saved simulation rows: 13,500.
- Added a duration-matched control that repeats the preceding task block, plus window-order invariance checks.

### Primary conditional simulation result

| Text | No-insertion mean index | Mean insertion change | Central 95% simulation interval | MCSE |
|---|---:|---:|---:|---:|
| S1 | 0.10193 | -0.00529 | [-0.01062, +0.00110] | 0.000068 |
| S2 | 0.10251 | -0.00603 | [-0.01024, -0.00135] | 0.000051 |
| S3 | 0.09088 | -0.00354 | [-0.01032, +0.00477] | 0.000087 |

These are conditional summaries over algorithmic insertion choices. Even S2's entirely negative interval is **not** a participant-population confidence interval or evidence of reduced measured oxygen consumption.

### Already-tested construction sensitivities

| Configuration | S1 mean change | S2 | S3 | Meaning |
|---|---:|---:|---:|---|
| Primary recorded waits/fixed calibration | -0.00529 | -0.00603 | -0.00354 | Reference construction rule |
| Composite-specific minimum/maximum scale | -0.00375 | -0.00436 | -0.00194 | Normalization matters |
| Final phoneme only per word | -0.02432 | -0.02954 | -0.00240 | Assembly unit strongly matters |
| Task indices 39–43 as insertions | +0.00041 | +0.00024 | +0.00317 | Speech-source mixture can reverse direction |
| Previous application file mapping | +0.00131 | +0.00181 | -0.00038 | Wrong mapping changes the result |
| Gamma restricted to 30–45 Hz | -0.00538 | -0.00587 | -0.00337 | Direction persists here; this does not certify artifact-free gamma |

Wait-minus-duration-control means were -0.00815, -0.01199 and -0.00149. Thus changes are not explained solely by adding elapsed time; the selected signal mixture matters. The aggregate modeled means were invariant to reordering the same analyzed windows. **These endpoints cannot demonstrate linguistic order, coarticulation, transition smoothing or sentence-context physiology.** Do not rerun those same aggregate controls expecting them to become order-sensitive.

### Consistency comparisons and their limits

Generated common-grid log spectra, source-map diagnostics using one common operator, and normalized-gamma channel-profile comparisons against trials 3–5. Coarse spectral/channel agreement was high, but that can reflect the shared donor set, common channels, instrumentation and normalization. High correlation is not distributional equality or biological equivalence. The “metabolic consistency” metric compares EEG-derived proxies from the same imposed mapping, not real physiological measurements.

The source diagnostic is a simplified homogeneous sphere with 202 radial shell sources. It is not the GUI's fsaverage cortical BEM or validated anatomy. **Its inherited electrode assignment was later contradicted by John; source-map numbers, Figure 3's spatial panels and affected consistency cells must be regenerated.** Channel-averaged spectral statistics are invariant to relabeling alone; that does not excuse leaving electrode-specific presentations wrong. An acquisition/preprocessing change could also affect the spectral results.

### Deliverables and verification

Four editable author-review DOCX files, Markdown counterparts, four scientific PNG/TIFF figure sets, complete draws/summaries, source/epoch receipts, pronunciations, model expressions and figure inputs. Recorded QA: **38 numerical checks and six research-property tests passed**; all **20 rendered document pages** were inspected. These validate specific computations/layout, not biological assumptions or later-corrected montage facts.

Start with [package README](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/README.md), [verification record](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/verification_record.json), [results JSON](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/results/results.json), and [author decisions](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Author_decisions_before_resubmission.md).

## 7. Public datasets: what was investigated versus actually analyzed

The [3 October shortlist](C:/Personal_Coding/thinktank-clearmind/paper_revisions/ONLINE_DATASET_SHORTLIST_2026-10-03.md) records primary papers, pinned releases, access/license wording, sample metadata and timing checks. It was a search-time proposal; its statement that full EEG had not yet been downloaded predates the later FEIS/Kara acquisitions.

| Dataset | Work actually done | Remaining possible role and limitation |
|---|---|---|
| FEIS v1.1 | Full 21-person imagined/rest cohort acquired; augmentation and stronger baselines run | Isolated English-phoneme benchmark; no full sentences or complete matching inventory |
| Kara One | All 14 archives acquired; MM05 format pilot; 13-person four-word models run | Word-level method evaluation; no sentences, and smaller prompts do not automatically form a matching word bank |
| Chisco v1.1.2 | Literature/release metadata inspected; no full EEG analysis | Mandarin imagined sentences; would require a new language/task design and lacks a matched isolated bank. English ARPAbet lookup cannot be transferred directly |
| Thinking Out Loud v2.1.2 | Literature/metadata inspected; no classifier study | Spanish direction words and contrasting tasks/external channels; no continuous sentences/isolated phoneme bank |
| VocalMind | Literature/release information inspected; no EEG experiment | Mandarin word/sentence intracranial recordings in one person; vocabulary overlap unverified and modality/cohort differ from scalp EEG |
| Synchronous EEG–fMRI v1.0.1 | Sampled header/events/fMRI metadata audited; no full model study | Conditional EEG-to-BOLD study; some sampled event durations were unexplained and need native timing audit. BOLD alone is not a direct CMRO2 calibration |

The EEG–fMRI timing sample had 140 of 319 entries below 0.1 second, with later entries at protocol-scale durations. No blanket timing correction was made. License/README wording discrepancies and preprocessed-data restrictions are preserved in the shortlist. Do not describe the four unrun candidates as successful replication or validation. A new dataset should answer an explicit new question, not just provide another chance of obtaining significance.

## 8. FEIS experiment 1: adapted augmentation did not establish benefit

**Question:** Can same-phoneme recombination improve prediction of real held-out isolated phoneme trials beyond matched augmentation controls?

Data: 21 English participants, 16 labels, 14 channels at native 256 Hz. There are 3,312 original imagined trials. The primary initial timing screen admitted 3,293; 1,008 later real trials were held out. The merged files do not establish independent session identities. This was later-block, within-person validation, not transfer to a new person or verified new session.

Processing/ownership: per-trial fixed filtering and common reference, central four-second crops, training-only scaling and quality thresholds, and training-only augmentation donors. Seven conditions had matched synthetic budgets; three augmentation seeds were averaged within participant. Full labels, predictions and donor-parent receipts were saved.

| Training condition | Mean balanced accuracy | Participant-bootstrap 95% interval |
|---|---:|---:|
| Real only | 6.55% | [4.96%, 8.13%] |
| Duplicate real | 6.05% | [4.66%, 7.44%] |
| Scale and jitter | 6.15% | [4.86%, 7.47%] |
| Feature mixup | 6.42% | [5.46%, 7.47%] |
| Same-phoneme splice | 6.58% | [5.29%, 7.94%] |
| Random-label splice | 5.92% | [5.22%, 6.61%] |
| Shuffled blocks | 6.05% | [4.66%, 7.44%] |

Chance is **6.25%**. The prespecified splice-minus-mixup difference was **+0.165 percentage points**, interval **[-0.827, +1.191]**; the improvement criterion failed. Stricter screening gave **-0.08 points**, interval **[-0.86, +0.69]**. Shuffled blocks and exact copies produced identical scores because the chosen spectral averages were order-invariant.

### Timestamp issue already resolved as a sensitivity

An overly strict initial tolerance rejected 19 trials from participant 04 in each phase. Inspection showed six-decimal rounding below one microsecond, not established missing samples. A separate sensitivity retained them, preserving the original inputs/results and the same test identities. Splice-minus-mixup then became **0.00 points**, interval **[-0.93, +1.03]**. Do not delete those trials again solely because of the original 0.1-microsecond tolerance, and do not generalize this exception to genuine gaps.

Recorded verification recovered **3,528 scores from 168,483 saved predictions** and checked **163,260 donor receipts**, with zero donor/test overlap. Seven meaningful implementation tests passed. Repeated conditions/draws are not additional biological participants.

**Decision:** preserve the negative adapted-augmentation result. It is not a direct test of John's English sentence generator, and it supplies no decoding-benefit claim for the manuscript.

Evidence: [initial report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/REPORT.md), [initial protocol](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/PROTOCOL.md), [experiment status](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/EXPERIMENT_STATUS.md).

## 9. Stronger FEIS follow-up: useful baseline still not demonstrated

Because the original simple design was near chance, a bounded follow-up tried **filter-bank log-covariance with a regularized classifier** and a **compact temporal-spatial EEGNet-style CNN**. It was exploratory: the first FEIS test scores had already been seen. This was a fixed follow-up, not an unrestricted neural-network search.

All 21 people and 3,312 valid imagined trials were retained using the justified timestamp tolerance. Original later-block ownership remained; 1,008 real test trials. Fixed 1–45 Hz processing/central four-second crops and training-only transforms were used. Covariance regularization candidates and CNN training lengths were selected inside two chronological training folds. CNN seed 17 was primary; seeds 29/43 were stability checks. The exact architecture/budget is in the [follow-up protocol](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/FOLLOWUP_PROTOCOL.md).

| Primary imagined model | Accuracy | Participant 95% interval | Holm conditional p |
|---|---:|---:|---:|
| Covariance | 5.56% | [4.07%, 7.14%] | 1.000 |
| EEGNet-style CNN | 6.05% | [4.56%, 7.54%] | 1.000 |

Chance: **6.25%**. The fixed gate required at least 8.25% accuracy, interval lower bound above chance, and corrected conditional p≤0.05. Both failed.

Already-run diagnostics include raw spectra/reference/common signal, amplitude/flags/flatness, strict screening, mean-power LDA, nuisance features, resting-phase models, one-person listening/spoken models, CNN seeds, conditional permutations and limited training-label refits. Strict covariance was 6.00%; strict CNN 6.15%; alternate CNN seeds gave 5.06% and 6.05%. None established a useful imagined-phoneme baseline.

Two tempting findings should **not** be recycled as success:

- Rest CNN reached 7.74% [6.55%, 8.83%], but rest follows speech and is an exploratory diagnostic. Thinking did not outperform it; carryover/task effects remain possible.
- A scheduling-only modulo-16 lookup reached 10.42% in the 20 complete recordings and 12.50% in the shorter one **without EEG**. All participants shared the same task-ID mapping. These post-baseline order diagnostics expose a confound; they are not neural decoding results.

For completeness, the 1–45-Hz imagined mean-power LDA diagnostic was 6.75% [5.36%, 8.13%], and the imagined nuisance model was 7.04% [5.16%, 9.23%]. Participant-01 listening/stimulus and speaking diagnostics were near chance with the fixed models (covariance 6.25%, CNN 4.17%, mean-power/nuisance 8.33%). These one-person phases did not establish a strong positive real-data control. The labeled synthetic spatial/temporal fixtures were recoverable; that checks implementation, not source-recording quality.

Verification recovered **352 scores from 16,814 predictions**, with labels, folds, source snapshots and caches checked. The baseline gate failure stopped further FEIS splicing. Evidence: [follow-up report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/followup_v1/REPORT.md), [follow-up decision](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/FOLLOWUP_DECISION.md).

## 10. Kara One: separate word study, not a successful replication

All 14 official archives were acquired: **24,824,165,739 bytes**, about 24.82 GB of archives alone. MM05 was reserved for format/timing audit and never received a classifier score. The remaining **13 people** formed the fixed independent cohort for **gnaw, knew, pat, pot**; no switch to easier binary tasks was made after results.

Native parsing verified 62 scalp predictors plus eye channels. The channel labeled `EMG` is a task-colour sensor, not verified muscle recording; it was excluded from predictors. Five-second supplied imagination intervals were filtered separately and cropped to the central four seconds. The cohort had **636 word trials: 424 training, 212 real test**. Primary ownership was later within each class and could interleave across classes; a separate globally later holdout was also run. Neither is a verified independent-day split.

### Acquisition-only failure preserved before scoring

The initial `kara_followup_v1` acquisition gate generated no classifier scores. Native metadata needed better task-boundary and repeated-attempt alignment. For MM08, 137 delivered prompts versus 131 retained intervals were reconciled using the task sensor, identifying six omitted attempts and matching all retained labels. Duplicate delivered-log receipts were also handled. Those fixes and the parser plan were recorded **before** v2 classification. Do not treat v1 as a failed classifier or realign trials by list position when prompt attempts were omitted.

### Actual v2 outcomes

| Model/condition | Accuracy | Participant 95% interval |
|---|---:|---:|
| Primary covariance | 28.85% | [23.56%, 35.58%] |
| Primary CNN, seed 17 | 25.10% | [20.67%, 29.42%] |
| Globally later covariance | 22.82% | [14.68%, 31.12%] |
| Globally later CNN | 21.19% | [14.34%, 27.92%] |
| Eye-only control | 30.96% | [24.04%, 38.17%] |
| Eye-regressed covariance | 26.06% | [19.81%, 33.17%] |
| Eye-regressed CNN | 27.69% | [22.60%, 32.98%] |
| Strict-screen covariance | 29.17% | [23.56%, 36.06%] |
| Strict-screen CNN | 24.78% | [20.29%, 29.58%] |
| Primary CNN, seed 29 | 22.60% | [16.83%, 27.88%] |
| Primary CNN, seed 43 | 25.38% | [20.67%, 30.67%] |
| Raw-amplitude nuisance control | 23.65% | [17.88%, 29.90%] |

Chance: **25%**. Gate: at least 30%, interval lower bound above 25%, and Holm conditional p≤0.05. Primary corrected p-values were **0.184 / 0.516**; both families failed. Strict screening and alternate seeds were also retained in the report. The wide covariance interval does not rule out useful information, but this study did not establish it. Eye-only performance neither proves contamination nor establishes imagery specificity.

Recorded verification recovered **156 scores from 2,540 predictions**, checked all 14 archive receipts, 151 listed extracted files and 12 additional metadata receipts, and preserved native source hashes/fold isolation. These are local integrity receipts, not publisher-authenticated checksums. The three external-analysis test modules recorded **21 passing meaningful tests**.

**Decision:** no Kara augmentation follow-on. This is a separately motivated four-word study, not confirmation of a positive FEIS effect, full sentences, or metabolism.

Evidence: [Kara report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/kara_followup_v2/REPORT.md), [parser plan](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/KARA_PARSER_PLAN.md), [completed status](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/FOLLOWUP_STATUS.md).

## 11. John's new recordings: download, metadata and exclusions

Downloaded `gtp000.zip`, `gtp003.rar`, `gtp004.rar`, `gtp007.rar` to D:. Archive integrity tests and SHA-256 receipts passed. Total archive size: **386,965,133 bytes**. Extraction produced **705 text files / 1,451,302,467 bytes**. Of those, 704 conventionally named files cover four reported participants and all 44 indices; `mapo.txt` is separate.

| Reported alias | Delivered sessions | Paired recalls before deduplication | Selected complete sessions |
|---|---|---:|---|
| GTP000 / DLR (John, per his confirmation) | 1–7 | 1,540 | 1, 3, 4, 5, 6, 7 |
| GTP003 | 1–3 | 655 | 1, 2 |
| GTP004 | 1–3 | 655 | 1, 3 |
| GTP007, filenames GT007 | 1–3 | 660 | 1, 2, 3 |

Total paired intervals were 3,510, with **3,500 distinct full-task signals** after exact duplicate accounting. Complete-session selection retains **13 sessions / 572 files / 2,860 unique recalls**. Those counts differ for a reason: otherwise valid parts of incomplete sessions were not included in the fixed primary study. The original 44 `DLR_*_1` files match the delivery after line-ending normalization; they are not newly independent recordings.

### Confirmed author facts

- Filename suffixes denote separate sessions covering the phoneme set; five paired recalls occur within each normal file.
- Task instruction is approximately two seconds of recall and one second of waiting; paired equal-value markers identify recall start/end.
- Raw files have 32 columns; Python columns 1–16 are EEG, 30 timestamps, 31 markers; column 0 is a packet counter, not elapsed time. Nominal rate is 250 Hz. Auxiliary columns are not verified eye/muscle channels.
- John identifies himself as DLR/GTP000 and says no full words or separate eye/muscle-action data were supplied.
- The recording electrode order is **`Fp1 Fp2 F7 F3 F4 F8 T3 C3 C4 T4 T5 T6 P3 P4 O1 O2`**, based on John's direction to the Frontiers paragraph's sequence. It differs in 14 positions from the inherited sphere assignment.
- `mapo.txt` is a useless dummy, and glitch files can be excluded while preserving complete runs.

### Exclusions already resolved

- `DLR_27_2` and `DLR_27_3` are identical; timestamp peer ranges support ownership in session 3. Retain `_3`, exclude the misplaced `_2` copy from analysis.
- `GTP003_2_2` and `GTP003_2_3` have identical task intervals/timestamps, although one outside-task cell differs. Timestamp peer ranges support session 2. Retain `_2`.
- `GTP003_18_3` (nine rows) and `GTP004_11_2` (thirteen rows) contain no usable paired trials.
- `mapo.txt` is excluded as an author-identified dummy.
- Entire affected incomplete sessions are omitted from the primary manifest; no copied recording is used to fill a missing index. **No raw file was deleted.** Duplicate ownership is timestamp-supported rather than confirmed by an original acquisition log.

### Unresolved metadata/quality

The supplied `stimulus.pptx` has 16 object-picture slides. The acquisition script expects a 91-slide phoneme presentation; a recovered public 91-slide candidate supplies an operational crosswalk. John's wording that the presentation defines the phones does not establish that these two decks are the same executed version. Continue using integer-index labels where actual phonetic interpretation is uncertain.

Units, gain, physical reference/ground, preprocessing history and the source of the periodic raw pattern remain unknown. Fixed raw examples showed pronounced pulses/waveforms and peaks near harmonics of roughly 10 Hz. Shared signal and high-frequency power are substantial. Their cause was **not** diagnosed; filtering and relative features do not prove clean neural data. A few task intervals have nonunit packet-counter transitions; these were documented, not silently relabeled as confirmed packet loss.

The Frontiers paper's larger cohort demographics/ethics cannot simply be copied into this four-person subset. Exact secondary-use, consent and public-release facts still require author confirmation.

Evidence: [initial delivery audit](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_data_audit/JOHN_DATA_ASSESSMENT_2026-10-05.md), [6 October clarification](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_data_audit/AUTHOR_CLARIFICATIONS_2026-10-06.md), [complete-session manifest](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_data_audit/complete_session_recordings.csv), [channel-order comparison](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_data_audit/confirmed_channel_order.csv).

## 12. John's independent-session experiment: the fixed gate failed

The [protocol](C:/Personal_Coding/thinktank-clearmind/research/john_phoneme_analysis/PROTOCOL.md) was saved before classifier scores on this delivery. It is exploratory, not formal preregistration: original session 1, raw examples and earlier external results had already been seen.

- Train each person's session 1: **880 training recalls** in total. Test all other complete sessions: **1,980 real recalls / nine sessions**.
- Keep all five recalls from a source file on one side; verify full-task hashes and exclude duplicates. Average sessions within person and then people equally, so John's extra sessions do not dominate.
- Use central 0.5–1.5-second task windows at 250 Hz, independently filtered within each full marked task. Wait controls use a complete one-second pre-onset window without crossing the previous task end.
- Fixed features: log-relative five-band power (80 features) and scale-normalized/shrunk matrix-log covariance (136). Training-only scaling and nearest-class centroids; **no tuning or additional model search**.
- Fixed controls: waits, timing/acquisition rank, no common-average reference, omitted harmonic bins, training-only amplitude/derivative screen, file-mean prediction, and a John different-day diagnostic.
- Used 999 class-label bijections per person for a conditional alignment null, Holm across two primary families, and a descriptive 10,000-draw participant bootstrap. The null assumes label exchangeability and does not test whether activity is neural.

| Primary task family | Participant-equal accuracy | Descriptive participant interval | Conditional p | Holm p | Above chance |
|---|---:|---:|---:|---:|---:|
| Spectral | 2.08% | [0.89%, 3.24%] | 0.626 | 1.000 | 1/4 people |
| Covariance | 2.17% | [1.93%, 2.40%] | 0.607 | 1.000 | 1/4 people |

Chance: **2.27%** for 44 labels. The chosen pilot gate required ≥7.27%, at least three of four people above chance and Holm p<0.05. Both failed. Four-person bootstrap intervals are discrete/unstable and not population validation.

| Person | Spectral task mean | Covariance task mean | Held-out sessions |
|---|---:|---:|---|
| GTP000 / John | 2.18% | 2.09% | 3, 4, 5, 6, 7 |
| GTP003 | 0.45% | 2.27% | 2 |
| GTP004 | 3.64% | 1.82% | 3 |
| GTP007 | 2.05% | 2.50% | 2, 3 |

Already-run control outcomes: spectral waits **2.41%**, covariance waits **3.43%**, timing-only **2.41%**, no-CAR spectral/covariance **2.12% / 2.30%**, harmonic-omission spectral **1.87%**. The screen rejected **zero** recalls and left results unchanged; that is not quality certification. Task did not outperform waits/timing on the equal-person mean. Covariance often predicted only a few classes, including exactly one class for one of nine sessions; nominal chance can arise from constant guessing in a balanced task.

John's different-day diagnostic trained sessions 1/3/4 from 21 March and tested 5/6/7 from 23 March: spectral **2.58%**, covariance **1.97%**. This is one person's diagnostic, not a larger cohort. Other participants' sessions are same-day; do not label all comparisons cross-day.

The held-out file-mean task accuracy was 1.99% spectral / 2.16% covariance. John's different-day file-mean accuracy was 0.76% / 2.27%. Averaging recalls within a file did not rescue repeatability and does not create another biological replicate.

Verification rechecked **572 source hashes**, **2,860 distinct task hashes**, re-extracted **80 fixed raw windows**, independently replayed **21,120 predictions / 96 scores**, and recovered eight conditional null arrays. Five positive/null/scaling/filter tests passed. Three scientific PNG/PDF figure pairs were produced; all PNGs were inspected.

**Decision:** no augmentation/composition-utility experiment followed this gate failure. It limits the tested bank/method's demonstrated repeatability, not all imagined-speech methods. It does not invalidate a known-text dictionary lookup by itself.

Evidence: [full report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_phoneme_analysis/REPORT.md), [summary](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_phoneme_analysis/summary.json), [verification](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_phoneme_analysis/verification.json).

## 13. Rothman/DiNuzzo: what our use does and does not establish

The physiological framework can motivate an illustrative transformation. It does not independently validate the input we supply to it.

In the replacement implementation, normalized gamma `u` is assigned `C = 2.1 + u` on an imposed 2.1–3.1 μmol/(g·min) CMRO2 scale. Fractional change is `d = C/2.1 - 1`. This corrects the previous absolute-versus-fractional misuse; it does **not** calibrate scalp gamma to actual oxygen consumption. CBF is calculated in mL/(g·min); multiply by 100 for per-100-g units.

All eight downstream quantities are deterministic functions of one driver under fixed constants. Their correlated changes do not provide eight independent confirmations. Parameter/equation provenance is not fully established for every implemented constant. A balance or coding test cannot establish physiological accuracy.

Use the [unit/interpretation note](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/results/units_and_interpretation.md) and [executable model](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/code/model.py). Retain model-based language unless there is a separately justified calibration and independent physiological measurement. More EEG, a theoretical citation, or BOLD alone does not supply that missing calibration.

## 14. John's 7 October message and the final scope decision

John suggested using the one-second inter-repetition gaps as background/noise if needed. He described the pipeline as substitution: known text → phonemes → recorded EEG lookup → CMRO2/metabolism model. He suggested optional phoneme-voiceover/TTS assembly, but emphasized that this is a major revision rather than a new project, with uncompleted biology acknowledged as limitations/future work.

The immediate interpretation agreed with his **scope**:

- Recorded waits are already used in the replacement analysis and latest controls. Call them instructed waits, not verified pure noise or natural pauses.
- A lookup demonstration can be described even after the decoder fails. Neural specificity and augmentation utility remain unestablished.
- Do not demand a new cohort, full sentence study or TTS feature merely to finish a narrower revision.
- Do not leave unsupported realism/metabolism claims in the main narrative and rely on a limitations paragraph to cancel them.
- Numerical validity, corrected electrode assignments, appropriate statistical units, explicit model assumptions and reviewer-requested comparisons remain necessary.

No TTS/jigsaw addition, new recruitment, new physiological calibration, or new uncontrolled model search was performed following that message.

The legacy GUI already contains pyttsx3 speech output. John's proposed phoneme-voiceover assembly would be an additional feature, not evidence that text-to-speech itself is absent from the application.

## 15. Avoid repeating these routes without a new reason

| Already attempted/rejected | Why repeating it unchanged is unlikely to help |
|---|---|
| Paired electrode t/Wilcoxon tests as biological N=16 | Spatially dependent channels are not independent people; old significance claims were withdrawn |
| Counting gap seeds as human replication | More seeds reduce Monte Carlo error only |
| Tail count over insertion-treatment draws as a null p-value | Those draws do not generate the stated biological null |
| Reporting eight transformed variables as independent effects | One deterministic input drives all of them |
| Packet-counter zero as imagination onset | Already shown to precede task markers in all original files |
| Calling indices 39–43 noise | They are speech stimuli under the recovered candidate inventory |
| Claiming order/context from window-averaged summaries | Exact window-order/shuffle controls already show invariance |
| FEIS same-phoneme augmentation with the same design | Benefit over matched controls was not established, including timing/screening sensitivities |
| Expanding FEIS CNN seeds/strict screening to rescue the baseline | These bounded checks were already run and remained near chance |
| Repeating Kara v1 parsing or substituting list-position labels | Omitted prompt attempts and task-boundary issues already resolved before v2 scores |
| Calling Kara's 28.85% or eye-only 30.96% a robust neural success | Uncertainty/specificity/gates do not support that interpretation |
| Random trial splits that separate recalls from one source file | Shared acquisition/duplicates can leak across train/test |
| Repeating John's fixed centroids with the same waits/reference/harmonic checks | The recorded near-chance result and checks already exist |
| Using high source-bank correlation as sentence/metabolic validation | Common source characteristics and imposed transforms can generate similarity |
| Treating a notebook/GUI/video or passing tests as corrected manuscript evidence | Quantitative source/version/assumptions and all affected outputs still need traceability |
| Copying old README or AGENTS recording metadata | Those files still contain superseded rate/channel/mapping assumptions |

These are not bans on scientific replication or a genuinely different hypothesis. A changed method should have a clear reason, a fixed budget and independent evaluation; reusing already-seen test data for tuning must be labeled exploratory.

## 16. Possible directions: completed versus proposed

| Direction | Status | What would justify doing it |
|---|---|---|
| Finish the narrower composition/model demonstration | Recommended immediate revision route; partly completed | Correct affected outputs, establish provenance, expose controls/assumptions and state a concrete computational contribution |
| Acquisition/quality investigation | Descriptive concerns found; cause unresolved | Equipment/log/reference/gain/units review and explicit quality protocol. Could explain failures; cannot be inferred from a notch alone |
| New composition/augmentation utility study | Initial FEIS version failed; later follow-ons withheld | A credible real-data baseline plus a separately fixed equal-budget comparison on genuinely held-out real EEG |
| Biological sequence-composition study | Not run; matching data absent | Same participants record isolated units and continuous short sequences, same units in different orders, independent repetitions/days, accurate markers and cue/eye/mouth controls; order-sensitive outcomes |
| EEG-to-metabolism calibration | Not run | Independent physiological measurements and a justified calibration/validation model, reviewed by a domain expert |
| Chisco/other unrun datasets | Search only | A language/modality-specific question, acquisition audit, reuse review and new protocol; not a direct repair of John's English bank |
| Order-sensitive/transition endpoints | Existing aggregates cannot answer them | A prespecified endpoint and matched biological reference if biological smoothing is claimed |
| TTS/voiceover “jigsaw” feature | John proposed; not implemented in this revision | Optional software feature with a distinct purpose. It does not validate the EEG or physiological mapping |
| GUI usability/scalability evaluation | Not established by this work | Separate operational/user study or measured runtime/resource benchmark; avoid claiming it from visual quality alone |

The unresolved biological studies are **future evidence requirements for stronger claims**, not automatic prerequisites for an accurately narrowed software revision. Sample size should follow the effect/precision question and pilot, not an invented publication minimum. Negative results can be reported honestly, but neither a negative experiment nor a narrower title guarantees a compelling paper.

## 17. Immediate checklist for the new owner

1. **Preserve the work.** Copy untracked revision/research directories and the D: data/manifests/snapshots. Confirm which manuscript is actually on file with PLOS and the deadline. Do not clean the checkout before transferring local-only work.
2. **Agree on the claim.** Begin with the narrow text-guided composition/model framing and decide what concrete question the corrected study answers. Keep the LLM/GUI as components unless separately evaluated.
3. **Review acquisition facts with John.** Units, gain, reference/ground, preprocessing, periodic signals and exact executed deck/crosswalk remain open. The timing/session/electrode facts already confirmed need not be asked from scratch.
4. **Apply the confirmed channel order.** Correct physical assignments in the sphere pipeline and any electrode-specific presentations; regenerate affected source outputs, consistency values, figures and linked prose. Preserve the old snapshot. If processing changes, regenerate all affected results, not selected cells.
5. **Update manuscript/response consistently.** Resolve old placeholders, use final numerical sources, distinguish conditional simulation intervals from participant uncertainty, and explain independent-reference limitations. Decide explicitly whether the external negative studies belong in main text, supplement or accompanying author evidence; do not suppress relevant contradictory results or present them as positive validation.
6. **Finalize human-data and author facts.** Confirm subset demographics, participant linkage, consent/ethics/secondary use/public sharing, affiliations/contact/funding/contributions, model provenance and author verification of AI-assisted material.
7. **Freeze the approved exact package.** Tie tables, figures, code, data availability and response to one stable approved version. Update stale repository metadata/docs. Build/render the final documents and the marked copy against the real submitted file.
8. **Have authors approve submission/release.** A local bundle or a moving main branch is not the final deposit. Corresponding author handles journal communication. No new submission or public release is authorized by this handoff.

Still open at handoff: acquisition/quality and exact crosswalk; four-person demographics/ethics/reuse; spatial regeneration; model/constant applicability; finalized disclosure/declarations; final claim choice; author-approved stable release and journal-ready marked manuscript.

## 18. Storage and transfer map

| Location | Contents to preserve |
|---|---|
| `C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/` | Replacement manuscript/response, results, figures, code, frozen inputs and QA. Includes known pending montage correction |
| `C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_work/` | Verified legacy text extraction and recovered acquisition/stimulus sources |
| `C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/` | All initial/sensitivity/follow-up/Kara summaries, decisions, figures, verification and source snapshots, including failed acquisition-only v1 |
| `C:/Personal_Coding/thinktank-clearmind/research/feis_validation/` | Acquisition/parsing/model/report scripts, protocols and tests |
| `C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_data_audit/` | Metadata, exclusions/session manifests, raw example figures and receipts |
| `C:/Personal_Coding/thinktank-clearmind/research/john_data_audit/` | Delivery preparation/audit/author-manifest scripts |
| `C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_phoneme_analysis/` | Verified summaries, figures, gate decisions and report |
| `C:/Personal_Coding/thinktank-clearmind/research/john_phoneme_analysis/` | Frozen plan, extraction/evaluation, independent verifier and tests |
| `D:/ResearchData/thinktank-clearmind/FEIS/v1.1/` | Original downloads/extractions, processed arrays, full predictions/donors, source terms and manifests |
| `D:/ResearchData/thinktank-clearmind/KaraOne/` | All 14 archives, necessary extractions, delivered prompt metadata, processed EEG and full predictions |
| `D:/ResearchData/thinktank-clearmind/John/2026-10-05/` | Four original archives, attachments, raw extractions, audit receipts and analysis caches |
| `D:/ResearchData/thinktank-clearmind/John/2026-10-05/analysis/phoneme_repeatability_v1/` | Exact scored John run: protocol/code snapshots, source/epoch receipts, features, predictions, permutations/nulls and verification |

The three D: dataset roots were confirmed present while preparing this handoff. Do not redownload their full archives merely because the current Git clone lacks them. Transfer requires substantially more space than the manuscript ZIP; Kara's archives alone are about 24.82 GB, with extracted/cached files additional. Retain original data separately from derivatives, attribution/reuse terms and locally recorded hashes. Obtain author approval before distributing private recordings.

## 19. Reproduction without overwriting the evidence

**Read saved reports first.** Do not rerun every model just to learn what already happened. Use each run's frozen source snapshot, inputs and recorded environment for exact reproduction. Main GUI requirements are not a complete environment specification for the later ML studies. The old `.venv` assumption and historical agent docs should be checked rather than trusted.

### Existing implementation tests

From `C:/Personal_Coding/thinktank-clearmind`:

```powershell
python paper_revisions/round2_revision/code/test_reanalysis.py
python -m pytest research/feis_validation/test_validation.py research/feis_validation/test_followup.py research/feis_validation/test_kara.py -q
python -m unittest discover -s research/john_phoneme_analysis -p test_analysis.py -v
```

These recorded tests check computational properties, known-positive/null behavior, parsing, label ownership and train-only transforms. Passing them is not a physiological validation. They were not rerun for this documentation task.

### Manuscript replacement scripts

```powershell
python paper_revisions/round2_revision/code/reanalyse.py --draws 2000
python paper_revisions/round2_revision/code/make_figures.py
python paper_revisions/round2_revision/code/build_documents.py
```

**These write replacement outputs/documents. Preserve a complete copy first and correct the montage/protocol before producing a new final run.** They do not automatically incorporate later John/FEIS/Kara results. The README documents `--data` for relocating the original recording bank. Document generation/rendering has separate dependencies; earlier Word-native export was used because the bundled PDF renderer could not find LibreOffice. Final editable documents still require a render-and-inspect pass.

### External experiment scripts

The [research README](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/README.md) documents downloads, original benchmark, summarizers and follow-ups. Many defaults overwrite named reports/runs; use a fresh output name and preserve old source snapshots. Some verifiers/summarizers also rewrite derivative files.

```powershell
# Examples of fresh output names, not instructions to launch another search:
python -m research.feis_validation.followup --pilot-diagnostics --output paper_revisions/feis_validation/results/followup_reproduction
python -m research.feis_validation.kara_benchmark --output paper_revisions/feis_validation/results/kara_reproduction
```

Full prediction paths are derived from result-directory names on D:. FEIS/Kara support external `--root` options in the relevant parsers; review each CLI before moving storage. Do not rename a scored run and assume all manifests/cache pointers change automatically.

### John audit and fixed run

```powershell
# Existing analysis; extract/evaluate refuse overwriting a completed default run:
python research/john_phoneme_analysis/analyze.py extract
python research/john_phoneme_analysis/analyze.py evaluate
# This replays predictions and regenerates derivative reports/verification:
python research/john_phoneme_analysis/verify_report.py
```

The default John run is already complete, so the first two commands should refuse to replace it. `--run` selects a fresh derivative run; it does not change the raw input root hardcoded in the current analysis. Audit tools have their own root/output options. If adapting paths or code, preserve the original snapshot and make the change explicit; code/protocol hashes guard the fixed run. Rebuilt figures require fresh visual inspection; the verifier initially records that inspection as pending.

Do not run the John `prepare` stage again against populated extractions, blindly overwrite the author manifests, or import the GUI to run a scientific test: it creates widgets at module scope. Batch research scripts do not require LLM calls. New text generations would change the frozen input experiment rather than reproduce it.

## 20. Recommended reading order

1. This handoff, then [author decision brief](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Author_decisions_before_resubmission.md). The brief is older: apply later confirmations and the pending montage correction above.
2. [Actual round-two comments](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_work/thinkTankComments2.txt) alongside [proposed response](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Response_round2_author_review.md).
3. [v1.27 manuscript](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Manuscript_v1.27_author_review.md), [changes/evidence](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/documents/Changes_and_evidence.md), [units note](C:/Personal_Coding/thinktank-clearmind/paper_revisions/round2_revision/results/units_and_interpretation.md).
4. [FEIS initial result](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/REPORT.md) and [FEIS/Kara follow-up decision](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/FOLLOWUP_DECISION.md).
5. [John's metadata update](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_data_audit/AUTHOR_CLARIFICATIONS_2026-10-06.md) and [independent-session report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/john_phoneme_analysis/REPORT.md).
6. [Research viability plan](C:/Personal_Coding/thinktank-clearmind/paper_revisions/RESEARCH_VIABILITY_PLAN_2026-10-03.md) and [dataset shortlist](C:/Personal_Coding/thinktank-clearmind/paper_revisions/ONLINE_DATASET_SHORTLIST_2026-10-03.md) if choosing a future study. Their proposals predate the completed failures recorded here.
7. Exact source code, run snapshots, numerical CSV/JSON and verification receipts only for the branch of work being continued.

**What to carry forward:** technically traceable computations, narrow claims, preserved adverse findings, the author's revision scope, and explicit unresolved facts. **What not to carry forward:** old significant-biological claims, stale recording metadata, uncorrected spatial evidence, or a plan to keep changing models until one looks positive.
