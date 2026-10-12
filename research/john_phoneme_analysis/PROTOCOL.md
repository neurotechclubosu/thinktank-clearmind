# Fixed exploratory phoneme-index repeatability analysis

6 October 2026. Saved before any classifier scores on this delivery. Prior raw signal, metadata, original DLR session1, FEIS and Kara results have already been inspected. This is an exploratory local analysis plan, not a registered confirmatory study. No model search or sample selection based on task performance is permitted in this run.

Question: Does recorded cued-phoneme index structure repeat across independent complete sessions, and is task alignment stronger than instructed-wait or acquisition-timing alignment? Index labels are primary; actual phonetic and dialect interpretation remains conditional on the recorded stimulus crosswalk.

## Data ownership

Use only the saved 13 complete-session manifest:572files/2860paired recalls, four reported participants. Session1 is training for each participant (220recalls/person). All other complete sessions are evaluation:1980recalls, nine test sessions. Five recalls from a file always share ownership. Exact/copy duplicate sessions and truncated files have already been excluded with timestamp evidence. Preserve input hashes and whole-task hashes, and verify them before extraction. No windows, donor data, features, scaling or model selection cross train/test ownership. No training hyperparameter search.

An additional John different-day diagnostic trains only sessions1,3,4 from21March and evaluates5,6,7 from23March. Because results from session1 were previously inspected, this is still exploratory. Other aliases do not supply different-day recordings; call their comparisons session holdouts rather than day holdouts. No between-person classifier is included; four people are a small within-person repeatability cohort.

## Extraction and processing

Nominal250Hz,16columns in John-confirmed order `Fp1 Fp2 F7 F3 F4 F8 T3 C3 C4 T4 T5 T6 P3 P4 O1 O2`. Per task, use samples125–374 after the onset marker (0.5–1.5nominal seconds). Filter the full marked interval, then crop:linear detrend, common average reference (CAR), fourth-order Butterworth1–45Hz, zero-phase SOS filtering with reflection padding. Never filter across recordings or task/wait boundaries. Nominal sample timing is distinct from timestamp delivery.

The wait control uses250samples immediately preceding each onset, only if entirely after the preceding end marker (or file start for marker1). Same processing/length as task. Wait is instructed waiting, with possible carryover/early visual-cue contamination; it is not a separately collected cue-only control or a natural sentence pause. If a wait window is incomplete or crosses a marker, stop extraction, record the failure, and amend only acquisition parsing before inspecting classifier results. Never pad a wait using task data.

Physical units/gain/original reference/processing status remain unverified. Treat values as recorded units, use relative/scale-normalized features, and do not infer physiological magnitude. CAR is an analysis transform, not a statement about the original hardware reference. No EOG/EMG or neural-specific artifact ground truth is available.

## Two fixed feature families and classifiers

1. Spectral:250-sample periodic-Hann Welch window/no overlap/constant detrending;1Hzbins. Log-relative band powers1–4,4–8,8–13,13–30,30–45Hz, one vector per16channels (80features).
2. Spatial covariance:demean each cropped channel, covariance normalized by mean diagonal,10%isotropic shrinkage, symmetric matrix log, upper triangle with sqrt(2)off-diagonal weights (136features). Fixed broadband1–45Hz; not anatomical localization.

For each family, StandardScaler fits training only. Each training index has five recalls; its standardized mean is the class centroid. Predict the nearest Euclidean centroid among all44classes. No tuning, feature selection, neural network, or post-result alternate model. This transparent retrieval baseline estimates cross-session alignment and does not claim optimal decoding. Fixed class-mean geometry follows the [nearest-centroid definition](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html); preprocessing uses [SciPy's documented signal routines](https://docs.scipy.org/doc/scipy/reference/signal.html).

## Outcomes and controls

Primary:44-class balanced accuracy on held-out task recalls; all sessions balanced so accuracy equals balanced accuracy. Nominal chance1/44=2.2727%. Average test sessions equally within participant and participants equally within cohort. Report individual participants and sessions; do not count channels/recalls as independent biologicalN.

Secondary:classify held-out file-mean features (44files/session) using the same training centroids; accuracy describes source-file repeatability. Report fixed score margins and confusion counts. This does not introduce another biological replicate.

Controls/sensitivities, all predetermined:
- Identical two-family pipelines on the recorded pre-onset waits, using phase-specific training scalers/centroids.
- Timing-only nearest-centroid features:within-session file acquisition rank/43, marker/5, interval duration, interval sample count/500, onset packet counter/255. Rank uses acquisition timestamps without task scores; every file's recalls remain grouped.
- Task without CAR, otherwise identical (reference sensitivity).
- Task spectral features with9–11,19–21,29–31,39–41Hz bins omitted from band and total power (fixed harmonic sensitivity motivated by already inspected periodic raw traces). Empty/zero powers fail instead of pretending information survives. Notches cannot prove artifact removal or establish harmonic cause.
- Task screen sensitivity:training upper outlier limits median+6*1.4826MAD for log central RMS and log maximum derivative, applied to training and test. These are relative diagnostic thresholds, not validated medical/EEG unit cutoffs. If training loses any class, record the sensitivity as unestimable. Keep44-class scoring even if evaluation class counts differ; report retained counts.
- John different-day diagnostic, same two task feature/classifier families.

## Conditional alignment null and uncertainty

Use999seeded bijections of44class identities per participant, shared across all their test sessions. A fitted centroid has its label renamed; this is exactly equivalent to refitting after a bijective relabeling of the single training session for the primary nearest-centroid pipeline. It preserves five-recall/file grouping and session dependence. It tests alignment of class identities under an exchangeability assumption, not imagery specificity or experimental randomization. Save every permutation and score. Group null independently relabels each participant and averages participant/session-weighted scores. Holm correction across the two primary task families only. Wait/timing/sensitivity p-values, if provided, remain exploratory and are not used to select models.

Give per-participant outcomes, and a10,000draw participant bootstrap for the mean as descriptive uncertainty (N=4 makes this unstable/discrete, not a general population guarantee). Do not substitute trial/bootstrap uncertainty for participant uncertainty. No inferential claim rests on one unusually good participant.

## Decision rule and stopping

A task family passes a limited **alignment feasibility criterion** only if mean balanced accuracy>=chance+5percentage points, at least3/4participants exceed chance, and group conditional permutation Holm p<.05. The5-point threshold is a chosen practical floor for this pilot, not a publication minimum or an externally established clinical utility target.

Report task-minus-wait and task-minus-timing effect sizes. Call task specificity unestablished if strong wait/timing alignment or harmonic/reference sensitivity offers an unresolved alternative explanation. Even passing the alignment criterion does not establish a neural imagined-phoneme mechanism without cue/artifact controls. Augmentation/composition experiments are not run in this stage; a successful, credible real-data baseline would motivate a separately fixed utility experiment. Failure stops this baseline design; do not expand models or omit adverse participants to obtain success.

No continuous-sentence, natural-pause, metabolic calibration or source-localization inference is performed. The previous v1.27 spatial outputs need a separate regeneration with the confirmed channel order. This run will produce source/epoch receipts, quality and prediction tables, null distributions, verified summaries, figures and a manuscript decision report. Original manuscripts and prior failed FEIS/Kara results remain distinct historical evidence.
