# FEIS feasibility and augmentation protocol

Written before running any classification comparison. This is a local prospective analysis plan, not a registered clinical or confirmatory study. Pilot inspection was limited to file formats, timing, class counts and trial organization.

**Question:** Does an adapted, phoneme-labelled segment-recombination augmentation improve prediction of real held-out phoneme trials, compared with simple augmentation and a strong feature-space mixing control? This does not test sentence reconstruction, imagined phoneme timing or metabolism.

**Data and storage.** FEIS v1.1, English participants 01–21, imagined (`thinking`) and subsequent `resting` phases. Downloads, extracted recordings and processed arrays live at `D:/ResearchData/thinktank-clearmind/FEIS/v1.1`. Immutable source checksums and extracted hashes are recorded. The data are ODC-BY; retain authors' supplied attribution and nonprofit-research wording. Source: [FEIS archive](https://zenodo.org/records/3554128), [experiment timing](https://github.com/scottwellington/FEIS/blob/v1.1/experiment-timeline.lua).

**Trial unit and split.** CSV rows contain a trial number, phase and phoneme label. Participant 01 has ten consecutive, class-balanced blocks of 16 trials. Verify that structure for every participant. For each participant, use the first floor(0.7 × number of complete blocks) for training and the remaining later blocks for testing. Every held-out trial must remain outside donor selection, scaling and model fitting. All samples and task phases belonging to one trial stay on the same side. Missing explicit session identifiers prevent calling this session-separated validation. The evaluation tests later trials of already observed people, not transfer to new people.

**Signal handling.** Read the sampling rate and channel order from the CSV header (pilot: 256 Hz, 14 channels). Require one label per trial, a five-second uniformly sampled phase, and finite values. Exclude malformed trials with reasons. Do not infer physical units from plausible amplitudes. Remove each channel's DC offset, use the within-trial common average, a 50 Hz notch (Q=30), and a fourth-order 1–90 Hz Butterworth filter applied separately to each original trial. Keep the central four seconds to reduce edge/cue-transition influence. No filter crosses a training/test boundary or a task-phase boundary. Filtering and averaging are fixed transformations, without a fitted across-trial parameter.

**Artifact limits.** Supplied flag values, channel peak-to-peak amplitudes and channel flatness are reported. Primary screening excludes nonfinite/malformed trials and completely flat trials; it does not claim eye/mouth artifact removal. A predeclared stricter sensitivity excludes flagged trials, trials with two or more flat channels, or more than three channels whose log peak-to-peak value exceeds a training-only median plus six robust standard deviations (1.4826 × MAD, with a floor of 0.1 in log units). No test labels enter rejection decisions. The rest phase follows spoken trials, so it is a diagnostic comparison with possible carryover, not a pure label-independent resting baseline.

**Features.** Compute per-channel log band power from nonoverlapping, one-second Welch windows averaged across the four-second epoch. Bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, low gamma 30–45 Hz. Integrate bins inside half-open band intervals using bin width. The primary feature set contains all five bands (70 features). Lower-frequency-only and low-gamma-only features are secondary. These features are intentionally insensitive to reordering whole one-second blocks; they cannot validate a sentence's phoneme order. High gamma or absolute physiological outputs are not primary outcomes.

**Classifier.** A training-fitted StandardScaler and linear discriminant analysis with `solver='lsqr'`, `shrinkage='auto'`, fixed for every condition. No test-set parameter selection. [Classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

**Training conditions.** Every augmentation condition adds exactly one synthetic example per real training trial. Keep all original real training examples. Use seeds 17, 29 and 43; average those three scores within each participant rather than treating them as independent participants.

| Condition | Added example |
| --- | --- |
| Real only | None |
| Duplicate real | Exact copy; controls the nominal sample count |
| Scale and jitter | Per-trial gain uniform 0.9–1.1 plus Gaussian noise at 1% of training-only channel standard deviation |
| Same-label feature mixup | Convex interpolation of two real training feature vectors of the same phoneme, weight from Beta(0.4, 0.4); simple strong comparator |
| Same-label segment splice | Four one-second blocks at their original within-phase positions, sampled from training trials of the same phoneme; require at least two distinct donors |
| Random-label segment splice | Same duration and construction, with donors from all training labels while keeping the recipient's target label; negative control |
| Shuffle blocks | Reorder four blocks from the original training trial; feature invariance should make the added features equivalent to copies |

The same-label splice is an explicitly adapted phoneme augmentation method. It is not the original cross-phoneme sentence generator. Parent-trial indices are saved and checked against the train/test split. Discontinuities may affect other feature methods; the selected nonoverlapping windows avoid interpreting join discontinuities as useful spectra.

**Outcomes and uncertainty.** Primary outcome: balanced accuracy on untouched, later real thinking trials. The primary comparison is same-label splice minus same-label feature mixup, averaged first over seeds and then across participants. Also report differences from scale/jitter, duplicates and real only. A provisional go/no-go criterion is a mean advantage of at least two percentage points over feature mixup with a participant-bootstrap 95% interval above zero. This criterion is a research decision, not a guarantee of significance or publication. Bootstrap participants 10,000 times with seed 20261003. Other comparisons, feature sets, phases and strict-screening results are exploratory; retain all results, including negative ones.

**Negative control.** Fit the real-only multiband classifier after independently permuting training labels within each chronological 16-trial block, 100 times per participant/phase. Keep real test labels fixed. Report this null distribution and the one-sided Monte Carlo exceedance value with the +1 correction. Do not count permutation draws as biological observations or present a cohort mean above nominal chance as proof of imagery-specific neural decoding.

**Stop and interpretation rules.** Stop for unresolved label/timing errors or unsupported splits; record exclusions. If the adapted method does not improve on simple controls, report that result and revise the research direction. A credible FEIS advantage would motivate a separately planned Kara One word replication. Chisco sentence studies require their own language and donor-library design. No FEIS result establishes sentence reconstruction or calibrated CMRO2.
