# Follow-up status

Completed4October2026UTC (3OctoberNewYork). Both bounded baseline evaluations and result checks are complete. The v1.27 author-review manuscript and the original FEIS augmentation result remain unchanged.

## Complete: stronger FEIS baselines

All21 English participants;3312 valid real imagined trials,1008 later real test trials. Two stronger models were chosen with a small, fixed training-only validation budget:

| Model | Balanced accuracy | Participant95% interval |
|---|---:|---:|
| Filter-bank spatial covariance/ridge |5.56%|4.07-7.14%|
| Compact temporal-spatial EEGNet-style CNN, primary seed17 |6.05%|4.56-7.54%|

Nominal16-class chance is6.25%. Neither meets the prospectively fixed signal threshold. CNN seeds29/43 and strict screening remain near chance. We therefore did not run further FEIS splicing comparisons. This is an exploratory negative result for these models/data/protocol, not proof that imagined speech is undecodable.

Diagnostic rest, raw amplitude/noise proxies, and one-person listening/spoken phases do not establish imagery-specific information. All21 share the same task-ID/prompt mapping; a supplementary scheduling-only control can exceed nominal chance without EEG. These limitations matter for interpreting small above-chance scores.

Independent verification recovered352 saved scores from16814 real-trial predictions, checked native labels and train/test/inner-fold isolation, and verified source snapshots and processed-cache hashes. Synthetic controls verify recoverable spatial and temporal signals and training-only transforms. See [complete report](results/followup_v1/REPORT.md), [decision](results/followup_v1/decision.json) and [integrity checks](results/followup_v1/integrity_verification.json).

## Complete: independent Kara One evaluation

All acquisitions are under D:/ResearchData/thinktank-clearmind/KaraOne. MM05 is an excluded format-only pilot, with no classifier score. The other13 participants form the predefined independent cohort for four words: gnaw,knew,pat,pot. Chance is25%; no switch to easier binary contrasts.

Pilot verifies62 scalp channels plus VEO/HEO eye channels. Supplied imagined intervals are filtered separately and cropped to exactly the central four seconds. Label alignment is checked against integrated release metadata and original delivered-prompt logs. Repeated failed attempts require original task-marker corroboration: MM08 has137 delivered prompts and131 retained intervals; the colour sensor uniquely maps the131 retained trials, identifies six omitted attempts, and agrees with all supplied labels. It measures task timing, not internal speech onsets. The sensor is excluded from model inputs.

All14 official archives are stored on D:, totalling24,824,165,739bytes (24.82GB). The13-person independent cohort has636 target word trials:424 real training and212 real test trials. All13 native CNT files have different source hashes. MM05 remained format-only and was excluded from classification.

Final acquisition verification checked all14 archive SHA256 receipts,151 listed extracted files and12 additional delivered-log receipt entries. These are local integrity checks; the publisher does not supply authenticated checksums. See [acquisition verification](kara_acquisition_integrity.json).

| Model | Balanced accuracy | Participant95% interval |
|---|---:|---:|
| Filter-bank spatial covariance/ridge |28.85%|23.56-35.58%|
| Compact temporal-spatial EEGNet-style CNN, primary seed17 |25.10%|20.67-29.42%|

Four-word chance is25%. Neither passed the predefined>=30%/interval-above-chance/permutation criterion (Holm p=.184/.516). Uncertainty includes chance. Strict global chronology yields22.82% and21.19%; primary eye-only control is30.96% [24.04,38.17]%, also uncertain. Do not describe these controls as proof of artifacts, but they do not establish robust imagery-specific decoding. A useful covariance effect is not ruled out with this imprecision; it was not convincingly demonstrated by this bounded design.

The cohort evaluation includes fixed spatial/temporal models, training-only selection, eye-only and amplitude controls, training-fitted eye regression, strict screening, and a separate globally chronological holdout. [Protocol](../../research/feis_validation/FOLLOWUP_PROTOCOL.md) and [parser/diagnostic refinement](../../research/feis_validation/KARA_PARSER_PLAN.md) were recorded before any Kara classifier results. The first acquisition-only gate is preserved as `kara_followup_v1`; it generated no classifier scores. Task-boundary and duplicate-log fixes preceded `kara_followup_v2` classification. Independent verification recovered156 scores from2540 real predictions, with native labels and primary/global trial/fold isolation and cache/source snapshots checked. See [Kara report](results/kara_followup_v2/REPORT.md), [decision](results/kara_followup_v2/decision.json) and [verification](results/kara_followup_v2/integrity_verification.json).

## Current research decision

Both datasets failed their bounded real-signal feasibility gates. No additional splicing comparison was run on either dataset. Keep the original negative FEIS augmentation finding. Do not continue expanding this model search to obtain a favourable result, claim biological sentence reconstruction, or describe imposed metabolic values as measured physiology. The next biological study needs approved paired isolated-unit and continuous-sequence recordings, with artifact measurements and an independent evaluation session. A narrower construction/limits manuscript is an option for author discussion, not a demonstrated publication-ready contribution. See [decision brief](FOLLOWUP_DECISION.md).

## Author work remains necessary

John/authors need to reconcile original protocol, participant/session identity, consent/reuse, channel order, gain/units and signal quality, and confirm the revision deadline. Real continuous imagined sentences matched to an isolated-unit bank remain missing. Metabolic outputs remain illustrative.21 meaningful parser/leakage/spatial/temporal/scale controls pass. No collaborator/journal message, submission, GitHub push or public deposit has been made in this follow-up.
