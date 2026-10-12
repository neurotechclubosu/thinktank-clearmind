# What to do after the first FEIS experiment

**Execution update:** the bounded follow-up below is now complete. Both FEIS and independent Kara baselines failed their prespecified signal gates. See [actual results and current decision](FOLLOWUP_DECISION.md); this dated proposal is preserved for provenance.

3 October 2026. This is a proposed follow-up, not a claim of additional results. The primary FEIS analysis and its negative finding remain unchanged.

**Recommendation:** make one bounded attempt to establish useful prediction of real imagined phonemes/words with stronger fixed baselines, then test whether the adapted splicing method adds value. Use FEIS for exploratory development and a new, explicitly held-out Kara One evaluation for independent evidence. Keep all new datasets on D:.

The initial FEIS real-only multiband classifier averaged 6.55% balanced accuracy against 6.25% nominal chance. The augmentation comparison also failed its provisional improvement criterion. This shows no useful advantage for our tested feature/classifier/splicing design. It does not establish that every model lacks information, or directly test the original cross-phoneme sentence generator. Improving an existing decoding baseline alone would not establish the paper's proposed contribution.

## A bounded study, with decision points

| Stage | Work | Decision |
| --- | --- | --- |
| 1. Signal and protocol diagnosis | Audit reference, units, timestamps, spectral noise and label-to-phase mapping. Use the existing pilot's listening/spoken phases as diagnostic comparisons. Check whether available artifact indicators predict labels. | Interpret spoken/listening success cautiously; it can reflect sensory input or muscles and does not validate imagined speech. Resolve acquisition errors before comparing augmentation. |
| 2. Stronger real-data baselines | Fix two complementary model families: a regularized spatial/covariance model and a compact temporal-spatial EEG model. Limit the tuning budget and fit all choices inside training folds. Treat FEIS follow-ups as exploratory because its first test results have already been inspected. | Require repeatable performance beyond shuffled-label controls, with adequate uncertainty and artifact/chronology checks. A selected favourable split or isolated high-performing person is insufficient. |
| 3. Independent dataset evaluation | Acquire and audit Kara One on D:. Define the task, split unit, exclusions, model-selection process and untouched final test portion before viewing its classification results. Four-word classification is a possible declared task; do not change to easier contrasts after seeing results. | Kara One has different classes and protocol, so it tests external method usefulness rather than exact replication of the FEIS phoneme task. Do not compare its raw accuracy directly with FEIS's 16-class accuracy. |
| 4. Utility of the adapted splicing method | If a real-data baseline works, compare splicing with real-only training, duplicate examples, ordinary augmentation and feature mixing. Match data budgets and keep donors strictly within training. Evaluate all conditions on the same untouched real test trials. | A credible advantage motivates replication. A useful baseline with no splicing advantage motivates abandoning the augmentation-benefit claim. |
| 5. Research direction | Decide from the full results, including failures and uncertainty. | If stronger baselines also fail, pause the composition claim and prioritize signal quality or a different supported research question. Do not keep enlarging the search until one positive result appears. |

These stages are hypotheses about what may help, not evidence that a more complex model or higher-density recording will succeed. The next protocol must set a scientifically meaningful effect and precision target for its chosen task, with all comparisons and stop rules fixed before the independent test. Do not automatically transfer the first experiment's provisional two-percentage-point rule to a different four-word task.

**Why these candidate baselines:** our initial whole-epoch spectral means omit detailed timing and electrode relationships. Covariance features and temporal-spatial models can test whether that omission matters. [EEGNet](https://arxiv.org/abs/1611.08024) is an established compact EEG architecture, but its original validation covered other BCI paradigms; it does not guarantee imagined-speech performance. Neither baseline should be presented as a novel contribution by this project.

**Why Kara One:** the [official release](https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html) supplies scalp EEG from a 64-channel cap, phonemic/syllabic and English word prompts, eye channels, and trial indices. Audit which external channels actually measure what: the release notes that its channel named EMG contains the colour sensors, so that channel must not be treated as a direct mouth-muscle measurement. A small download/parsing pilot precedes any full-cohort evaluation. Published binary phonological-category accuracy is not an expected accuracy for an eleven-class or four-word experiment.

## The original sentence hypothesis still needs a different experiment

Neither FEIS nor Kara One supplies a complete matched English phoneme library and independent full-sentence imagination recordings. If the team wants to retain a biological composition claim, obtain approved recordings of isolated units and short continuous imagined sequences from the same people, with repeated trials and a separate evaluation session. Use sequences with the same units in different orders. Measure real-versus-real repeatability alongside constructed-versus-real similarity, using features that can distinguish order. Cue timing must not be described as measured internal phoneme timing.

Chisco can support a Mandarin sentence study, but does not remove the missing isolated-unit-bank requirement. The current metabolic outputs remain illustrative until independently calibrated. Successful phoneme/word prediction alone would validate neither sentences nor CMRO2.

## Coordinate the paper in parallel

John and the authors should review the negative external result and confirm the original acquisition protocol, participant/session linkage, channel order, physical units and consent/reuse. Confirm the journal's revision deadline before designing the next study around it. The corresponding author can ask the editor for more time if needed; no message has been sent.

PLOS ONE explicitly considers negative/null results and requires conclusions supported by the data. That makes a rigorous limits study a possible direction, but this single weak-baseline experiment does not by itself supply a strong publishable negative claim. The rationale, appropriate baselines, controls, precision and literature positioning still need to be established. [Publication criteria](https://journals.plos.org/plosone/s/criteria-for-publication).

Immediate priorities are stages 1 and 2 and the author checks. Kara One acquisition and new classifier experiments have not been started by this decision brief. The v1.27 manuscript remains author-review material.
