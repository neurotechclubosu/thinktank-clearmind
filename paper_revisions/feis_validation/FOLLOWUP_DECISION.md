# Decision after the bounded baseline follow-ups

4October2026UTC. The work is complete; these are actual results, not a proposed experiment. Both studies use real held-out trials and preserve their source/model/protocol snapshots. FEIS is exploratory because its first results had already been inspected. Kara's task/model budget was fixed before classification; acquisition-only corrections were recorded while all classifier results remained unexamined.

**Decision: stop further splicing experiments under this design. Neither dataset demonstrates the useful, reliable real-EEG decoding baseline required by the plan.** This does not prove that imagined speech is undecodable or that every possible augmentation method fails. It means the proposed contribution remains unestablished.

## Evidence

| Data/task | Spatial covariance | Temporal-spatial CNN | Nominal chance |
|---|---:|---:|---:|
| FEIS,21 people/16 imagined phonemes |5.56% [4.07,7.14]%|6.05% [4.56,7.54]%|6.25%|
| Kara,13 people/four imagined words |28.85% [23.56,35.58]%|25.10% [20.67,29.42]%|25%|

Figures are balanced accuracy and participant-bootstrap95% intervals; primary CNN seed17. Raw percentages should not be compared between tasks as if they had the same chance level. Both families fail the prespecified feasibility criteria on both datasets. Kara's spatial-model interval is wide enough that a useful effect remains possible, but the present study does not establish it. Strict global chronology produces22.82%/21.19%, and eye-only control30.96% [24.04,38.17]% does not establish specificity. Eye-regression/screening/alternate-seed results are retained, including failures. No isolated high-performing participant was selected as proof.

FEIS model/fold choices were restricted to training; its shared task order and after-speech rest phase complicate interpretation. Kara primary chronology is within each class and can interleave across classes; a separate globally later holdout uses its own training-only choices. Conditional label permutations require exchangeability assumptions, and trial labels denote the instructed task rather than measured internal speech onsets. Eye controls do not independently measure mouth-muscle contamination. Physical gain and metabolism remain uncalibrated by us.

## What this means for the manuscript

The paper cannot use these results to claim successful neural sentence reconstruction, established augmentation benefit, or measured CMRO2/neurovascular biology. Merely adding more models or polishing the wording will not resolve that evidence gap. The v1.27 package remains author-review material, not an approved submission.

A construction/visualization and limits framing is possible, but authors must decide whether it answers a worthwhile scientific question beyond a demonstration. It would need a clear rationale, positioning against existing construction/simulation/augmentation methods, complete provenance, appropriate controls and precisely bounded negative conclusions. [PLOS ONE considers technically sound negative results](https://journals.plos.org/plosone/s/criteria-for-publication), but these model-specific findings do not automatically make a compelling negative paper or guarantee acceptance.

If biological composition remains the question, the next study should collect approved paired data: isolated units and continuous short imagined sequences from the same participants, including the same units in different orders, repeated across an independent evaluation session. Measure eyes/mouth artifacts, retain complete cue/acquisition markers and verified units/channel/reference metadata, and compare real-versus-real repeatability with constructed-versus-real performance using order-sensitive endpoints. Choose sample size/precision from a pilot and the desired effect rather than inventing a publication minimum. Successful phoneme/word classification alone would still not validate sentences or metabolism.

## Immediate author actions

1. Share these actual negative results within the author team; no message has been sent by this work.
2. Verify original recording/participant/session/consent/reuse facts and the journal deadline. The supplied bank must not be called John's own recording without confirmation.
3. Decide whether to request more time for a genuinely matched study or submit a much narrower, accurately supported account. The corresponding author handles editorial communication.
4. Do not run an open-ended model search or add a new dataset merely to obtain a positive result. A new study should have a different, explicit hypothesis and independent evidence.

Reports: [FEIS](results/followup_v1/REPORT.md), [Kara](results/kara_followup_v2/REPORT.md). All new archives, EEG, full predictions and caches are on D:. The original negative FEIS augmentation result, original submissions and author-review package are preserved. No human recruitment, collaborator/journal message, GitHub push or public deposit has been performed.
