# External-data experiment: current research position

3 October 2026, America/New_York. This brief accompanies the FEIS results; it does not amend the frozen round-two manuscript.

**The first external experiment found no reliable augmentation advantage.** We now have independently recorded EEG from 21 English participants and a reproducible comparison on untouched real trials. That improves the evidence base, but the results do not establish useful phoneme recognition, sentence reconstruction or physiology.

| Result | Finding |
| --- | --- |
| Dataset acquired | FEIS v1.1; all 21 English participants' imagined/rest phases, plus a full recording for the parsing pilot |
| Original imagined trials | 3,312 |
| Real test trials | 1,008, selected from later chronological blocks; not verified independent sessions |
| Initial real-only balanced accuracy | 6.55%, versus 6.25% nominal chance |
| Initial phoneme-splice accuracy | 6.58% |
| Initial simple feature-mixup accuracy | 6.42% |
| Prespecified splice-minus-mixup comparison | +0.17 percentage points; participant-bootstrap 95% interval −0.83 to +1.19 |
| Timestamp-rounding sensitivity | Both splice and feature mixup 6.58%; difference 0.00 points, interval −0.93 to +1.03 |
| Stricter-screening comparison | Splice-minus-mixup −0.08 points, interval −0.86 to +0.69 |
| Block-order control | Shuffled blocks and exact copies give identical scores; these averages cannot establish phoneme order |

All seven conditions used the same real test set within each screening analysis. Augmentation conditions added the same number of examples; synthetic donors were restricted to real training trials. Three random seeds were averaged within each participant. Label-permutation controls, real predictions, donor receipts and data checks are retained. Seven implementation tests pass, including known signal and shuffled-label controls; those tests do not prove biological validity.

**Timestamp caveat.** The original very strict timing tolerance rejected 19 trials in participant 04. Detailed inspection showed decimal rounding below one microsecond rather than established missing samples. A clearly labelled sensitivity retained those trials without changing raw values, test identities or the original analysis. The negative conclusion persists.

**Storage.** Downloads and extracted data are under [the FEIS directory](D:/ResearchData/thinktank-clearmind/FEIS/v1.1). The [download manifest](D:/ResearchData/thinktank-clearmind/FEIS/v1.1/download_manifest.json) preserves versioned URLs, source checksums, extracted hashes, attribution and access terms. Processed EEG and complete prediction/donor files are also on D:. Small results and reports stay in the project.

**What to do with the manuscript.** Keep the v1.27 author-review package's narrow claim boundaries. Do not describe this experiment as positive validation or add a decoding-accuracy benefit. If reported, identify it as a new, negative, external phoneme-level augmentation experiment with its own methods. The test uses a different recording bank and an adapted same-phoneme augmentation, not John's English sentence generator.

**Next research decision.** A positive-effect replication on Kara One is not justified by this result. A separately motivated exploratory test on its higher-density word recordings could still be useful, but would answer a new dataset/method question. Before investing in full-sentence construction, establish a robust phoneme/word signal with reliable acquisition, task timing and artifact assessment, using stronger predefined features/models and proper validation. This result does not rule out every imagined-speech model; it shows that this fixed spectral-feature/linear-classifier design supplies no convincing benefit for our adapted splicing method.

The pending author protocol, participant linkage, ethics/reuse, original-signal-quality and public-deposit checks remain necessary for the round-two submission. This experiment resolves none of those by implication.

Deliverables: [full results report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/REPORT.md), [prospective local protocol](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/PROTOCOL.md), [reproduction guide](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/README.md).
