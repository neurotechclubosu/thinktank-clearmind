# FEIS validation: first external experiment

This is an exploratory external-data experiment following the prospective local [protocol](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/PROTOCOL.md). It evaluates an adapted phoneme augmentation method, using real recorded trials as the test set. It does not validate sentence reconstruction or metabolism.

**The provisional improvement criterion was not met.** Same-phoneme splicing changed balanced accuracy by **+0.17 percentage points** relative to feature mixup; the participant-bootstrap 95% interval is **[-0.83, +1.19]**. 11 participants improved and 9 worsened.

## Recorded data and checks

The [FEIS v1.1 release](https://zenodo.org/records/3554128) was acquired on D:. Our file audit found 3,312 original imagined trials, of which 3,293 passed basic format/finite-value screening. Across 21 participants, 1,008 real later trials were reserved for testing. Labels and timing were checked from native CSV columns. Nominal chance is 6.25% for 16 classes.

The merged CSV files do not establish original session identities. This is a within-person later-trial-block test, not cross-session or new-person validation. Every augmentation donor was confined to the real training set, with parent receipts retained on D:. Features/model scaling used training examples only. Three augmentation seeds were averaged within a person.

Basic exclusions across both task phases: {'nonuniform_time': 38}. The stricter artifact sensitivity uses training-only amplitude thresholds and available flags. It is not a validated eye/mouth cleanup. Full trial reasons and strict-screening counts are in `data_audit.json`.

## Timestamp precision sensitivity

The 19 excluded trials of participant 04 in each phase have six-decimal timestamp rounding, with a maximum step deviation of 0.75 microseconds and grid deviation of 0.5 microseconds. The initial tolerance of 0.1 microseconds was overly conservative; these are not established dropped-sample events. A separate audit-informed sensitivity used a tolerance of one microsecond, retained all 160 trials and kept exactly the same 48 real test trials. Original recordings and the primary results remain unchanged.

With this change to the admission tolerance, the cohort splice-minus-feature-mixup difference is +0.00 percentage points, with a 95% interval [-0.93, +1.03]. The provisional criterion remains unmet. See [sensitivity results](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/timestamp_rounding_sensitivity/run_manifest.json). This sensitivity is supplemental, not a replacement prespecified outcome.

## Primary imagined-phoneme comparison

| Training condition | Mean balanced accuracy | Participant-bootstrap 95% interval |
| --- | ---: | ---: |
| Real only | 6.55% | 4.96–8.13% |
| Duplicate real | 6.05% | 4.66–7.44% |
| Scale and jitter | 6.15% | 4.86–7.47% |
| Feature mixup | 6.42% | 5.46–7.47% |
| Phoneme splice | 6.58% | 5.29–7.94% |
| Random-label splice | 5.92% | 5.22–6.61% |
| Shuffle blocks | 6.05% | 4.66–7.44% |

![Imagined-phoneme accuracy](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/figures/thinking_accuracy.png)

![Participant differences](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/figures/primary_paired_effect.png)

## Controls and limits

- Real-only thinking: 6.55% balanced accuracy; shuffled-training-label mean 6.15%. Per-person permutation results are retained without treating permutation draws as new participants.
- Real-only resting: 5.75% balanced accuracy; shuffled-training-label mean 6.17%. Per-person permutation results are retained without treating permutation draws as new participants.
- Maximum difference between block shuffling and exact-copy augmentation: 0.000000 percentage points. The chosen spectral averages cannot establish correct phoneme order.
- Rest follows spoken trials and can retain phoneme/task carryover. It is a diagnostic comparison, not proof of a label-independent baseline. Native units and eye/mouth contamination remain limitations.
- The primary comparison was specified before classification. Secondary bands, strict screening and other comparators are exploratory. Confidence intervals describe variability across the supplied participants; they do not remove acquisition or protocol confounds.
- Results use one fixed, simple feature/classifier design. A weak result here does not prove that no other imagined-speech algorithm can work. A positive result here would need independent replication and artifact review.

## What this means for the paper

These results do not support adding a performance claim for phoneme splicing to the manuscript. Keep the negative finding and controls. Review the phoneme-information and artifact results before investing in a larger sentence model. A Kara One study would be a separately justified exploratory experiment, rather than confirmation of an improvement established here.

The round-two author-review manuscript was not changed by this experiment. Author verification, ethics/reuse checks for the original donor files and the public-deposit decision remain separate requirements.

## Reproducibility

Data root: `D:\ResearchData\thinktank-clearmind\FEIS\v1.1`. Full real-trial predictions and donor receipts are saved under its `analysis/initial_v1` directory. Small tables, data audit, the decision record and software/protocol hashes are alongside this report. Dataset citation and ODC-BY terms are retained with the downloaded source README.

Commands from the project root:

```powershell
python -m pytest research/feis_validation/test_validation.py -q
python -m research.feis_validation.benchmark
python -m research.feis_validation.summarize
```
