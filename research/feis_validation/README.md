# FEIS external validation

This research experiment tests an adapted phoneme augmentation method on independently recorded, held-out real FEIS trials. It is separate from the GUI and the round-two manuscript's original donor-bank analysis. It does not reconstruct sentences or calibrate physiology.

All downloaded recordings, extracted CSVs, processed EEG arrays, full predictions and donor records are stored on **D:**:

```
D:/ResearchData/thinktank-clearmind/FEIS/v1.1/
  archives/             original versioned downloads
  extracted/            verified unpacked CSV recordings
  processed/            filtered EEG arrays and their audit metadata
  analysis/             full real-trial predictions and augmentation donor receipts
  download_manifest.json
```

Small results, figures, software snapshots and reports remain in the project:

```
paper_revisions/feis_validation/
  timestamp_rounding_audit.json
  results/
    data_audit_only/
    initial_v1/
    timestamp_rounding_sensitivity/
```

The [prospective local protocol](C:/Personal_Coding/thinktank-clearmind/research/feis_validation/PROTOCOL.md) fixes seven training conditions, later-block holdout, the feature/classifier design and participant-level summaries. The initial evaluation covers all 21 English participants. The supplied merged CSV does not establish independent session identities.

The [primary report](C:/Personal_Coding/thinktank-clearmind/paper_revisions/feis_validation/results/initial_v1/REPORT.md) contains the negative result and limitations. The additional timestamp sensitivity retains 19 trials of participant04 whose six-decimal timestamps were rejected by an overly strict initial tolerance. That sensitivity preserves the original analysis and uses the identical real test trials. Source data were not changed or resampled.

Run from the project root with the working system Python:

```powershell
# Acquire just a pilot, using D: by default.
python -m research.feis_validation.download --participants 01

# Meaningful checks of parsing, holdout isolation, donor labels,
# feature invariance, training-only quality screening and positive/null controls.
python -m pytest research/feis_validation/test_validation.py -q

# These commands overwrite their named result run; use --output for a fresh run.
python -m research.feis_validation.benchmark --audit-only --output paper_revisions/feis_validation/results/data_audit_only
python -m research.feis_validation.benchmark
python -m research.feis_validation.summarize
python -m research.feis_validation.timestamp_sensitivity
python -m research.feis_validation.verify
```

`--root` can set a different external storage directory. The classifier does not use an API or upload EEG. Native sampling/channel metadata are checked; physical units and eye/mouth artifacts are not independently established by this experiment. A timestamp precision sensitivity is not permission to ignore actual missing samples or timing errors.

Dependencies are the installed numpy, scipy, pandas, scikit-learn, matplotlib, threadpoolctl and pytest. Exact run versions, input hashes, protocol hashes and code hashes are saved in each run manifest. The initial code/protocol snapshot is preserved under `initial_v1/source_snapshot/`.

Dataset attribution: Scott Wellington and Jonathan Clayton, Fourteen-channel EEG with Imagined Speech (FEIS), University of Edinburgh, 2019, [v1.1 archive](https://zenodo.org/records/3554128), concept DOI 10.5281/zenodo.3369178. Retain ODC-BY and the author README's research-use wording. The original README and experiment material accompany the archives on D:.

## Bounded stronger-baseline follow-ups

The [follow-up protocol](FOLLOWUP_PROTOCOL.md) adds filter-bank log-covariance/ridge and an EEGNet-style temporal-spatial model, bounded training-only chronological selection, quality/eye/order controls and decision gates. [Parser refinements](KARA_PARSER_PLAN.md) document blind native-metadata corrections before any Kara classifier result. Source snapshots preserve the exact model/protocol version for each run. No new package was installed; this study additionally used installed PyTorch2.10 and MNE1.12.1, with dependency receipts in results.

The completed [FEIS report](../../paper_revisions/feis_validation/results/followup_v1/REPORT.md) and [Kara report](../../paper_revisions/feis_validation/results/kara_followup_v2/REPORT.md) show both families failing their signal gates. No further splicing experiment followed. The [decision brief](../../paper_revisions/feis_validation/FOLLOWUP_DECISION.md) records implications without treating these model-specific null results as general impossibility.

Kara archives and all full EEG/predictions are under `D:/ResearchData/thinktank-clearmind/KaraOne`. All14 official archives were acquired; MM05 stayed format-only, and the other13 were evaluated on four words. The failed acquisition-only `kara_followup_v1` generated no classification scores and is preserved; corrected metadata parsing precedes all `kara_followup_v2` classifier results. Cite Zhao and Rudzicz2015 and retain the official academic/nonprofit release terms. The62 scalp predictors exclude two eye channels and the EMG-labelled task-colour sensor. Sensor markers measure task timing, not internal speech timing.

```powershell
# Implementation checks, including spatial/time known positives and label/mapping failures.
python -m pytest research/feis_validation/test_validation.py research/feis_validation/test_followup.py research/feis_validation/test_kara.py -q

# Review/rebuild reports from the existing completed results, without refitting models.
python -m research.feis_validation.summarize_followup
python -m research.feis_validation.followup_diagnostics
python -m research.feis_validation.summarize_kara
python -m research.feis_validation.kara_figure
python -m research.feis_validation.verify_kara_acquisition

# Model reruns require a fresh output directory; never replace a scored independent run.
python -m research.feis_validation.followup --pilot-diagnostics --output paper_revisions/feis_validation/results/followup_reproduction
python -m research.feis_validation.kara_benchmark --output paper_revisions/feis_validation/results/kara_reproduction
python -m research.feis_validation.summarize_kara --output paper_revisions/feis_validation/results/kara_reproduction
```

Full predictions are automatically stored on D: using the result directory's name. Report rebuilds replace derived summaries only, while the scored source snapshots and native acquisitions remain unchanged. Reusing known test data for new tuning is exploratory.
