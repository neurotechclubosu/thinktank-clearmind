# Round two revision working package

This package improves the supplied v1.26 draft and response through an audited replacement analysis. It is **for author review, not ready for submission**. Read `documents/Author_decisions_before_resubmission.docx` first. The original reviewer notes, manuscript and reply are unchanged.

The data audit found paired task markers, a repeating packet counter previously used as an event anchor, a 250 Hz acquisition specification, and a file inventory that differs from the application's ARPAbet order. All 44 local recordings match the public March 21 source files after line-ending normalization. The replacement analysis follows the archived stimulus slides, with declared vowel/dialect approximations. It does not infer participant identity from initials.

## Deliverables

- `documents/Manuscript_v1.27_author_review.docx`: replacement manuscript with new tables and figures and explicit author-verification items.
- `documents/Response_round2_author_review.docx`: point-by-point proposed response covering all five reviewer requests.
- `documents/Changes_and_evidence.docx`: substantive changes and numerical traceability.
- `documents/Author_decisions_before_resubmission.docx`: remaining decisions and a recommended sequence for John and the authors.
- `figures/`: four regenerated figure sets, PNG and TIFF, 300 dpi. Their underlying numbers are in `results/`.
- `results/`: every simulation draw, summaries, event rows, signal-quality diagnostics, source and spectrum arrays, and recorded numerical checks.
- `inputs/`: frozen texts, exact dictionary entries, declared pronunciation overrides, original stimulus labels and acquisition script, reference metadata, and upstream file hashes.

## Reproduce the analysis

From the repository root, with NumPy, SciPy and matplotlib installed:

```powershell
python paper_revisions/round2_revision/code/reanalyse.py --draws 2000
python paper_revisions/round2_revision/code/test_reanalysis.py
python paper_revisions/round2_revision/code/make_figures.py
```

The analysis uses the existing 44 files in `data/eeg_recordings/`. It has no language-model calls, needs no CMUdict installation or downloads, and does not modify the original files. To run a relocated package:

```powershell
python code/reanalyse.py --data C:/path/to/eeg_recordings --draws 2000
python code/test_reanalysis.py --data C:/path/to/eeg_recordings
python code/make_figures.py --data C:/path/to/eeg_recordings
```

The full dictionary is not needed: the exact used entries and license are included. The source recordings are linked by exact upstream hashes rather than duplicated in this working package. `data_source_manifest.json` records stable paths.

The document builder requires python-docx and uses the local results and figures:

```powershell
python code/build_documents.py
```

For Codex artifact work, the builder was run with the bundled Python runtime. Scientific calculations were run with the existing system interpreter. Versions are captured in `results/results.json` and `environment_versions.json`. Results are conditional on the documented versions, input hashes, seeds and protocol.

## Interpretation

The primary measure is a dimensionless normalized gamma index. The numerical CMRO2 proxy difference has the same value only because an imposed scale with slope 1 maps the index into 2.1–3.1 μmol/(g·min). That scale is not calibrated against metabolism. All eight model outputs are dependent transformations of one input.

The primary ensemble contains 2,000 seeded insertion realizations per text. Sensitivities contain 500 per text. Percentile ranges are **central simulation intervals**, not participant-population confidence intervals. No biological hypothesis-test p-value is calculated. The inserted blocks are recorded instructed waits, not natural sentence pauses. Two-second insertions concatenate two independently chosen one-second waiting blocks.

Trial 1 calibrates the index and common source operator, trial 2 supplies construction blocks, and task trials 3–5 supply the reference. These are disjoint samples within one recording set, not independent participants. Source maps use a simplified sphere and inherited electrode assumptions, not a validated fsaverage anatomical inverse. High correlation does not prove biological equivalence. A matched continuous recording of the three sentences and independent metabolism measurements were not present in the supplied set.

## What remains before a journal submission

Confirm the acquisition version, electrode assignment, participant provenance, ethics/consent and reuse terms, crosswalk and pronunciation choices. Review artifacts, gain/units and line noise, then rerun if the agreed preprocessing or mappings change. Have the domain expert review the model constants and expressions. Approve the revised study framing and author declarations. Deposit the approved package under a stable identifier and create the journal's marked-up copy against the exact manuscript currently on file. None of those external steps is represented as completed here.

The Word documents were reconstructed from extracted legacy text and newly authored content. Historical figures were replaced rather than silently retained. Layout QA used Microsoft Word PDF export because the packaged renderer could not find LibreOffice on this Windows host. Rendered PDFs and page images under `qa/` are internal verification material.
