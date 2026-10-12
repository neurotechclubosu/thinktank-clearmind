# Revision and research working files

This directory contains the PLOS ONE author-review work and the subsequent EEG feasibility analyses. These are working drafts and recorded experiments, not an approved journal submission or a final validated biological model.

Start with [the 12 October handoff](REVISION_HANDOFF_2026-10-12.md). Its repository-status section records the checkout before this material was committed; consult Git history for later publication of the files.

## Manuscript revision

- [Replacement package and reproduction guide](round2_revision/README.md)
- [Working manuscript](round2_revision/documents/Manuscript_v1.27_author_review.md)
- [Proposed reviewer response](round2_revision/documents/Response_round2_author_review.md)
- [Change and evidence record](round2_revision/documents/Changes_and_evidence.md)

The proposed scope is text-guided composition of recorded EEG and illustrative neurovascular-model outputs. Unknown speech, continuous imagined sentences and physiological calibration have not been validated. The inherited source-space electrode assignment still requires correction and regeneration following John's metadata confirmation. Do not submit the manuscript or interpret its existing spatial figures as final evidence.

The older `Round2_revision_author_review.zip` includes four Word drafts and the earlier analysis snapshot. It predates later metadata and feasibility results. Markdown files and subsequent reports must be reviewed together; the ZIP is not a current submission-ready bundle.

## Completed feasibility experiments

- [Initial FEIS augmentation report](feis_validation/results/initial_v1/REPORT.md): no reliable splicing advantage over matched feature mixup.
- [Bounded FEIS/Kara follow-up decision](feis_validation/FOLLOWUP_DECISION.md): both fixed baseline families failed their feasibility criteria.
- [John's metadata and exclusions](john_data_audit/AUTHOR_CLARIFICATIONS_2026-10-06.md)
- [John's independent-session results](john_phoneme_analysis/REPORT.md): spectral 2.08% and covariance 2.17%, versus 2.27% chance for 44 recorded indices.

Negative findings apply to the stated methods and data. They neither establish general imagined-speech impossibility nor invalidate execution of a known-text lookup. They do limit claims of demonstrated neural usefulness, augmentation benefit and biological reconstruction.

## Data and remaining decisions

Downloaded FEIS/Kara archives and John's expanded EEG delivery, full processed arrays and prediction caches remain outside this repository on D:. Recorded manifests, summaries, plots and source snapshots are included here. The manuscript's original donor-bank derivative arrays are separate supporting artifacts, not the new downloaded datasets.

Exact acquisition/stimulus provenance, units/reference/preprocessing, signal quality, applicable participant/reuse facts, model assumptions and final author approval remain open. A Git commit of these working materials does not resolve those facts or constitute an approved final data deposit. See the handoff for the full checklist, locations and commands that can overwrite derivative runs.
