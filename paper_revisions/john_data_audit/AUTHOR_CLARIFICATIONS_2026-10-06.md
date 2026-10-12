# John's clarification and analysis selection

**Subsequent analysis, 6 October:** The fixed independent-session evaluation is complete. Spectral and covariance accuracy are 2.08% and 2.17%, versus 2.27% chance; both pilot alignment criteria fail. See [the verified report](../john_phoneme_analysis/REPORT.md). The selection and author facts below remain the pre-analysis record.

6 October 2026. John’s reply was relayed by the user in this chat. The original wording is preserved in `author_metadata_2026-10-06.json`. This addendum updates the 5 October audit; earlier descriptive counts and original files remain intact.

## What is now established by the author

- The instructed recall lasts two seconds, the wait one second, and matching marker values identify recall start and end.
- Filename suffixes identify separate sessions covering all phonemes. Individual recalls are the five marked intervals within each file.
- Recording columns follow the electrode order listed in Section 2.4 of the Frontiers GhostTalker paper. John explicitly directed us to that paragraph’s order.
- Glitch files can be excluded and analysis can use complete runs. `mapo.txt` is a dummy and has no research role.
- No full-word recordings or separate eye/muscle-action recordings are available in this delivery, according to John. No independent continuous imagined-sentence reference was supplied.

John's first answer confirms that the presentation defines the phone order. It does not make the supplied 16-slide object deck equivalent to the recovered 91-slide phoneme deck. We retain the recovered script/deck order as the operational candidate and the observed deck incompatibility as provenance. Exact executed-deck equivalence and dialect/ARPAbet approximations should not be silently marked confirmed. This distinction need not stop a clearly labeled exploratory index-level audit.

## Electrode assignment correction

Python EEG columns 1–16, in John’s indicated order:

`Fp1 Fp2 F7 F3 F4 F8 T3 C3 C4 T4 T5 T6 P3 P4 O1 O2`

The published paragraph lists this sequence. The paragraph alone describes the electrode inventory; John’s reply supplies the additional statement that its sequence is the recording-column order. [Frontiers, Section 2.4](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1306277/full).

Fourteen of these labels differ from the inherited `CHANNELS` list in `paper_revisions/round2_revision/code/sphere.py`. Existing v1.27 source-space outputs therefore remain conditional historical outputs and require regeneration with this corrected assignment. Any electrode-specific presentation also needs relabeling. Channel-averaged spectral statistics are invariant to relabeling alone, but a source inverse depends on the physical channel assignment. This addendum does not regenerate or endorse the existing numerical manuscript results, and does not turn the sphere model into validated anatomy.

The author did not specify gain, physical reference/ground, numerical units, preprocessing state, or the cause of the periodic signal pattern. These remain unknown entries, not guessed equipment defaults. His reply also supplies no four-person demographic summary or new secondary-use/public-release approval. Existing author-review requirements remain open; no further message has been sent.

## Complete-session manifest

Duplicate ownership was assigned from the recorded timestamps relative to the other files in the corresponding sessions, without any classifier scores:

- The shared `DLR_27_2`/`DLR_27_3` recording occurs inside session 3's peer time range and outside session 2's peer time range. Retain `_3`; exclude the misplaced `_2` copy.
- The shared `GTP003_2_2`/`GTP003_2_3` task intervals occur inside session 2's peer range and outside session 3's peer range. Retain `_2`; exclude the misplaced `_3` copy. The files differ in one cell outside their identical task intervals.
- Exclude `GTP003_18_3` and `GTP004_11_2`, which are truncated and lack paired events.
- Exclude `mapo.txt` as author-identified dummy data.

These are timestamp-supported ownership inferences, not confirmation from an acquisition log. Applying complete-session selection excludes the entire affected incomplete run from the primary pool rather than substituting a copied recording from another session.

| Participant | Structurally complete session suffixes | Recording files | Marked recalls |
|---|---|---:|---:|
| GTP000 / DLR | 1, 3, 4, 5, 6, 7 | 264 | 1,320 |
| GTP003 | 1, 2 | 88 | 440 |
| GTP004 | 1, 3 | 88 | 440 |
| GTP007 / GT007 | 1, 2, 3 | 132 | 660 |
| Total | 13 complete sessions | 572 | 2,860 |

Every selected session has all 44 indices and five paired recalls per file. All 2,860 complete task signals have distinct hashes. The complete-session pool is smaller than the earlier 3,500 unique-event pool because it omits the otherwise valid parts of incomplete sessions. No physical file was deleted or moved. “Complete” certifies index/marker coverage and duplicate ownership only; it does not certify artifact-free or neural signal.

A separate earliest-complete-session-per-person manifest provides four sessions, 176 files and 880 recalls. It is a descriptive subset selected by session ID, not by performance. John’s statement that one run is sufficient is an author opinion; information content, repeatability and validation precision still have to be measured. All other complete sessions remain available for independent evaluation. No train/test assignment or new model fitting occurred here.

## Consequence for the next analysis

We can proceed with a bounded exploratory analysis of acquisition quality and repeatability across independent cued-phoneme recordings. Use the confirmed electrode order, fixed preprocessing sensitivities, whole-session ownership, and participant-equal summaries. Keep the prior FEIS/Kara outcomes visible. The stimulus index is the safest primary label until the candidate phone crosswalk is accepted; even a correct phone mapping cannot remove confounding by the visual cue.

For any proposed composition/augmentation experiment, first specify the task, controls, outcome, model budget and stopping rule, then establish a useful real-data baseline on held-out sessions. Source-localization diagnostics additionally require regeneration and review of the reference/model assumptions. Do not reduce the available data to one session per person merely because the author said one run might suffice.

The supplied material cannot establish biological continuous-sentence composition, natural sentence-pause realism, eye/muscle specificity, or an EEG-to-CMRO2 calibration. More runs improve repeatability assessment; they do not independently measure those outcomes. No publication-readiness claim follows from these confirmations.

## Files

- `author_metadata_2026-10-06.json`: supplied reply, confirmed facts, unknown facts, assignment correction and selection counts.
- `confirmed_channel_order.csv`: explicit column-to-electrode mapping and comparison with prior sphere labels.
- `session_eligibility.csv`: every delivered session and its selection status.
- `complete_session_recordings.csv`: selected 572 files and provenance.
- `one_session_per_person_recordings.csv`: optional descriptive 176-file subset.
- `analysis_exclusions.csv`: glitch/dummy exclusions and incomplete-session decisions.
- `duplicate_session_ownership.json`: peer timestamp ranges supporting copy ownership.

Reproduce with `python research/john_data_audit/author_confirmations.py`. It updates only these derivative manifests and never changes original recordings or archives. Assertions verify complete index coverage, five paired recalls per selected file, unique task hashes, and unambiguous timestamp ownership for each duplicate pair.
