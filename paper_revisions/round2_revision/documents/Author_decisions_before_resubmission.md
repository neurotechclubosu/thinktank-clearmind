# Author decisions before the next PLOS revision

Working revision 1.27 — 3 October 2026 — Author verification required

**The main recommendation is to review and resolve the data provenance and signal-quality issues before resubmission. The new draft is materially stronger and more transparent, but it is not ready to send as a validated biological study.**

John’s draft takes useful steps toward narrower claims and seeded simulations. It still retains old paired-electrode tables and adds a simulation-tail p-value that is not a valid biological hypothesis test. The local reanalysis also uncovered earlier event-selection, file-mapping, and sampling assumptions. These are scientific issues rather than copyediting issues.

## What is completed

The replacement bundle includes a rewritten editable manuscript, a proposed point-by-point response, a detailed change record, corrected numerical analysis, all simulation draws and figure inputs, and four regenerated figure sets. The calculations use marked trials and frozen pronunciations, distinguish recorded waits from auxiliary speech, keep construction and reference blocks separate, and report descriptive simulation intervals. All historical inferential tables and biological-equivalence claims are withdrawn from this draft.

## What John and the authors need to verify

### Decisions before resubmission

| Priority | Question or action | Why it matters |
| --- | --- | --- |
| 1 | Confirm the acquisition version and electrode-to-column order for the March 21 DLR files | The public stimulus order conflicts with the app; source-space results depend on the montage |
| 1 | Confirm participant linkage and original approval/consent coverage for these exact files | DLR initials and the current conversation do not establish participant identity |
| 1 | Review acquisition gain, units, packet/timestamp behavior, artifacts and line noise; agree on a preprocessing protocol | Large offsets and substantial 60 Hz neighborhood power limit physiological interpretation |
| 2 | Approve the CMUdict-to-stimulus crosswalk, vowel approximations, and BCIs/streetside pronunciations | A declared crosswalk is auditable but needs protocol and dialect confirmation |
| 2 | Accept the study as an exploratory composition/modeling demonstration or provide independent validation data | The supplied data contain isolated cued trials, not the three continuously imagined full sentences |
| 2 | Verify every model equation and parameter with the domain expert, or retain their illustrative status | The absolute metabolic scale is imposed; parameter provenance is incomplete |
| 3 | Finalize author contact, affiliations, funding, contributions, competing interests, reuse terms and AI-assistance disclosure | These facts and the authors’ independent verification must be confirmed; see Section 2.9 |
| 3 | Deposit the approved exact package, insert its stable identifier, and compare changes with the exact journal-submitted file | Local outputs and a live main branch do not complete the publication reproducibility requirement |

## How to handle the journal revision

Use this package as the author discussion draft. Resolve priorities 1 and 2, freeze the analysis protocol, and rerun if those confirmations change it. Then finalize the manuscript and response with the same tables, figures, and stable release. If the journal deadline is too close, the corresponding author should request an extension explaining that the team is verifying event selection, stimulus mapping, and corrected numerical outputs. No message has been sent and no revision has been submitted.

## What publication depends on

There is a publishable methodological direction only if the computation is technically sound, the data and model assumptions are accurately described, and the claims stay within the evidence. The current experiment does not validate thought decoding or sentence metabolism. Reframing can remove unsupported claims, but it cannot replace missing biological measurements or an unresolved acquisition-quality review. Acceptance remains an editorial decision.

## Start with these files

Read Manuscript_v1.27_author_review.docx and Response_round2_author_review.docx together. Use Changes_and_evidence.docx for traceability. The results folder contains the numerical audit and all simulation draws. The analysis README gives reproduction commands and explains which original files are needed.
