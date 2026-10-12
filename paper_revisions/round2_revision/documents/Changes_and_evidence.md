# Changes and evidence for the second round revision

Working revision 1.27 — 3 October 2026 — Author verification required

This record compares the supplied version 1.26 manuscript and response with the replacement working revision. It records substantive changes and why they were necessary. The original .doc files remain unchanged. The new manuscript was reconstructed as an editable .docx from verified extracted text and a new section structure; its old layout and historical figures are not preserved as quantitative evidence.

### Revision record

| Original issue or claim | Replacement and evidence | Location |
| --- | --- | --- |
| Imagined speech reconstruction title | Text-guided composition; no unknown utterance decoding evaluated | Title, Abstract, Section 1 |
| Fractal-inspired or self-similar brain model | Removed unsupported geometric claim | Sections 1 and 2.7 |
| LLM trained on autobiography | Retrieval and prompting; weights not fine-tuned | Section 2.2 |
| Responses constrained to 20 words | Frozen responses exceed target; enforcement not claimed | Section 2.2, Table 1 |
| Counter zero treated as task onset | Paired markers; 44/44 old blocks before first task | Section 2.1, events.csv, data_audit.csv |
| 256 Hz from counter/block convention | Documented nominal 250 Hz and timestamp diagnostics | Section 2.1 |
| Alphabetical ARPAbet-to-file mapping | Public-slide crosswalk; vowel/dialect approximations declared | Section 2.1, code/reanalyse.py |
| Files 39–43 as random gap sources | Those stimuli are NG R L W Y; use marked wait intervals | Section 2.3 |
| One final phoneme per word presented as all phonemes | Primary all-phoneme design; final-only sensitivity | Sections 2.3 and 3.3 |
| Condition-dependent contraction splitting | Same word token stream in both conditions | Section 2.3 |
| Missing hand-transcribed probabilistic fixture | Frozen public CMUdict and declared BCIs/streetside overrides | inputs/pronunciations.json |
| Repeated sample and trial-indexing defects | Marker-based blocks and complete 125-sample windows | Sections 2.1 and 2.4 |
| Overlapping Welch claim with 32-sample resolution | Explicit nonoverlapping 125-sample periodograms, 2 Hz bins | Section 2.4 |
| Absolute CMRO2 treated as fractional change | d = C/2.1 − 1 and dimensional CBF conversion | Section 2.5 |
| Gamma estimates measured oxygen metabolism | Dimensionless index primary; imposed physiological proxy | Sections 2.4–2.5 |
| Derived eight outcomes as independent evidence | One deterministic driver; supporting outputs only | Sections 2.5 and 3.5 |
| Channels n=16 or microgap draws as biological n | Conditional simulation variation, no population tests | Section 2.6 |
| 95% CI and exact MC p from treatment draws | Central simulation interval and MCSE; p values withdrawn | Abstract, Section 2.6, Tables 3–4 |
| High rho interpreted as indistinguishable real metabolism | Rank similarity with nonzero error; proxy and reference limits | Sections 2.7 and 3.4 |
| Reference includes reused construction blocks | Disjoint within-session trial roles 1/2/3–5 | Section 2.3 |
| Source operators vary with each recording | One common reference covariance and inverse | Section 2.7 |
| Custom sphere described as validated MNE equivalent | Simplified radial sphere sensitivity; no anatomical validation | Section 2.7, Figure 3 |
| Smoothness or biological pause effects inferred | Duration and window-order controls; effects concern composition | Sections 2.6 and 4 |
| Old paired t tables and inconsistent Trial 2 values remain | All old inferential tables replaced from frozen results | Tables 2–5 |
| Old GUI images treated as corrected evidence | Four new reproducible figures; old animations excluded | Figures 1–4 |
| No live subjects used as sufficient ethics statement | Secondary human EEG provenance and approval items explicit | Declarations |
| Main URL treated as exact reproducible release | Stable approved release identifier still required | Data availability |
| Speculative scalability clinical or truly-random benefits | Removed; concrete independent validation requirements | Discussion and Conclusions |

AI assistance in analysis-code development, manuscript revision, and figure-script preparation is disclosed in Section 2.9 and the figure legends. The disclosure still requires the authors to describe their own independent verification before submission.

## Numerical traceability

### Source mapping

| Manuscript material | Machine-readable source |
| --- | --- |
| Abstract changes and Table 3 | results.json summary where config=primary and variable=CMRO2; subtract 2.1 only for baseline index |
| Table 1 | inputs/sentences.json |
| Table 2 | data_audit.csv, events.csv, stimulus_labels.json, upstream_matches.json |
| Table 4 | simulation_summary.csv where variable=CMRO2 |
| Table 5 | consistency.csv where kind=composite |
| Signal quality | signal_quality.csv |
| S1 constants | results.json constants and code/model.py |
| Figure 2 | results.json summary |
| Figure 3 | spectra.npz and source_maps.npz |
| Figure 4 | public file DLR_0_1.txt and seed-0 assembly manifest |

The mean CMRO2-proxy difference numerically equals the dimensionless index difference because the imposed scale has a slope of 1 μmol/(g·min) per index unit. This identity does not make those outputs independent or validate the physical scale. Raw CBF values in the CSV are mL/(g·min); the manuscript explains the ×100 conversion for the customary per-100-g unit.

## Pending author decisions

Confirm the recording and montage metadata and the pronunciation crosswalk; review source-signal quality; establish participant/ethics and reuse facts; finalize declarations and contact details; approve the substantially narrowed study framing; and deposit the approved reproducible version. A marked-up journal manuscript must use the exact submitted source as its comparison base. The detailed changes above are not a claim that all of these external facts have been resolved.
